"""Fixed-recipe cross-RNA transport screening with structural controls."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from improve import overlap_block_experiment as common
from improve import graph_prediction_precision as precision
from improve import multiscale_transport as method

OUT=common.base.ROOT/'results/multiscale_transport_20260915'
KINDS=('full','collapsed','feature_only','no_capacity','independent')


def prepare(out,smoke=False):
    out=Path(out).resolve()
    if out.exists():
        raise ValueError('Use a fresh output directory')
    parent=common.protocol(common.OUT)
    own=('multiscale_transport.py','multiscale_transport_experiment.py',
         'test_multiscale_transport.py','graph_prediction_precision.py',
         'overlap_block_experiment.py','overlap_block_models.py','non_tabular_experiment.py')
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),smoke=smoke,tasks=parent['tasks'],
        kinds=list(KINDS),seed=666,folds=5,levels=4,epsilon=.1,alpha=.5,
        outer_steps=2 if smoke else 40,strong_weight=.1,lnc_absolute_tolerance=.001,
        calibration='StandardScaler plus LogisticRegression(C=1,max_iter=2000) on cross-fitted transfer logit and log1p RNA/drug target support counts. The identical three-column calibrator is used by every control.',
        literature=dict(title='Unsupervised Multi-Scale Gromov-Wasserstein Hypergraph Alignment',
                        arxiv='2608.29635v1',date='2026-08-30',peer_review_verified=False),
        adaptation='Drug neighborhoods become hyperedges; rectangular soft coupling between different RNA populations, no asserted bijection. Four cumulative pooled-quantile degree filtrations and balanced scale weights. Fused shared-drug profile/LncMi edge cost, degree+1 marginals, 40 damped entropic GW iterations, posterior association transfer, calibrated from support-masked training folds. Not the original FALCON alignment benchmark.',
        controls='Full shared multiscale GW; collapsed single-clique GW; feature-only capacity-constrained OT; feature matching without column capacities; independent mass product. Identical calibration, folds, fusion weight and reference.',
        selection='No hyperparameter, initialization or coefficient search. Every registered arm is evaluated; a deterministic operator is not counted as multiple seed replications.',
        gate='MiDrug global AUC/AP > shared anchor; LncDrug both losses <=0.001. Full must exceed each matched control on MiDrug AUC and AP. Still requires new-partition confirmation.',
        boundary='For each support fold, queried target facts are deleted before incidence, degree, profile cost, filtration, transport and calibration features. Auxiliary task relations follow the historical transductive policy. Validation labels never enter transport or calibration fitting.',
        source_sha256={n:common.base.digest(common.base.ROOT/n) for n in own},inputs={},
        original_test_access=False,historical_outer_query_label_access=False,confirmed_innovations=0)
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    for n in own:
        shutil.copy2(common.base.ROOT/n,out/'source_snapshot'/n)
    for task in p['tasks']:
        source=common.OUT/'inputs'/task/'arrays.npz'
        dest=out/'inputs'/task/'arrays.npz';dest.parent.mkdir(parents=True)
        shutil.copy2(source,dest)
        for f in (source,dest):p['inputs'][str(f)]=common.base.digest(f)
    common.base.write_json(out/'protocol.json',p)
    print('FROZEN',out,flush=True)


def features(a,p,kind,device):
    _,auxiliary,bridge=method.association_tables(a)
    train=np.zeros((len(a['y_train']),3),dtype=np.float64)
    validation=[];plans=[];audits=[]
    for fold in range(p['folds']):
        ix=np.flatnonzero(a['folds']==fold)
        support=method.support_matrix(a,ix)
        alternate={**a,'y_train':a['y_train'].copy()}
        alternate['y_train'][ix]=1-alternate['y_train'][ix]
        np.testing.assert_array_equal(support,method.support_matrix(alternate,ix))
        with precision.deterministic_evaluation():
            transfer,plan,detail=method.transport(support,auxiliary,bridge,mode=kind,device=device,
                levels=p['levels'],epsilon=p['epsilon'],alpha=p['alpha'],outer_steps=p['outer_steps'])
        rna_mass=(support.sum(1).astype(np.float64)+1.)
        rna_mass/=rna_mass.sum()
        reconstructed=plan.astype(np.float64)@auxiliary.astype(np.float64)/rna_mass[:,None]
        np.testing.assert_allclose(reconstructed,transfer,atol=2e-7,rtol=2e-6)
        train[ix]=method.calibration_features(transfer,support,a['pairs_train'][:,ix])
        q=method.calibration_features(transfer,support,a['pairs_val'])
        reverse=method.calibration_features(reconstructed,support,a['pairs_val'][:,::-1])[::-1]
        np.testing.assert_allclose(q,reverse,atol=3e-5,rtol=3e-5)
        validation.append(q);plans.append(plan)
        audits.append(dict(fold=fold,query_fact_overlap=0,query_outcome_invariance=True,
                           stored_plan_replay_max_abs=float(abs(reconstructed-transfer).max()),**detail))
        print('TRANSPORT',kind,fold,'iterations',len(detail['trace']),
              'converged',detail['converged'],'column error',detail['column_marginal_l1'],flush=True)
    return train,np.stack(validation),np.stack(plans),audits


def run(out,task,device):
    out=Path(out).resolve();p=common.protocol(out)
    a=common.load(out/'inputs'/task/'arrays.npz')
    torch.set_num_threads(4)
    for kind in p['kinds']:
        dest=out/'runs'/task/kind
        if (dest/'result.json').exists():
            common.verify(dest,common.base.digest(out/'protocol.json'))
            continue
        dest.mkdir(parents=True,exist_ok=True);start=time.monotonic()
        x,q,plans,audits=features(a,p,kind,device)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000))
        model.fit(x,a['y_train'])
        score=logit(np.clip(model.predict_proba(q.reshape(-1,3))[:,1].reshape(q.shape[:2]).mean(0),1e-7,1-1e-7))
        with (dest/'calibrator.pkl').open('wb') as stream:pickle.dump(model,stream,protocol=4)
        with (dest/'calibrator.pkl').open('rb') as stream:reloaded=pickle.load(stream)
        replay=logit(np.clip(reloaded.predict_proba(q.reshape(-1,3))[:,1].reshape(q.shape[:2]).mean(0),1e-7,1-1e-7))
        np.testing.assert_allclose(replay,score,atol=1e-12,rtol=1e-12)
        scores=dict(y=a['y_val'],pairs=a['pairs_val'],standalone=score)
        for name in ('anchor','tabicl'):
            scores[name]=a[name+'_'+str(p['seed'])]
            scores[name+'_composed']=(1-p['strong_weight'])*scores[name]+p['strong_weight']*score
        np.savez_compressed(dest/'scores.npz',**scores)
        np.savez_compressed(dest/'operator.npz',plans=plans,calibration_train=x,calibration_val=q)
        common.base.write_json(dest/'transport_audit.json',audits)
        result=dict(task=task,kind=kind,seconds=time.monotonic()-start,
            protocol_sha256=common.base.digest(out/'protocol.json'),
            metrics={k:common.base.metrics(scores['y'],v,scores['pairs']) for k,v in scores.items() if k not in ('y','pairs')},
            files={n:common.base.digest(dest/n) for n in ('scores.npz','operator.npz','calibrator.pkl','transport_audit.json')})
        common.base.write_json(dest/'result.json',result)
        print('VERIFIED',task,kind,result['metrics']['anchor_composed'],result['seconds'],flush=True)


def report(out):
    import pandas as pd
    out=Path(out);p=common.protocol(out);rows=[]
    ph=common.base.digest(out/'protocol.json')
    for task in p['tasks']:
        for kind in p['kinds']:
            dest=out/'runs'/task/kind
            if not (dest/'result.json').exists():continue
            result=common.verify(dest,ph);z=common.load(dest/'scores.npz')
            for score in result['metrics']:
                actual=common.base.metrics(z['y'],z[score],z['pairs'])
                for metric in ('auc','ap'):
                    assert abs(actual[metric]-result['metrics'][score][metric])<1e-12
                rows.append(dict(task=task,kind=kind,score=score,**actual))
    df=pd.DataFrame(rows);df.to_csv(out/'metrics.csv',index=False)
    complete=len(df)==len(p['tasks'])*len(p['kinds'])*5
    gate=mechanism=False
    if complete:
        subset=df[df.score=='anchor_composed'].set_index(['task','kind'])
        reference=df[(df.score=='anchor') & (df.kind=='full')].set_index('task')
        gate=all(subset.loc[('MiDrug','full'),m]>reference.loc['MiDrug',m] and
                 subset.loc[('LncDrug','full'),m]>=reference.loc['LncDrug',m]-p['lnc_absolute_tolerance'] for m in ('auc','ap'))
        mechanism=all(subset.loc[('MiDrug','full'),m]>subset.loc[('MiDrug',kind),m]
                      for kind in p['kinds'][1:] for m in ('auc','ap'))
    common.base.write_json(out/'ASSESSMENT.json',dict(complete=complete,strong_reference_gate=bool(gate),
        mechanism_gate=bool(mechanism),smoke=p['smoke'],confirmed_innovations=0,
        fresh_partitions_required=True,original_test_access=False))
    print(df[df.score=='anchor_composed'].to_string(index=False),flush=True)
    print('GATES',gate,mechanism,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--out',default=str(OUT));parser.add_argument('--task',choices=common.base.TASKS)
    parser.add_argument('--device',default='cuda:1');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.command=='prepare':prepare(args.out,args.smoke)
    elif args.command=='run':run(args.out,args.task,args.device)
    else:report(args.out)
