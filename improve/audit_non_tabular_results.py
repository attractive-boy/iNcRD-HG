"""Recompute all non-tabular outcomes and grouped uncertainty from saved scores."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

from improve import non_tabular_experiment as base
from improve import confirm_non_tabular_kernel as kernel
from improve import typed_sheaf_experiment as sheaf
from improve import graph_signal_experiment as signal
from improve import graph_mask_resampling as masking

OUT=base.ROOT/'results'/'non_tabular_audit_20260914'


def audit_screen(folder,loader,kind):
    p=loader(folder);rows=[];inputs={};run_count=0;seconds=0.;max_error=0.
    for task in p['tasks']:
        for cfg in p['configs']:
            for seed in p['seeds']:
                dest=Path(folder)/'screen'/task/cfg['id']/str(seed)
                r=json.loads((dest/'result.json').read_text())
                assert (r['task'],r['seed'],r['config'])==(task,seed,cfg)
                for n,h in r['files'].items():
                    if base.digest(dest/n)!=h:raise ValueError('Artifact changed: '+str(dest/n))
                inputs[str(dest/'result.json')]=base.digest(dest/'result.json')
                inputs[str(dest/'scores.npz')]=base.digest(dest/'scores.npz')
                with np.load(dest/'scores.npz',allow_pickle=False) as z:
                    variants={'pred':r} if kind in ('direct','sheaf') else (
                        {'pred':r['primary'],'full_support':r['full_support']} if kind=='masking' else r['metrics'])
                    for score_name,stored in variants.items():
                        m=base.metrics(z['y'],z[score_name],z['pairs'])
                        for key,value in m.items():
                            if abs(value-stored[key])>1e-10:raise ValueError('Metric mismatch: '+str(dest))
                        reference=base.GRAPH/('screen' if seed==666 else 'confirm')/task/'base/c0'/('s'+str(seed))/'scores.npz'
                        with np.load(reference,allow_pickle=False) as g:
                            np.testing.assert_array_equal(g['pairs_val'],z['pairs'])
                            np.testing.assert_array_equal(g['y_val'],z['y'])
                            ref=base.metrics(g['y_val'],g['val'],g['pairs_val'])
                        rows.append(dict(experiment=kind,task=task,config=cfg['id'],score=score_name,seed=seed,
                                         epoch=r['epoch'],**m,gnn_auc=ref['auc'],gnn_ap=ref['ap'],
                                         delta_auc=m['auc']-ref['auc'],delta_ap=m['ap']-ref['ap']))
                run_count+=1;seconds+=r['seconds'];max_error=max(max_error,r.get('replay_max_abs',0.))
    return rows,inputs,dict(runs=run_count,fit_seconds=seconds,max_replay_error=max_error)


def paired_metrics(y,new,old,w):
    if any(w[y==c].sum()<=0 for c in (0,1)):return None
    return np.array([roc_auc_score(y,new,sample_weight=w)-roc_auc_score(y,old,sample_weight=w),
                     average_precision_score(y,new,sample_weight=w)-average_precision_score(y,old,sample_weight=w)])


def kernel_bootstrap(repetitions=1000):
    """Shared endpoint weights across the overlapping resplits; no retraining.

    Poisson RNA and drug weights are drawn independently, then multiplied at
    each pair. This is a descriptive two-way cluster perturbation interval,
    not an exact finite-sample confidence guarantee or independent-data study.
    """
    p=kernel.protocol(kernel.OUT);result={}
    for task in p['tasks']:
        data=[]
        with np.load(base.OUT/'inputs'/task/'arrays.npz') as z:nr,nd=len(z['ur']),len(z['ud'])
        for split in p['splits']:
            root=kernel.OUT/'runs'/task/str(split)
            with np.load(root/'kernel_1/scores.npz',allow_pickle=False) as z:
                a={n:z[n].copy() for n in ('gnn_composed','gnn','y','pairs')}
            with np.load(root/'additive_control/scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(a['pairs'],z['pairs']);np.testing.assert_array_equal(a['y'],z['y'])
                a['additive']=z['gnn_composed'].copy()
            data.append(a)
        rng=np.random.default_rng(202609141+(task=='MiDrug'))
        draws={'gnn':[],'additive':[]}
        for _ in range(repetitions):
            rw=rng.poisson(1,nr);dw=rng.poisson(1,nd)
            for ref in draws:
                changes=[]
                for a in data:
                    w=rw[a['pairs'][0]]*dw[a['pairs'][1]]
                    d=paired_metrics(a['y'],a['gnn_composed'],a[ref],w)
                    if d is None:break
                    changes.append(d)
                if len(changes)==len(data):draws[ref].append(np.mean(changes,axis=0))
        result[task]={}
        for ref,values in draws.items():
            values=np.stack(values)
            point=np.mean([paired_metrics(a['y'],a['gnn_composed'],a[ref],np.ones(len(a['y']))) for a in data],axis=0)
            result[task][ref]={m:dict(point=float(point[i]),quantile_025=float(np.quantile(values[:,i],.025)),
                quantile_975=float(np.quantile(values[:,i],.975)),resamples=len(values)) for i,m in enumerate(('auc','ap'))}
    return dict(method='Shared RNA Poisson(1) weight times shared drug Poisson(1) weight. Identical endpoint weights across all three resplits in each replicate.',
                repetitions=repetitions,model_refitting=False,
                interpretation='Descriptive two-way endpoint-cluster perturbation interval conditional on fixed models and overlapping partitions; not a new external evaluation or a formal significance claim.',results=result)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',default=str(OUT));ap.add_argument('--include-mask',action='store_true')
    ap.add_argument('--bootstrap',type=int,default=1000);args=ap.parse_args();out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    rows=[];inputs={};audit={}
    jobs=[(base.OUT,base.load_protocol,'direct'),(sheaf.OUT,sheaf.protocol,'sheaf'),(signal.OUT,signal.protocol,'diffusion')]
    if args.include_mask:jobs.append((masking.OUT,masking.protocol,'masking'))
    for folder,loader,kind in jobs:
        r,i,a=audit_screen(folder,loader,kind);rows+=r;inputs.update(i);audit[kind]=a
    df=pd.DataFrame(rows);df.to_csv(out/'inner_metrics.csv',index=False)
    summary=df.groupby(['experiment','task','config','score'])[['auc','ap','macro_drug_ap','delta_auc','delta_ap','epoch']].mean()
    summary.to_csv(out/'inner_summary.csv')
    # Frozen kernel confirmation is separately checked by its original reporter.
    kernel.report(kernel.OUT)
    kdf=pd.read_csv(kernel.OUT/'metrics.csv')
    kdf.groupby(['task','model','kind'])[['auc','ap','macro_drug_ap']].mean().to_csv(out/'kernel_resplit_summary.csv')
    boot=kernel_bootstrap(args.bootstrap);base.write_json(out/'KERNEL_BOOTSTRAP.json',boot)
    base.write_json(out/'AUDIT.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        screen=audit,kernel_confirmation_runs=12,screen_results=sum(x['runs'] for x in audit.values()),
        all_metrics_recomputed=True,all_input_sources_and_artifact_hashes_checked=True,
        uncertainty_method=boot['method'],frontier_innovations_confirmed=0,
        goal_complete=False,original_test_labels_access=False,
        source_sha256=base.digest(__file__),input_artifacts_sha256=inputs,
        output_sha256={n:base.digest(out/n) for n in ('inner_metrics.csv','inner_summary.csv','kernel_resplit_summary.csv','KERNEL_BOOTSTRAP.json')}))
    print(summary.to_string());print(json.dumps(boot,ensure_ascii=False,indent=2));print('AUDITED',out)


if __name__=='__main__':main()
