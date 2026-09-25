"""Frozen graph candidates on new development partitions with paired controls.

All checkpoint choices come from the historical development partition. Fresh
query outcomes are used only after fixed-step training and saved-model replay.
The three overlapping partitions are not independent biological validation.
"""
import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score
import torch
from torch.nn import functional as F

from improve import tabular_foundation as base
from improve import overlap_block_experiment as block
from improve import lorentz_graph_experiment as geometry
from improve import graph_prediction_precision as precision
from improve import validate_transfer_splits as inputs
from improve import mechanism_validation_inputs as data_source

ROOT=Path(__file__).resolve().parent


def freeze(out,parent,label,primary='full'):
    out,parent=Path(out).resolve(),Path(parent).resolve()
    dp=inputs.protocol(out);p=block.protocol(parent)
    dest=out/'candidates'/(label+'.json')
    if dest.exists():
        raise ValueError('Candidate is already frozen')
    table=block.all_rows(parent)
    selected=base.read_json(parent/'SELECTION.json')
    assert selected['protocol_sha256']==base.digest(parent/'protocol.json')
    seeds=[p['screen_seed']]+p['confirm_seeds']
    recipes={};results={};contrasts=[]
    for task in p['tasks']:
        recipes[task]={}
        for kind in p['kinds']:
            lr=selected['chosen'][kind]['lr']
            t=table[(table.task==task)&(table.kind==kind)&(table.lr==lr)]
            candidate=t[t.score=='anchor_composed'].set_index('seed').loc[seeds]
            reference=t[t.score=='anchor'].set_index('seed').loc[seeds]
            recipes[task][kind]=dict(lr=lr,epochs=int(np.median(candidate.epoch.to_numpy())))
            for seed in seeds:
                path=parent/'runs'/task/kind/('lr'+str(lr))/('s'+str(seed))/'result.json'
                results[str(path)]=base.digest(path)
            contrasts.append(dict(task=task,kind=kind,
                **{m:float((candidate[m]-reference[m]).mean()) for m in ('auc','ap')},
                **{m+'_wins':int((candidate[m]>reference[m]).sum()) for m in ('auc','ap')},
                **{m+'_min':float((candidate[m]-reference[m]).min()) for m in ('auc','ap')}))
    indexed={(r['task'],r['kind']):r for r in contrasts}
    if primary not in p['kinds']:raise ValueError('Unknown primary mechanism')
    mi,ln=indexed['MiDrug',primary],indexed['LncDrug',primary]
    performance=all(mi[m]>0 and mi[m+'_wins']>=2 and ln[m+'_min']>=-.001 for m in ('auc','ap'))
    mechanism=all(mi[m]>indexed['MiDrug',k][m] for k in p['kinds'] if k!=primary for m in ('auc','ap'))
    if not performance or not mechanism:
        raise ValueError('Historical confirmation gate failed; do not promote to fresh partitions')
    registry=out/'STUDY_REGISTRY.json'
    if not registry.exists():
        parents={'block':ROOT/'results/overlap_block_precision_v2_20260915',
                 'geometry':ROOT/'results/lorentz_graph_precision_v2_20260915',
                 'transport':ROOT/'results/multiscale_transport_20260915'}
        base.write_json(registry,base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
            data_protocol_sha256=dp['sha256'],
            registered_candidates={name:dict(path=str(path),protocol_sha256=base.digest(path/'protocol.json'))
                                   for name,path in parents.items()},
            neural_epoch_rule='Per task and matched arm: median selected checkpoint step across historical seeds 666,0,1, at the independently chosen shared learning rate. No epoch selection on fresh queries.',
            fusion_weight=.1,partition_seeds=dp['split_seeds'],
            evaluation_gate='MiDrug mean global AUC/AP increases, each in at least 2/3 partitions. LncDrug every partition loses no more than 0.001 absolute. The development-selected primary mechanism must beat each matched control on mean MiDrug AUC/AP. Endpoint-cluster intervals are reported; this is a descriptive gate, not a significance or biological validation claim.',
            original_test_access=False,fresh_scores_inspected_before_registry=False)))
    own=('graph_mechanism_validate.py','test_graph_mechanism_validate.py')
    cp=base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),label=label,method=p['method'],
        parent=str(parent),parent_protocol_sha256=base.digest(parent/'protocol.json'),
        data_protocol_sha256=dp['sha256'],study_registry_sha256=base.digest(registry),
        recipes=recipes,primary_kind=primary,training_parameters=p,historical_contrasts=contrasts,
        historical_result_sha256=results,model_seed=dp['model_seed'],strong_weight=.1,
        source_sha256={**p['source_sha256'],**{n:base.digest(ROOT/n) for n in own}},
        training_policy='Fixed steps, no fresh query labels or query features passed to optimizer. Original 400-step KL annealing denominator retained when fitting fewer steps. Reinitialize per arm with the same fixed model seed. No fresh-data tuning.',
        historical_performance_gate=performance,historical_mechanism_gate=mechanism,
        fresh_predictions_available=False,original_test_access=False))
    dest.parent.mkdir(parents=True,exist_ok=True);base.write_json(dest,cp)
    snapshot=out/'candidate_sources'/label;snapshot.mkdir(parents=True)
    for n in cp['source_sha256']:shutil.copy2(ROOT/n,snapshot/n)
    print('FROZEN_CANDIDATE',label,recipes,flush=True)


def candidate(out,label):
    out=Path(out);dp=inputs.protocol(out)
    cp=base.check_seal(base.read_json(out/'candidates'/(label+'.json')))
    assert cp['data_protocol_sha256']==dp['sha256']
    assert base.digest(out/'STUDY_REGISTRY.json')==cp['study_registry_sha256']
    for n,sha in cp['source_sha256'].items():
        if base.digest(ROOT/n)!=sha:raise ValueError('Frozen candidate source changed: '+n)
    for path,sha in cp['historical_result_sha256'].items():
        if base.digest(path)!=sha:raise ValueError('Historical selection artifact changed')
    return dp,cp


def verify_artifacts(dest,record):
    for name,sha in record['files'].items():
        if base.digest(Path(dest)/name)!=sha:raise ValueError('Result artifact changed: '+name)


def train_fixed(training,p,kind,recipe,seed,device,method):
    # A training dictionary containing no heldout pairs or outcomes is required.
    if 'y_val' in training or 'pairs_val' in training:
        raise ValueError('Fresh query data must not be passed to train_fixed')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    library=block if method=='block' else geometry
    model,xs,graphs,folds=library.setup(training,p,kind,device)
    pairs=torch.as_tensor(training['pairs_train']+training['pair_offset'][:,None],device=device)
    labels=torch.as_tensor(training['y_train'],device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['lr'],weight_decay=p['weight_decay'])
    generator=torch.Generator(device=device).manual_seed(seed+29015)
    trace=[];start=time.monotonic()
    for step in range(1,recipe['epochs']+1):
        fold=(step-1)%len(graphs);ids=folds[fold]
        model.train();optimizer.zero_grad(set_to_none=True)
        if method=='block':
            pred,kl=model(xs,graphs[fold],pairs[:,ids],generator,p['temperature'])
            loss=F.binary_cross_entropy_with_logits(pred,labels[ids])+p['beta']*min(1.,step/(p['epochs']*.2))*kl
        else:
            pred=model(xs,graphs[fold],pairs[:,ids])
            loss=F.binary_cross_entropy_with_logits(pred,labels[ids])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fresh fit loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%20==0 or step==recipe['epochs']:
            trace.append(dict(step=step,training_loss=float(loss.detach())))
    return model,trace,dict(epochs=recipe['epochs'],seconds=time.monotonic()-start,
                           fresh_query_outcomes_used_for_fitting=False)


def reference(out,label,task,device):
    from tabicl import TabICLClassifier
    from improve.tabicl_context_program_modules import ContextPrograms
    from improve import context_anchor_verify as anchor
    out=Path(out);dp,cp=candidate(out,label)
    evidence=ContextPrograms.load(dp['evidence_dir'])
    torch.cuda.set_device(device)
    for split in dp['split_seeds']:
        a,_=inputs.arrays(out,dp,task,split)
        dest=out/'references'/task/str(split);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():
            r=base.read_json(dest/'result.json');assert r['data_protocol_sha256']==dp['sha256']
            verify_artifacts(dest,r);continue
        prefix='raw' if dp['input_configs'][task]['input_view']=='raw' else 'x'
        arm=dp['graph_recipes'][task]['arm']
        if arm not in ('query_bag','fold_match'):raise ValueError('Unknown fixed reference context')
        graph_scores={};errors=[];start=time.monotonic()
        for variant in ('graph','tiered'):
            components={};vd=dest/variant;vd.mkdir(exist_ok=True)
            et=evidence.transform(task,a['pairs_train'],'tiered') if variant=='tiered' else None
            ev=evidence.transform(task,a['pairs_val'],'tiered') if variant=='tiered' else None
            kinds=['full']+(['fold'+str(i) for i in range(5)] if arm=='fold_match' else [])
            for kind in kinds:
                full=kind=='full'
                ids=np.arange(len(a['y_train'])) if full else np.flatnonzero(a['fold_ids']==int(kind[-1]))
                x=a[prefix+'_train'][ids]
                q=(np.concatenate([a[prefix+'_val'][None],a[prefix+'_query_views']])
                   if full and arm=='query_bag' else a[prefix+'_val'][None] if full
                   else a[prefix+'_query_views'][int(kind[-1]):int(kind[-1])+1])
                if variant=='tiered':
                    x=np.concatenate([x,et[ids]],axis=1).astype(np.float32)
                    q=np.concatenate([q,np.broadcast_to(ev,(len(q),*ev.shape))],axis=2).astype(np.float32)
                model=base.fit_context(base.estimator(dict(n_estimators=dp['members']),
                    dp['reference_seed'],dp['icl_protocol'],device),x,a['y_train'][ids])
                sha=base.parameter_digest(model)
                flat=q.reshape(-1,x.shape[1])
                pred=base.query_scores(model,flat)
                model.save(vd/(kind+'.pkl'),save_model_weights=False,save_training_data=True,save_kv_cache=False)
                del model;gc.collect();torch.cuda.empty_cache()
                model=TabICLClassifier.load(vd/(kind+'.pkl'),device=device)
                assert base.parameter_digest(model)==sha
                replay=base.query_scores(model,flat[::-1].copy())[::-1]
                np.testing.assert_allclose(pred,replay,atol=1e-6,rtol=1e-6)
                errors.append(float(abs(pred-replay).max()))
                components[kind]=pred.reshape(q.shape[:2])
                del model;gc.collect();torch.cuda.empty_cache()
            original=components['full'][0]
            context=(components['full'][1:].mean(0) if arm=='query_bag'
                     else np.stack([components['fold'+str(i)][0] for i in range(5)]).mean(0))
            graph_scores[variant]=.5*original+.5*context
            np.savez_compressed(vd/'components.npz',**components)
        tiered=.5*graph_scores['graph']+.5*graph_scores['tiered']
        member_models={};member_logits=[]
        for name in dp['anchor']['members']:
            model=anchor.learner(name,seed=dp['anchor']['seed'])
            model.fit(a['context76_train'],a['y_train'])
            member_models[name]=model
            member_logits.append(anchor.anchors.logit(model.predict_proba(a['context76_val'])[:,1]))
        with (dest/'anchor_members.pkl').open('wb') as stream:pickle.dump(member_models,stream,protocol=4)
        with (dest/'anchor_members.pkl').open('rb') as stream:loaded=pickle.load(stream)
        replay_logits=np.stack([anchor.anchors.logit(loaded[n].predict_proba(a['context76_val'][::-1])[:,1])[::-1]
                                for n in dp['anchor']['members']]).mean(0)
        mean_logits=np.stack(member_logits).mean(0)
        np.testing.assert_allclose(replay_logits,mean_logits,atol=1e-10,rtol=1e-10)
        scores=dict(y=a['y_val'],pairs=a['pairs_val'],tabicl=anchor.anchors.logit(tiered),
                    anchor=.7*anchor.anchors.logit(tiered)+.3*mean_logits)
        np.savez_compressed(dest/'scores.npz',**scores)
        r=dict(data_protocol_sha256=dp['sha256'],seed=dp['reference_seed'],
               candidate_registry_sha256=base.digest(out/'STUDY_REGISTRY.json'),
               seconds=time.monotonic()-start,reload_reverse_max_abs=max(errors),
               metrics={n:block.base.metrics(scores['y'],scores[n],scores['pairs']) for n in ('tabicl','anchor')},
               files={str(f.relative_to(dest)):base.digest(f) for f in dest.rglob('*') if f.is_file()})
        base.write_json(dest/'result.json',r)
        print('REFERENCE_VERIFIED',task,split,r['metrics'],flush=True)


def run(out,label,task,device):
    out=Path(out);dp,cp=candidate(out,label);p=cp['training_parameters']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    library=block if cp['method']=='block' else geometry
    for split in dp['split_seeds']:
        a,_=inputs.arrays(out,dp,task,split)
        rd=out/'references'/task/str(split)
        rr=base.read_json(rd/'result.json');verify_artifacts(rd,rr)
        assert rr['data_protocol_sha256']==dp['sha256']
        ref=block.load(rd/'scores.npz')
        np.testing.assert_array_equal(a['y_val'],ref['y']);np.testing.assert_array_equal(a['pairs_val'],ref['pairs'])
        training={k:v for k,v in a.items() if k in ('pairs_train','y_train','edges','relation',
                  'target_codes','node_counts','pair_offset','nd','folds') or k in ('x_0','x_1','x_2','x_3')}
        for kind,recipe in cp['recipes'][task].items():
            dest=out/'runs'/label/task/str(split)/kind;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():
                result=base.read_json(dest/'result.json');assert result['candidate_sha256']==cp['sha256']
                verify_artifacts(dest,result);continue
            model,trace,detail=train_fixed(training,p,kind,recipe,cp['model_seed'],device,cp['method'])
            torch.save(model.state_dict(),dest/'model.pt');base.write_json(dest/'trace.json',trace)
            reloaded,xs,graphs,_=library.setup(training,p,kind,device)
            reloaded.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True))
            pv=torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=device)
            def prediction(fitted,query):
                if cp['method']=='block':
                    return precision.block_prediction(fitted,xs,graphs,query,p['posterior_samples']).cpu().numpy()
                return precision.geometry_prediction(fitted,xs,graphs,query).cpu().numpy()
            pred=prediction(model,pv);replay=prediction(reloaded,pv.flip(1))[::-1]
            chunk=np.concatenate([prediction(reloaded,pv[:,i:i+800]) for i in range(0,pv.shape[1],800)])
            for value in (replay,chunk):np.testing.assert_allclose(pred,value,atol=3e-5,rtol=3e-5)
            scores=dict(y=a['y_val'],pairs=a['pairs_val'],standalone=pred,
                        anchor=ref['anchor'],tabicl=ref['tabicl'])
            for name in ('anchor','tabicl'):
                scores[name+'_composed']=(1-cp['strong_weight'])*scores[name]+cp['strong_weight']*pred
            np.savez_compressed(dest/'scores.npz',**scores)
            result=dict(candidate_sha256=cp['sha256'],data_protocol_sha256=dp['sha256'],
                task=task,split=split,kind=kind,seed=cp['model_seed'],recipe=recipe,**detail,
                replay_max_abs=float(max(abs(pred-replay).max(),abs(pred-chunk).max())),
                metrics={n:block.base.metrics(scores['y'],v,scores['pairs']) for n,v in scores.items() if n not in ('y','pairs')},
                files={n:base.digest(dest/n) for n in ('scores.npz','model.pt','trace.json')})
            base.write_json(dest/'result.json',result)
            print('FRESH_VERIFIED',label,task,split,kind,result['metrics']['anchor_composed'],flush=True)
            del model,reloaded,xs,graphs;gc.collect();torch.cuda.empty_cache()


def endpoint_intervals(cells,repeats=1000,seed=20260915,primary='full'):
    """Paired two-way pigeonhole bootstrap, shared across overlapping splits.

    One RNA and one drug resample defines weights in every split. This does
    not falsely treat an entity or pair seen in several splits as independent.
    It remains conditional on the fitted models and the chosen development
    surfaces, and does not correct historical model-selection bias.
    """
    rng=np.random.default_rng(seed)
    rnas=np.unique(np.concatenate([c['pairs'][0] for c in cells]))
    drugs=np.unique(np.concatenate([c['pairs'][1] for c in cells]))
    maps=[(np.searchsorted(rnas,c['pairs'][0]),np.searchsorted(drugs,c['pairs'][1])) for c in cells]
    names=list(cells[0]['scores']);controls=[n for n in names if n!=primary]
    samples={n:{m:[] for m in ('auc','ap')} for n in controls}
    for iteration in range(repeats):
        wr=np.bincount(rng.integers(0,len(rnas),len(rnas)),minlength=len(rnas))
        wd=np.bincount(rng.integers(0,len(drugs),len(drugs)),minlength=len(drugs))
        delta={n:{m:[] for m in ('auc','ap')} for n in controls};valid=True
        for cell,(ri,di) in zip(cells,maps):
            w=wr[ri]*wd[di];y=cell['y']
            if not all(w[y==value].sum()>0 for value in (0,1)):
                valid=False;break
            mm={n:dict(auc=roc_auc_score(y,s,sample_weight=w),
                       ap=average_precision_score(y,s,sample_weight=w)) for n,s in cell['scores'].items()}
            for n in controls:
                for m in ('auc','ap'):delta[n][m].append(mm[primary][m]-mm[n][m])
        if valid:
            for n in controls:
                for m in ('auc','ap'):samples[n][m].append(float(np.mean(delta[n][m])))
    return dict(method='Paired two-way endpoint pigeonhole bootstrap; same endpoint draws across all overlapping splits; mean per-split metric difference.',
        requested_resamples=repeats,valid_resamples=len(samples[controls[0]]['auc']),
        unique_rnas=len(rnas),unique_drugs=len(drugs),
        selection_bias_corrected=False,new_biological_validation=False,
        intervals={n:{m:dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)),
                               positive_fraction=float(np.mean(np.asarray(v)>0)))
                       for m,v in scores.items()} for n,scores in samples.items()})


def report(out,label):
    out=Path(out);dp,cp=candidate(out,label);rows=[];uncertainty={};files={}
    primary=cp['primary_kind']
    for task in dp['tasks']:
        cells=[]
        for split in dp['split_seeds']:
            cell=None
            for kind in cp['recipes'][task]:
                dest=out/'runs'/label/task/str(split)/kind
                if not (dest/'result.json').exists():raise ValueError('Incomplete registered fit: '+str(dest))
                r=base.read_json(dest/'result.json');assert r['candidate_sha256']==cp['sha256']
                verify_artifacts(dest,r);z=block.load(dest/'scores.npz')
                files[str(dest/'result.json')]=base.digest(dest/'result.json')
                if cell is None:cell=dict(y=z['y'],pairs=z['pairs'],scores={'anchor':z['anchor']})
                np.testing.assert_array_equal(cell['y'],z['y']);np.testing.assert_array_equal(cell['pairs'],z['pairs'])
                cell['scores'][kind]=z['anchor_composed']
                for name,expected in r['metrics'].items():
                    actual=block.base.metrics(z['y'],z[name],z['pairs'])
                    for metric in ('auc','ap','macro_drug_ap'):
                        assert abs(actual[metric]-expected[metric])<1e-12
                    rows.append(dict(task=task,split=split,kind=kind,score=name,**actual))
            cells.append(cell)
        uncertainty[task]=endpoint_intervals(cells,primary=primary)
        print('UNCERTAINTY_COMPLETE',label,task,flush=True)
    df=pd.DataFrame(rows);dest=out/'reports'/label;dest.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest/'metrics.csv',index=False)
    differences=[]
    for task in dp['tasks']:
        for split in dp['split_seeds']:
            t=df[(df.task==task)&(df.split==split)&(df.score=='anchor_composed')].set_index('kind')
            anchor=df[(df.task==task)&(df.split==split)&(df.kind==primary)&(df.score=='anchor')].iloc[0]
            for control in ['anchor']+[k for k in t.index if k!=primary]:
                reference=anchor if control=='anchor' else t.loc[control]
                differences.append(dict(task=task,split=split,control=control,
                    **{m:float(t.loc[primary,m]-reference[m]) for m in ('auc','ap','macro_drug_ap')}))
    delta=pd.DataFrame(differences);delta.to_csv(dest/'contrasts.csv',index=False)
    summary=delta.groupby(['task','control'])[['auc','ap','macro_drug_ap']].mean()
    summary.to_csv(dest/'summary.csv')
    mi=delta[(delta.task=='MiDrug')&(delta.control=='anchor')]
    ln=delta[(delta.task=='LncDrug')&(delta.control=='anchor')]
    performance=all(mi[m].mean()>0 and (mi[m]>0).sum()>=2 and ln[m].min()>=-.001 for m in ('auc','ap'))
    mechanism=all(summary.loc[('MiDrug',k),m]>0 for k in cp['recipes']['MiDrug'] if k!=primary for m in ('auc','ap'))
    base.write_json(dest/'endpoint_uncertainty.json',uncertainty)
    assessment=dict(candidate_sha256=cp['sha256'],data_protocol_sha256=dp['sha256'],primary_kind=primary,
        created_at_utc=datetime.now(timezone.utc).isoformat(),performance_gate=bool(performance),
        mechanism_gate=bool(mechanism),fresh_development_gate=bool(performance and mechanism),
        partitions_per_task=3,initializations_per_partition=1,all_metrics_recomputed=True,
        original_test_access=False,independent_biological_validation=False,
        confirmed_final_innovations=0,
        limitations=['Overlapping repartitions of previously used data, not independent biological cohorts.',
                     'Intervals are conditional on selected models and exclude selection/training uncertainty.',
                     'A 0.001 LncDrug tolerance is descriptive, not a demonstrated noise level or noninferiority test.',
                     'Final reserved evaluation has not been performed.'],
        result_sha256=files,
        artifacts={n:base.digest(dest/n) for n in ('metrics.csv','contrasts.csv','summary.csv','endpoint_uncertainty.json')})
    base.write_json(dest/'ASSESSMENT.json',assessment)
    print(summary.to_string(),flush=True)
    print('FRESH_GATES',performance,mechanism,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','reference','run','report'))
    parser.add_argument('--out',default=str(data_source.OUT))
    parser.add_argument('--label',choices=('block','geometry'),required=True)
    parser.add_argument('--parent')
    parser.add_argument('--primary-kind',default='full')
    parser.add_argument('--task',choices=('MiDrug','LncDrug'))
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.command=='freeze':
        if not args.parent:parser.error('freeze requires --parent')
        freeze(args.out,args.parent,args.label,args.primary_kind)
    elif args.command=='reference':reference(args.out,args.label,args.task,args.device)
    elif args.command=='run':run(args.out,args.label,args.task,args.device)
    else:report(args.out,args.label)
