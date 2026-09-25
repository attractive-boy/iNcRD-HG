"""Versioned numerical correction and resumable shared-reference construction.

The sealed validation executor is preserved. Its MLP reverse-query tolerance
was below float32 arithmetic noise. This executor uses unchanged fitted MLP
weights with stable matrix arithmetic while preserving native sigmoid dtype.
Cached TabICL fits are loaded, their fitting data checked, and their saved
predictions replayed; completed fitting is never repeated for this correction.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import torch

from improve import graph_mechanism_validate as validation
from improve import context_anchor_verify as anchor
from improve import anchor_prediction_precision as precision
from improve import tabular_foundation as base


def array_hash(x):
    x=np.ascontiguousarray(x)
    digest=hashlib.sha256()
    digest.update(str(x.shape).encode());digest.update(str(x.dtype).encode());digest.update(x.tobytes())
    return digest.hexdigest()


def prepare(out):
    out=Path(out);dp=validation.inputs.protocol(out)
    path=out/'REFERENCE_NUMERICS_V2.json'
    if path.exists():raise ValueError('Reference correction is already frozen')
    own=('mechanism_reference_precision.py','anchor_prediction_precision.py','test_anchor_prediction_precision.py')
    p=base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),data_protocol_sha256=dp['sha256'],
        source_sha256={n:base.digest(validation.ROOT/n) for n in own},
        parent_executor_sha256=base.digest(validation.ROOT/'graph_mechanism_validate.py'),
        reason='Native float32 MLP yielded 2.855597327489079e-7 max reverse-query logit difference; the mean of four members differed by 7.13899331e-8, exceeding the unchanged 1e-10 tolerance.',
        correction='Only fitted MLP affine arithmetic and standardized inputs are evaluated in float64. Final logits return to the fitted float32 dtype before the original sigmoid, preserving its saturation and probability quantization. No parameters, member recipe, coefficient, partition or candidate choices are changed.',
        recovered_pair_check='On the failed reference: reverse-query mean-logit error 4.440892098500626e-16; max mean-logit difference versus native arithmetic 1.2102796512536429e-5. Metric differences are audited for every partition.',
        reuse='Load saved TabICL estimators and anchor members. Check serialized fitting X/y and reverse-query replay against cached predictions. Persist each successful replay independently.',
        tolerances=dict(tabicl_probability=1e-6,anchor_logit=1e-10),
        fixed_reference_seed=dp['reference_seed'],original_test_access=False,
        architecture_or_hyperparameters_selected_on_fresh_outcomes=False))
    source=out/'reference_sources_v2';source.mkdir()
    for n in own:shutil.copy2(validation.ROOT/n,source/n)
    base.write_json(path,p)
    print('FROZEN_REFERENCE_NUMERICS',p['sha256'],flush=True)


def protocol(out):
    out=Path(out);dp=validation.inputs.protocol(out)
    p=base.check_seal(base.read_json(out/'REFERENCE_NUMERICS_V2.json'))
    assert p['data_protocol_sha256']==dp['sha256']
    assert p['parent_executor_sha256']==base.digest(validation.ROOT/'graph_mechanism_validate.py')
    for n,sha in p['source_sha256'].items():
        if base.digest(validation.ROOT/n)!=sha:raise ValueError('Frozen reference correction changed')
    return dp,p


def run(out,label,task,device):
    from tabicl import TabICLClassifier
    from improve.tabicl_context_program_modules import ContextPrograms
    out=Path(out);dp,p=protocol(out);validation.candidate(out,label)
    evidence=ContextPrograms.load(dp['evidence_dir']);torch.cuda.set_device(device)
    for split in dp['split_seeds']:
        a,_=validation.inputs.arrays(out,dp,task,split)
        dest=out/'references'/task/str(split);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():
            r=base.read_json(dest/'result.json');assert r['reference_numerics_sha256']==p['sha256']
            validation.verify_artifacts(dest,r);continue
        prefix='raw' if dp['input_configs'][task]['input_view']=='raw' else 'x'
        arm=dp['graph_recipes'][task]['arm'];start=time.monotonic()
        graph_scores={};errors=[];parameter_hashes=[];reuse_count=0
        for variant in ('graph','tiered'):
            vd=dest/variant;vd.mkdir(exist_ok=True)
            cached=validation.block.load(vd/'components.npz') if (vd/'components.npz').exists() else {}
            components={}
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
                    x=np.concatenate([x,et[ids]],1).astype(np.float32)
                    q=np.concatenate([q,np.broadcast_to(ev,(len(q),*ev.shape))],2).astype(np.float32)
                flat=q.reshape(-1,x.shape[1]);path=vd/(kind+'.pkl')
                record_path=vd/(kind+'_replay.json');prediction_path=vd/(kind+'_scores.npz')
                input_hashes={n:array_hash(value) for n,value in dict(x=x,y=a['y_train'][ids],q=flat).items()}
                if record_path.exists():
                    record=base.read_json(record_path)
                    assert record['input_hashes']==input_hashes
                    assert record['reference_numerics_sha256']==p['sha256']
                    assert record['model_sha256']==base.digest(path)
                    assert record['predictions_sha256']==base.digest(prediction_path)
                    pred=validation.block.load(prediction_path)['pred']
                    errors.append(record['replay_max_abs']);parameter_hashes.append(record['parameter_sha256'])
                    components[kind]=pred.reshape(q.shape[:2]);reuse_count+=1
                    continue
                reused=path.exists()
                if reused:
                    model=TabICLClassifier.load(path,device=device)
                else:
                    model=base.fit_context(base.estimator(dict(n_estimators=dp['members']),dp['reference_seed'],
                        dp['icl_protocol'],device),x,a['y_train'][ids])
                    model.save(path,save_model_weights=False,save_training_data=True,save_kv_cache=False)
                np.testing.assert_array_equal(model.ensemble_generator_.X_,model.X_encoder_.transform(x))
                np.testing.assert_array_equal(model.ensemble_generator_.y_,a['y_train'][ids].astype(int))
                parameter_sha=base.parameter_digest(model)
                if kind in cached:
                    pred=cached[kind].reshape(-1)
                    assert pred.shape==(len(flat),)
                else:
                    pred=base.query_scores(model,flat)
                del model;gc.collect();torch.cuda.empty_cache()
                reloaded=TabICLClassifier.load(path,device=device)
                assert base.parameter_digest(reloaded)==parameter_sha
                replay=base.query_scores(reloaded,flat[::-1].copy())[::-1]
                np.testing.assert_allclose(pred,replay,atol=1e-6,rtol=1e-6)
                error=float(abs(pred-replay).max())
                np.savez_compressed(prediction_path,pred=pred)
                base.write_json(record_path,dict(reference_numerics_sha256=p['sha256'],input_hashes=input_hashes,
                    model_sha256=base.digest(path),predictions_sha256=base.digest(prediction_path),
                    parameter_sha256=parameter_sha,replay_max_abs=error,reused_fitted_model=reused,
                    fitting_arrays_exactly_matched=True))
                errors.append(error);parameter_hashes.append(parameter_sha);reuse_count+=int(reused)
                components[kind]=pred.reshape(q.shape[:2])
                print('REFERENCE_CONTEXT_REPLAY',task,split,variant,kind,'reused',reused,'error',error,flush=True)
                del reloaded;gc.collect();torch.cuda.empty_cache()
            original=components['full'][0]
            context=(components['full'][1:].mean(0) if arm=='query_bag'
                     else np.stack([components['fold'+str(i)][0] for i in range(5)]).mean(0))
            graph_scores[variant]=.5*original+.5*context
            if not (vd/'components.npz').exists():np.savez_compressed(vd/'components.npz',**components)
        assert len(set(parameter_hashes))==1
        tiered=.5*graph_scores['graph']+.5*graph_scores['tiered']
        member_path=dest/'anchor_members.pkl';reused_members=member_path.exists()
        if reused_members:
            with member_path.open('rb') as stream:members=pickle.load(stream)
        else:
            members={}
            for name in dp['anchor']['members']:
                model=anchor.learner(name,seed=dp['anchor']['seed']).fit(a['context76_train'],a['y_train'])
                members[name]=model
            with member_path.open('wb') as stream:pickle.dump(members,stream,protocol=4)
        member_hash=base.digest(member_path)
        native=np.stack([anchor.anchors.logit(members[n].predict_proba(a['context76_val'])[:,1])
                         for n in dp['anchor']['members']]).mean(0)
        calibrated=precision.member_mean(members,dp['anchor']['members'],a['context76_val'])
        with member_path.open('rb') as stream:reloaded_members=pickle.load(stream)
        replay=precision.member_mean(reloaded_members,dp['anchor']['members'],a['context76_val'][::-1])[::-1]
        np.testing.assert_allclose(calibrated,replay,atol=1e-10,rtol=1e-10)
        assert base.digest(member_path)==member_hash
        tiered_logit=anchor.anchors.logit(tiered)
        scores=dict(y=a['y_val'],pairs=a['pairs_val'],tabicl=tiered_logit,
                    anchor=.7*tiered_logit+.3*calibrated,anchor_native=.7*tiered_logit+.3*native)
        np.savez_compressed(dest/'scores.npz',**scores)
        shutil.copy2(out/'REFERENCE_NUMERICS_V2.json',dest/'reference_numerics.json')
        metrics={n:validation.block.base.metrics(scores['y'],scores[n],scores['pairs']) for n in ('tabicl','anchor','anchor_native')}
        r=dict(data_protocol_sha256=dp['sha256'],reference_numerics_sha256=p['sha256'],seed=dp['reference_seed'],
            candidate_registry_sha256=base.digest(out/'STUDY_REGISTRY.json'),seconds=time.monotonic()-start,
            reused_tabicl_models=reuse_count,reused_anchor_members=reused_members,
            reload_reverse_max_abs=max(errors),anchor_reverse_max_abs=float(abs(calibrated-replay).max()),
            numerical_audit=dict(max_anchor_logit_change=float(abs(scores['anchor']-scores['anchor_native']).max()),
                metric_changes={m:metrics['anchor'][m]-metrics['anchor_native'][m] for m in ('auc','ap')}),
            metrics=metrics,files={str(f.relative_to(dest)):base.digest(f) for f in dest.rglob('*') if f.is_file()})
        base.write_json(dest/'result.json',r)
        print('REFERENCE_VERIFIED',task,split,metrics,'numerical audit',r['numerical_audit'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--out',default=str(validation.data_source.OUT))
    parser.add_argument('--label',default='block');parser.add_argument('--task',choices=('MiDrug','LncDrug'))
    parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    if args.command=='prepare':prepare(args.out)
    else:run(args.out,args.label,args.task,args.device)
