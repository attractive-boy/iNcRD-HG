"""Fixed graph-foundation kernel confirmation on three development resplits."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.special import logit
import torch

from improve import non_tabular_experiment as base
from improve import reification_transfer as graph
from improve import reification_kernel as method
from improve.graph_signal_calibrated import make_folds

OUT=base.ROOT/'results'/'reification_kernel_confirmation_20260914'
SPLITS=base.ROOT/'results'/'transfer_splits_20260914'
KERNEL=base.ROOT/'results'/'non_tabular_kernel_confirmation_20260914'


def graph_arrays(g,task,a):
    offsets={};n=0
    for nt in g.node_types:offsets[nt]=n;n+=g[nt].num_nodes
    rna='miRNA' if task=='MiDrug' else 'lncRNA'
    edges=[];rels=[];codes=[];keys=[]
    for (s,name,t),index in g.edge_index_dict.items():
        if name.startswith('rev_'):continue
        e=index.cpu().numpy()
        if s==t:e=e[:,e[0]<e[1]]
        if not e.shape[1]:continue
        c=e[0]*g['drug'].num_nodes+e[1] if (s,name,t)==(rna,task,'drug') else np.full(e.shape[1],-1,dtype=np.int64)
        edges.append(e+np.array([offsets[s],offsets[t]])[:,None])
        rels.append(np.full(e.shape[1],len(keys),dtype=np.int64));codes.append(c);keys.append([s,name,t])
    e,r,c=graph.facts_from_physical(np.concatenate(edges,1),np.concatenate(rels),np.concatenate(codes),keys)
    b=dict(edges=e,relation=r,target_codes=c,num_entities=np.array(n),num_relations=np.array(len(keys)),
           target_relation=np.array([x[1] for x in keys].index(task)),
           pair_offset=np.array([offsets[rna],offsets['drug']]),nd=np.array(g['drug'].num_nodes),nr=np.array(g[rna].num_nodes))
    positive=a['pairs_train'][:,a['y_train']==1]
    if set(c[c>=0])!=set(positive[0]*int(b['nd'])+positive[1]):raise ValueError('Fitting graph and sampled positive pairs differ')
    q=a['pairs_val'];qc=q[0]*int(b['nd'])+q[1]
    if np.isin(qc,c).any():raise ValueError('Evaluation query remains in graph')
    return b


def prepare(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new confirmation directory')
    parent=method.protocol(method.OUT)
    selected=json.loads((method.OUT/'ASSESSMENT.json').read_text())
    if not selected['shared_gate']:raise ValueError('Historical screen did not pass')
    names=('confirm_reification_kernel.py','reification_kernel.py','reification_transfer.py','non_tabular_models.py','non_tabular_experiment.py')
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),tasks=list(base.TASKS),splits=[42,2026,31415],
           kinds=['kernel','pretrained','random'],seed=parent['seed'],folds=parent['folds'],batch_size=8,
           epochs=parent['epochs'],lr=parent['lr'],
           primary='pretrained: 0.5 GNN logit + 0.5 augmented-kernel logit; same shared modification for both tasks.',
           frozen='Same checkpoint, feature readout, normalization, regularization, epochs and orientation as historical screen. No selection, early stopping or coefficient search on these resplits.',
           graph_policy='Read each resplit actual fitting graph; exclude all OOF query facts before six-edge reification. Query-negative positions are not support edges. No original test or outer-query outcome access.',
           selection_history='These three overlapping development partitions were used for other methods. They are not untouched biological/external tests.',
           gate='MiDrug mean AUC/AP > GNN with >=2/3 positive split signs each. Both controls (kernel and random) must be exceeded in both mean MiDrug metrics, with >=2/3 positive signs. LncDrug mean and every split lose <=0.001 absolute in each metric.',
           comparisons='Standalone, GNN composition, and fixed half-logit composition with the already-composed strong TabICL reference are all retained. Comparator mixtures are not separate innovations.',
           encoder_paths={kind:str(graph.OUT/'runs'/'MiDrug'/kind/'model.pt') for kind in ('pretrained','random')},
           source_sha256={n:base.digest(base.ROOT/n) for n in names},input_sha256={},
           original_test_access=False,outer_query_labels_access=False)
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    for name in names:shutil.copy2(base.ROOT/name,out/'source_snapshot'/name)
    for path in (method.OUT/'protocol.json',method.OUT/'ASSESSMENT.json',*[Path(x) for x in p['encoder_paths'].values()]):
        p['input_sha256'][str(path)]=base.digest(path)
    for task in p['tasks']:
        nodes=base.OUT/'inputs'/task/'arrays.npz'
        with np.load(nodes,allow_pickle=False) as z:node={n:z[n].copy() for n in ('ur','ud')}
        p['input_sha256'][str(nodes)]=base.digest(nodes)
        for split in p['splits']:
            src=SPLITS/'inputs'/task/str(split)
            with np.load(src/'arrays.npz',allow_pickle=False) as z:a={n:z[n].copy() for n in ('pairs_train','y_train','pairs_val','y_val')}
            g=torch.load(src/'graph.pt',map_location='cpu',weights_only=False)
            a.update(graph_arrays(g,task,a));a.update(node)
            a['folds']=make_folds(a['pairs_train'],a['y_train'],p['seed'],p['folds'])
            paths=[src/'arrays.npz',src/'graph.pt']
            for name,arm in [('gnn','gnn'),('tabicl','tiered')]:
                path=SPLITS/'runs'/task/str(split)/arm/'scores.npz'
                with np.load(path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(a['pairs_val'],z['pairs']);np.testing.assert_array_equal(a['y_val'],z['y']);a[name]=z['pred'].copy()
                paths.append(path)
            path=KERNEL/'runs'/task/str(split)/'kernel_1'/'scores.npz'
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(a['pairs_val'],z['pairs']);a['kernel_expected']=z['standalone'].copy()
            paths.append(path)
            dest=out/'inputs'/task/str(split)/'arrays.npz';dest.parent.mkdir(parents=True)
            np.savez_compressed(dest,**a);paths.append(dest)
            for path in paths:p['input_sha256'][str(path)]=base.digest(path)
    base.write_json(out/'protocol.json',p);print('FROZEN',out,flush=True)


def protocol(out):
    method.protocol(method.OUT)
    p=json.loads((Path(out)/'protocol.json').read_text())
    for name,h in p['source_sha256'].items():
        if base.digest(base.ROOT/name)!=h:raise ValueError('Source changed: '+name)
    for name,h in p['input_sha256'].items():
        if base.digest(name)!=h:raise ValueError('Input changed: '+name)
    return p


def encoder(kind,p,device):
    m=graph.model_for(kind,p['seed'],device)
    m.load_state_dict(torch.load(p['encoder_paths'][kind],map_location=device,weights_only=True),strict=True)
    return m.eval()


def load_inputs(out,task,split):
    with np.load(Path(out)/'inputs'/task/str(split)/'arrays.npz',allow_pickle=False) as z:
        a={n:z[n].copy() for n in z.files}
    for part in ('train','val'):
        q,y=a['pairs_'+part],a['y_'+part]
        if q.shape!=(2,len(y)) or not np.isin(y,[0,1]).all():
            raise ValueError('Invalid query or label array')
        if np.any(q<0) or np.any(q[0]>=int(a['nr'])) or np.any(q[1]>=int(a['nd'])):
            raise ValueError('Query outside frozen dictionary')
        codes=q[0]*int(a['nd'])+q[1]
        if len(np.unique(codes))!=len(codes):raise ValueError('Duplicate query')
    train=a['pairs_train'][0]*int(a['nd'])+a['pairs_train'][1]
    val=a['pairs_val'][0]*int(a['nd'])+a['pairs_val'][1]
    if np.intersect1d(train,val).size:raise ValueError('Train/evaluation overlap')
    if not np.isfinite(a['gnn']).all() or not np.all((a['tabicl']>0)&(a['tabicl']<1)):
        raise ValueError('Invalid reference logits/probabilities')
    return a


def run(out,task,device):
    out=Path(out);p=protocol(out)
    if task not in p['tasks']:raise ValueError('Specify a task')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol_hash=base.digest(out/'protocol.json')
    for split in p['splits']:
        a=load_inputs(out,task,split)
        for kind in p['kinds']:
            dest=out/'runs'/task/str(split)/kind
            if (dest/'result.json').exists():
                row=json.loads((dest/'result.json').read_text())
                if row['protocol_sha256']!=protocol_hash:raise ValueError('Run protocol changed')
                graph.verify_files(dest,row);continue
            dest.mkdir(parents=True,exist_ok=True)
            records=[]
            if kind=='kernel':
                x_train=np.empty((len(a['y_train']),0),dtype=np.float32)
                x_val=np.empty((len(a['y_val']),0),dtype=np.float32)
            else:
                enc=encoder(kind,p,device)
                x_train=np.full((len(a['y_train']),192),np.nan,dtype=np.float32)
                val_features=[]
                for fold in range(p['folds']):
                    path=dest/('features_'+str(fold)+'.npz')
                    record_path=dest/('features_'+str(fold)+'.json')
                    ix=np.flatnonzero(a['folds']==fold)
                    omitted=a['pairs_train'][:,ix]
                    omit_codes=omitted[0]*int(a['nd'])+omitted[1]
                    q=np.concatenate([omitted,a['pairs_val']],axis=1)
                    if record_path.exists():
                        row=json.loads(record_path.read_text())
                        if row['protocol_sha256']!=protocol_hash or base.digest(path)!=row['sha256']:
                            raise ValueError('Cached features or protocol changed')
                    else:
                        start=time.perf_counter()
                        ctx=graph.context_graph(a,omit_codes,device)
                        features,raw=method.extract(enc,ctx,a,q,p['batch_size'])
                        if not np.isfinite(features).all() or not np.isfinite(raw).all():
                            raise FloatingPointError('Nonfinite frozen representations')
                        sample=np.arange(min(17,q.shape[1]))[::-1]
                        check,raw_check=method.extract(enc,ctx,a,q[:,sample],3)
                        np.testing.assert_allclose(check,features[sample],atol=2e-4,rtol=2e-4)
                        np.testing.assert_allclose(raw_check,raw[sample],atol=1e-4,rtol=1e-4)
                        # Independently call the ordinary model score path: the
                        # representation hook must leave its scores unchanged.
                        drugs=np.unique(q[1,sample])
                        ordinary=graph.score_drugs(enc,ctx,a,drugs,3)
                        ordinary_query=ordinary[q[0,sample],np.searchsorted(drugs,q[1,sample])]
                        np.testing.assert_allclose(ordinary_query,raw[sample],atol=1e-4,rtol=1e-4)
                        np.savez_compressed(path,oof=features[:len(ix)],val=features[len(ix):],
                                            oof_indices=ix,omitted_codes=omit_codes)
                        row=dict(task=task,split=split,kind=kind,fold=fold,
                                 protocol_sha256=protocol_hash,seconds=time.perf_counter()-start,
                                 sha256=base.digest(path),facts=int(ctx.num_facts),
                                 oof_count=len(ix),excluded_positive_facts=int(np.isin(a['target_codes'],omit_codes).sum()),
                                 excludes_queries_before_six_edge_reification=True,
                                 feature_replay_max_abs=float(np.max(np.abs(check-features[sample]))),
                                 ordinary_score_replay_max_abs=float(np.max(np.abs(ordinary_query-raw[sample]))))
                        base.write_json(record_path,row)
                        print('FEATURES',task,split,kind,fold,round(row['seconds'],1),flush=True)
                        del ctx
                    with np.load(path,allow_pickle=False) as z:
                        np.testing.assert_array_equal(ix,z['oof_indices'])
                        np.testing.assert_array_equal(omit_codes,z['omitted_codes'])
                        x_train[ix]=z['oof'];val_features.append(z['val'].copy())
                    records.append(row)
                x_val=np.stack(val_features).mean(0)
                if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
                    raise ValueError('Incomplete OOF coverage or invalid features')
                del enc
            model,pred,trace,seconds=method.fit(a,x_train,x_val,p['epochs'][task],p['lr'],device)
            expected_error=None
            if kind=='kernel':
                np.testing.assert_allclose(pred,a['kernel_expected'],atol=5e-5,rtol=5e-5)
                expected_error=float(np.max(np.abs(pred-a['kernel_expected'])))
            torch.save(model.state_dict(),dest/'model.pt')
            loaded=method.TransferKernel(a,x_train).to(device)
            loaded.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True),strict=True)
            loaded.eval()
            with torch.no_grad():
                q=torch.as_tensor(a['pairs_val'],device=device);v=torch.as_tensor(x_val,device=device)
                replay=loaded(q,v).cpu().numpy()
                reverse=loaded(q.flip(1),v.flip(0)).cpu().numpy()[::-1]
                batched=torch.cat([loaded(q[:,ix],v[ix]) for ix in torch.tensor_split(torch.arange(len(v),device=device),7)]).cpu().numpy()
            for value in (replay,reverse,batched):np.testing.assert_allclose(value,pred,atol=5e-5,rtol=5e-5)
            end_to_end_error=0.
            if kind!='kernel':
                enc=encoder(kind,p,device)
                sample=np.arange(min(17,len(a['y_val'])))
                probes=a['pairs_val'][:,sample]
                fresh_features=[]
                for fold in range(p['folds']):
                    omitted=a['pairs_train'][:,a['folds']==fold]
                    ctx=graph.context_graph(a,omitted[0]*int(a['nd'])+omitted[1],device)
                    z,_=method.extract(enc,ctx,a,probes[:,::-1],3)
                    fresh_features.append(z[::-1].copy());del ctx
                with torch.no_grad():
                    fresh=loaded(torch.as_tensor(probes,device=device),
                                 torch.as_tensor(np.stack(fresh_features).mean(0),device=device)).cpu().numpy()
                np.testing.assert_allclose(fresh,pred[sample],atol=5e-4,rtol=5e-4)
                end_to_end_error=float(np.max(np.abs(fresh-pred[sample])))
                del enc
            scores=dict(standalone=pred,gnn=a['gnn'],tabicl=a['tabicl'],
                        gnn_composed=.5*a['gnn']+.5*pred,
                        tabicl_composed=.5*logit(a['tabicl'].astype(np.float64))+.5*pred,
                        y=a['y_val'],pairs=a['pairs_val'])
            measured={n:base.metrics(scores['y'],value,scores['pairs'])
                      for n,value in scores.items() if n not in ('y','pairs')}
            np.savez_compressed(dest/'scores.npz',**scores);base.write_json(dest/'trace.json',trace)
            feature_files=['features_'+str(f)+ext for f in range(len(records)) for ext in ('.json','.npz')]
            files=['model.pt','scores.npz','trace.json',*feature_files]
            base.write_json(dest/'result.json',dict(task=task,split=split,kind=kind,
                protocol_sha256=protocol_hash,metrics=measured,features=records,
                fit_seconds=seconds,epochs=p['epochs'][task],kernel_expected_max_abs=expected_error,
                neural_finetuning=False,end_to_end_encoder_readout_replay_max_abs=end_to_end_error,
                replay_max_abs=float(max(np.abs(value-pred).max() for value in (replay,reverse,batched))),
                files={n:base.digest(dest/n) for n in files}))
            print('COMPLETE',task,split,kind,json.dumps(measured),flush=True)
            del model,loaded


def compare(frame,task,kind,score,ref_kind,ref_score):
    left=frame[(frame.task==task)&(frame.kind==kind)&(frame.score==score)].set_index('split').sort_index()
    right=frame[(frame.task==task)&(frame.kind==ref_kind)&(frame.score==ref_score)].set_index('split').sort_index()
    if len(left)!=3 or len(right)!=3 or not left.index.equals(right.index):
        raise ValueError('Missing or unmatched split in contrast')
    row=dict(task=task,kind=kind,score=score,reference_kind=ref_kind,reference_score=ref_score)
    for metric in ('auc','ap','macro_drug_ap'):
        delta=left[metric]-right[metric]
        row.update({metric:float(left[metric].mean()),'reference_'+metric:float(right[metric].mean()),
                    'delta_'+metric:float(delta.mean()),'wins_'+metric:int((delta>0).sum()),
                    'min_delta_'+metric:float(delta.min()),
                    'split_delta_'+metric:{str(k):float(v) for k,v in delta.items()}})
    return row


def assess(frame):
    mi=compare(frame,'MiDrug','pretrained','gnn_composed','pretrained','gnn')
    ln=compare(frame,'LncDrug','pretrained','gnn_composed','pretrained','gnn')
    controls={k:compare(frame,'MiDrug','pretrained','gnn_composed',k,'gnn_composed') for k in ('kernel','random')}
    performance=all(mi['delta_'+m]>0 and mi['wins_'+m]>=2 for m in ('auc','ap'))
    lnc=all(ln['delta_'+m]>=-.001 and ln['min_delta_'+m]>=-.001 for m in ('auc','ap'))
    mechanism=all(row['delta_'+m]>0 and row['wins_'+m]>=2 for row in controls.values() for m in ('auc','ap'))
    return dict(stage='Frozen confirmation on three overlapping development resplits',
        mi_performance=performance,lnc_shared_performance=lnc,mechanism_supported=mechanism,
        shared_gate=performance and lnc and mechanism,mi_vs_gnn=mi,lnc_vs_gnn=ln,
        mi_pretrained_vs_controls=controls,
        development_validated_graph_transfer_hypotheses=int(performance and lnc and mechanism),
        independently_validated_frontier_innovations=0,
        original_test_access=False,independent_biological_validation=False,
        novel_architecture_claim=False,neural_finetuning=False,
        limitation='Descriptive development gates; repeated model search and overlapping splits do not provide independent external or biological confirmation. The pretrained architecture is an existing method; readout/kernel variants count as one graph-transfer hypothesis.')


def report(out):
    out=Path(out);p=protocol(out);rows=[];completed=[]
    for task in p['tasks']:
        for split in p['splits']:
            a=load_inputs(out,task,split)
            for kind in p['kinds']:
                dest=out/'runs'/task/str(split)/kind
                r=json.loads((dest/'result.json').read_text());graph.verify_files(dest,r)
                if r['protocol_sha256']!=base.digest(out/'protocol.json'):raise ValueError('Protocol mismatch')
                with np.load(dest/'scores.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(a['pairs_val'],z['pairs'])
                    np.testing.assert_array_equal(a['y_val'],z['y'])
                    np.testing.assert_array_equal(a['gnn'],z['gnn'])
                    np.testing.assert_array_equal(a['tabicl'],z['tabicl'])
                    np.testing.assert_array_equal(z['gnn_composed'],.5*z['gnn']+.5*z['standalone'])
                    np.testing.assert_array_equal(z['tabicl_composed'],.5*logit(z['tabicl'].astype(np.float64))+.5*z['standalone'])
                    for name,old in r['metrics'].items():
                        if not np.isfinite(z[name]).all():raise ValueError('Nonfinite prediction')
                        m=base.metrics(z['y'],z[name],z['pairs'])
                        for key,value in m.items():
                            if abs(value-old[key])>1e-12:raise ValueError('Metric mismatch')
                        rows.append(dict(task=task,split=split,kind=kind,score=name,**m))
                completed.append(r)
    df=pd.DataFrame(rows);df.to_csv(out/'metrics.csv',index=False)
    summary=df.groupby(['task','kind','score'])[['auc','ap','macro_drug_ap']].mean().reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    contrasts=[]
    for task in p['tasks']:
        for kind in p['kinds']:
            for score,ref in [('standalone','gnn'),('gnn_composed','gnn'),('tabicl_composed','tabicl')]:
                contrasts.append(compare(df,task,kind,score,kind,ref))
    base.write_json(out/'contrasts.json',contrasts)
    flat=[{k:v for k,v in r.items() if not k.startswith('split_delta_')} for r in contrasts]
    pd.DataFrame(flat).to_csv(out/'contrasts.csv',index=False)
    assessment=assess(df);base.write_json(out/'ASSESSMENT.json',assessment)
    base.write_json(out/'COMPLETION.json',dict(all_metrics_recomputed=True,all_hashes_verified=True,
        readout_fits=len(completed),context_passes=sum(len(r['features']) for r in completed),
        max_encoder_readout_replay_error=max(r['end_to_end_encoder_readout_replay_max_abs'] for r in completed),
        max_readout_replay_error=max(r['replay_max_abs'] for r in completed),
        files={n:base.digest(out/n) for n in ('metrics.csv','summary.csv','contrasts.json','contrasts.csv','ASSESSMENT.json')}))
    print(summary.to_string(index=False));print(json.dumps(assessment,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','run','report'))
    ap.add_argument('--out-dir',default=str(OUT));ap.add_argument('--task',choices=base.TASKS)
    ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    if a.command=='prepare':prepare(a.out_dir)
    elif a.command=='run':run(a.out_dir,a.task,a.device)
    else:report(a.out_dir)


if __name__=='__main__':main()
