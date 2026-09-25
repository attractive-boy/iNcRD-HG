"""Frozen graph-foundation features as an additional kernel with attributes.

The earlier zero-shot scalar score failed. This follow-up keeps the pretrained
GAT frozen and learns a task readout from OOF representations. An original
attribute/gene kernel remains available for structurally indistinguishable RNA
nodes. This is one graph-transfer family, not a second claimed innovation.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from improve import non_tabular_experiment as base
from improve.non_tabular_models import KernelLogistic
from improve import reification_transfer as prior

OUT=base.ROOT/'results'/'reification_kernel_20260914'


def query_features(states,known,relation_nodes,rna_start,rna_count):
    """The three query-conditioned channels used in the paper's QC readout."""
    b=torch.arange(len(states),device=states.device)
    x=states[:,rna_start:rna_start+rna_count]
    h=states[b,known][:,None,:]
    r=states[b,relation_nodes][:,None,:]
    return torch.cat((x,x*r,x*h),dim=-1)


@torch.no_grad()
def extract(model,graph,a,pairs,batch_size=8):
    device=next(model.parameters()).device
    pairs=np.asarray(pairs,dtype=np.int64)
    drug_ids=np.unique(pairs[1])
    features=np.empty((pairs.shape[1],model.dim*3),dtype=np.float32)
    logits=np.empty(pairs.shape[1],dtype=np.float32)
    captured={}
    def hook(module,args,result):
        captured['states']=result.reshape(captured['batch'],graph.num_nodes,model.dim)
    handle=model.norms[-1].register_forward_hook(hook)
    try:
        for start in range(0,len(drug_ids),batch_size):
            ds=drug_ids[start:start+batch_size]
            seed=torch.as_tensor(ds+int(a['pair_offset'][1]),device=device)
            rel=torch.full_like(seed,int(a['target_relation']))
            role=torch.full_like(seed,prior.SEED_OBJECT)
            captured['batch']=len(ds)
            score=model(graph,seed,rel,role)
            z=query_features(captured.pop('states'),seed,rel+graph.rel_offset,
                             int(a['pair_offset'][0]),int(a['nr']))
            for j,d in enumerate(ds):
                ix=np.flatnonzero(pairs[1]==d)
                rr=torch.as_tensor(pairs[0,ix],device=device)
                features[ix]=z[j,rr].cpu().numpy()
                logits[ix]=score[j,rr+int(a['pair_offset'][0])].cpu().numpy()
    finally:
        handle.remove()
    return features,logits


class TransferKernel(nn.Module):
    def __init__(self,a,x):
        super().__init__()
        self.kernel=KernelLogistic(a['ur'],a['ud'],float(a['y_train'].mean()))
        mean=x.mean(0)
        scale=np.maximum(x.std(0),1e-3)
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.w=nn.Parameter(torch.zeros(x.shape[1]))

    def forward(self,pairs,features):
        z=(features-self.mean)/self.scale/(max(len(self.w),1)**.5)
        return self.kernel(pairs)+z@self.w

    def penalty(self):
        return self.kernel.penalty(1.,1.)+.5*self.w.square().sum()


def prepare(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new experiment directory')
    parent=prior.protocol(prior.OUT)
    names=('reification_kernel.py','reification_transfer.py','non_tabular_models.py','non_tabular_experiment.py')
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),tasks=list(base.TASKS),
           kinds=['pretrained','random'],seed=parent['seed'],folds=parent['folds'],batch_size=8,
           epochs={'MiDrug':480,'LncDrug':210},lr=.03,
           primary='Fixed half-logit GNN + attribute/product kernel augmented with frozen pretrained REIFM query-conditioned features.',
           training='No neural fine-tuning. Only convex logistic kernel/readout weights are fitted on OOF training representations; one shared regularization value 1 for all components and both tasks.',
           features='[candidate_state, candidate_state*relation_state, candidate_state*known_drug_state]; 192 coordinates from the final GAT normalization. Same features from a random-weight control.',
           normalization='Per-coordinate mean/std fitted on OOF training features only, std floor 0.001 to avoid amplifying floating-point noise; divide by sqrt(192). No validation fitting.',
           context='Reuse exactly the five query-masked contexts of the earlier screen; all six connections of excluded facts absent. Mean logits over the five matching contexts at evaluation.',
           controls='Original kernel without graph features, random-weight graph features with identical readout and budgets. Pretrained transfer must beat both in MiDrug AUC and AP.',
           limitation='Attribute branch addresses representation availability for isolated nodes, but is not an outcome guarantee. This follow-up belongs to the same graph-transfer hypothesis.',
           original_test_access=False,outer_query_labels_access=False,
           source_sha256={n:base.digest(base.ROOT/n) for n in names},
           input_sha256={str(prior.OUT/'protocol.json'):base.digest(prior.OUT/'protocol.json')})
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    for n in names:shutil.copy2(base.ROOT/n,out/'source_snapshot'/n)
    for task in p['tasks']:
        graph_input=prior.OUT/'inputs'/task/'arrays.npz'
        node_input=base.OUT/'inputs'/task/'arrays.npz'
        with np.load(graph_input,allow_pickle=False) as z:a={n:z[n].copy() for n in z.files}
        with np.load(node_input,allow_pickle=False) as z:a.update({n:z[n].copy() for n in ('ur','ud')})
        dest=out/'inputs'/task/'arrays.npz';dest.parent.mkdir(parents=True)
        np.savez_compressed(dest,**a)
        for path in (graph_input,node_input,dest):p['input_sha256'][str(path)]=base.digest(path)
        for kind in p['kinds']:
            for fold in range(p['folds']):
                path=prior.OUT/'runs'/task/kind/('fold_'+str(fold)+'.npz')
                p['input_sha256'][str(path)]=base.digest(path)
    base.write_json(out/'protocol.json',p);print('FROZEN',out,flush=True)


def protocol(out):
    prior.protocol(prior.OUT)
    p=json.loads((Path(out)/'protocol.json').read_text())
    for n,h in p['source_sha256'].items():
        if base.digest(base.ROOT/n)!=h:raise ValueError('Source changed: '+n)
    for n,h in p['input_sha256'].items():
        if base.digest(n)!=h:raise ValueError('Input changed: '+n)
    return p


def fit(a,x_train,x_val,epochs,lr,device):
    start=time.perf_counter()
    model=TransferKernel(a,x_train).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    q=torch.as_tensor(a['pairs_train'],device=device)
    y=torch.as_tensor(a['y_train'],device=device)
    z=torch.as_tensor(x_train,device=device)
    trace=[]
    for epoch in range(1,epochs+1):
        opt.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(model(q,z),y)+model.penalty()/len(y)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite graph-kernel loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if epoch%20==0 or epoch==epochs:trace.append(dict(epoch=epoch,regularized_training_loss=float(loss)))
    model.eval()
    with torch.no_grad():
        pred=model(torch.as_tensor(a['pairs_val'],device=device),torch.as_tensor(x_val,device=device)).cpu().numpy()
    return model,pred,trace,time.perf_counter()-start


def run(out,task,device):
    out=Path(out);p=protocol(out);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with np.load(out/'inputs'/task/'arrays.npz',allow_pickle=False) as z:a={n:z[n].copy() for n in z.files}
    for kind in ['kernel',*p['kinds']]:
        dest=out/'runs'/task/kind
        if (dest/'result.json').exists():
            prior.verify_files(dest,json.loads((dest/'result.json').read_text()));continue
        dest.mkdir(parents=True,exist_ok=True)
        feature_records=[]
        if kind=='kernel':
            x_train=np.empty((len(a['y_train']),0),dtype=np.float32)
            x_val=np.empty((len(a['y_val']),0),dtype=np.float32)
        else:
            encoder=prior.model_for(kind,p['seed'],device)
            x_train=np.full((len(a['y_train']),192),np.nan,dtype=np.float32)
            v=[]
            for fold in range(p['folds']):
                path=dest/('features_'+str(fold)+'.npz')
                record_path=dest/('features_'+str(fold)+'.json')
                ix=np.flatnonzero(a['folds']==fold)
                oq=a['pairs_train'][:,ix]
                q=np.concatenate([oq,a['pairs_val']],axis=1)
                if record_path.exists():
                    row=json.loads(record_path.read_text())
                    if base.digest(path)!=row['sha256']:raise ValueError('Features changed')
                else:
                    start=time.perf_counter()
                    graph=prior.context_graph(a,oq[0]*int(a['nd'])+oq[1],device)
                    features,raw=extract(encoder,graph,a,q,p['batch_size'])
                    source=prior.OUT/'runs'/task/kind/('fold_'+str(fold)+'.npz')
                    with np.load(source,allow_pickle=False) as z:expected=z['full'][q[0],q[1]]
                    np.testing.assert_allclose(raw,expected,atol=1e-4,rtol=1e-4)
                    sample=np.arange(min(17,len(q.T)))[::-1]
                    check,_=extract(encoder,graph,a,q[:,sample],3)
                    np.testing.assert_allclose(check,features[sample],atol=2e-4,rtol=2e-4)
                    np.savez_compressed(path,oof=features[:len(ix)],val=features[len(ix):],oof_indices=ix)
                    row=dict(fold=fold,seconds=time.perf_counter()-start,sha256=base.digest(path),
                             original_logit_max_abs=float(np.max(np.abs(raw-expected))),
                             feature_replay_max_abs=float(np.max(np.abs(check-features[sample]))),
                             excludes_query_facts=True,query_permutation_and_batches_checked=True)
                    base.write_json(record_path,row)
                    print('FEATURES',task,kind,fold,round(row['seconds'],1),flush=True)
                    del graph
                with np.load(path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(ix,z['oof_indices'])
                    x_train[ix]=z['oof'];v.append(z['val'].copy())
                feature_records.append(row)
            x_val=np.stack(v).mean(0)
            if not np.isfinite(x_train).all():raise ValueError('OOF coverage incomplete')
            del encoder
        model,pred,trace,seconds=fit(a,x_train,x_val,p['epochs'][task],p['lr'],device)
        torch.save(model.state_dict(),dest/'model.pt')
        loaded=TransferKernel(a,x_train).to(device)
        loaded.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True));loaded.eval()
        with torch.no_grad():
            q=torch.as_tensor(a['pairs_val'],device=device);v=torch.as_tensor(x_val,device=device)
            replay=loaded(q,v).cpu().numpy()
            reversed_score=loaded(q.flip(1),v.flip(0)).cpu().numpy()[::-1]
            batched=torch.cat([loaded(q[:,ix],v[ix]) for ix in torch.tensor_split(torch.arange(len(v),device=device),7)]).cpu().numpy()
        for value in (replay,reversed_score,batched):np.testing.assert_allclose(value,pred,atol=5e-5,rtol=5e-5)
        end_to_end_error=0.
        if kind!='kernel':
            # Recompute the graph encoder after fitting the readout: small
            # raw feature differences must not be amplified into unstable scores.
            encoder=prior.model_for(kind,p['seed'],device)
            sample=np.arange(min(17,len(a['y_val'])))
            probes=a['pairs_val'][:,sample]
            new_features=[]
            for fold in range(p['folds']):
                omitted=a['pairs_train'][:,a['folds']==fold]
                graph=prior.context_graph(a,omitted[0]*int(a['nd'])+omitted[1],device)
                z,_=extract(encoder,graph,a,probes[:,::-1],3)
                new_features.append(z[::-1].copy());del graph
            with torch.no_grad():
                fresh=loaded(torch.as_tensor(probes,device=device),
                    torch.as_tensor(np.stack(new_features).mean(0),device=device)).cpu().numpy()
            end_to_end_error=float(np.max(np.abs(fresh-pred[sample])))
            np.testing.assert_allclose(fresh,pred[sample],atol=5e-4,rtol=5e-4)
            del encoder
        scores=dict(pred=pred,y=a['y_val'],pairs=a['pairs_val'])
        for n,g in a.items():
            if n.startswith('gnn_'):
                scores[n]=g;scores['composed_'+n]=.5*g+.5*pred
        measured={n:base.metrics(scores['y'],value,scores['pairs']) for n,value in scores.items() if n not in ('y','pairs')}
        np.savez_compressed(dest/'scores.npz',**scores);base.write_json(dest/'trace.json',trace)
        feature_files=['features_'+str(f)+ext for f in range(len(feature_records)) for ext in ('.json','.npz')]
        files=['model.pt','scores.npz','trace.json',*feature_files]
        base.write_json(dest/'result.json',dict(task=task,kind=kind,metrics=measured,
            fit_seconds=seconds,features=feature_records,epochs=p['epochs'][task],
            end_to_end_encoder_readout_replay_max_abs=end_to_end_error,
            replay_max_abs=float(max(np.abs(v-pred).max() for v in (replay,reversed_score,batched))),
            files={n:base.digest(dest/n) for n in files}))
        print('COMPLETE',task,kind,measured,flush=True)


def report(out):
    out=Path(out);p=protocol(out);rows=[]
    for task in p['tasks']:
        for kind in ['kernel',*p['kinds']]:
            dest=out/'runs'/task/kind
            r=json.loads((dest/'result.json').read_text());prior.verify_files(dest,r)
            with np.load(dest/'scores.npz',allow_pickle=False) as z:
                for name,old in r['metrics'].items():
                    m=base.metrics(z['y'],z[name],z['pairs'])
                    for key,value in m.items():
                        if abs(value-old[key])>1e-12:raise ValueError('Metric mismatch')
                    rows.append(dict(task=task,kind=kind,score=name,**m))
    df=pd.DataFrame(rows);df.to_csv(out/'metrics.csv',index=False)
    summaries=[]
    for task in p['tasks']:
        for kind in ['kernel',*p['kinds']]:
            part=df[(df.task==task)&(df.kind==kind)]
            new=part[part.score.str.startswith('composed_')]
            ref=part[part.score.str.startswith('gnn_')]
            row=dict(task=task,kind=kind)
            for metric in ('auc','ap','macro_drug_ap'):
                row[metric]=float(new[metric].mean())
                row['delta_'+metric]=float(new[metric].mean()-ref[metric].mean())
                row['standalone_'+metric]=float(part[part.score=='pred'][metric].iloc[0])
            summaries.append(row)
    result=pd.DataFrame(summaries);result.to_csv(out/'summary.csv',index=False)
    def row(task,kind):return next(x for x in summaries if x['task']==task and x['kind']==kind)
    mi,ln=row('MiDrug','pretrained'),row('LncDrug','pretrained')
    controls={name:{m:mi[m]-row('MiDrug',name)[m] for m in ('auc','ap')} for name in ('kernel','random')}
    performance=all(mi['delta_'+m]>0 and ln['delta_'+m]>=-.001 for m in ('auc','ap'))
    mechanism=all(v>0 for r in controls.values() for v in r.values())
    assessment=dict(stage='Historical inner follow-up',two_task_performance=performance,
                    mechanism_controls=controls,mechanism_supported=mechanism,
                    shared_gate=performance and mechanism,frontier_innovations_confirmed=0,
                    original_test_access=False,independent_biological_validation=False,
                    neural_finetuning=False,readout_fits=len(summaries))
    base.write_json(out/'ASSESSMENT.json',assessment)
    base.write_json(out/'COMPLETION.json',dict(all_metrics_recomputed=True,all_hashes_verified=True,
        files={n:base.digest(out/n) for n in ('metrics.csv','summary.csv','ASSESSMENT.json')}))
    print(result.to_string(index=False));print(json.dumps(assessment,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run','report'))
    ap.add_argument('--out-dir',default=str(OUT));ap.add_argument('--task',choices=base.TASKS)
    ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    if a.command=='prepare':prepare(a.out_dir)
    elif a.command=='run':run(a.out_dir,a.task,a.device)
    else:report(a.out_dir)


if __name__=='__main__':main()
