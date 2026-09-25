"""Graph state-memory experiment with separately selected matched controls."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve import overlap_block_models as graph_lib
from improve import graph_prediction_precision as precision
from improve import state_space_graph as method

OUT=common.base.ROOT/'results/state_space_graph_20260915'
KINDS=('full','static','no_state','random_order')


def prepare(out,smoke=False):
    out=Path(out).resolve()
    if out.exists():raise ValueError('Use a new output directory')
    parent=common.protocol(common.OUT)
    own=('state_space_graph.py','state_space_experiment.py','test_state_space_graph.py',
         'graph_prediction_precision.py','overlap_block_experiment.py',
         'overlap_block_models.py','non_tabular_experiment.py')
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),method='state_space',smoke=smoke,
        tasks=parent['tasks'],kinds=list(KINDS),screen_seed=666,confirm_seeds=[0,1],
        width=32 if smoke else 64,depth=1 if smoke else 3,heads=4,state=8,chunk=64,
        dropout=.1,folds=5,order_seed=20260915,epochs=2 if smoke else 400,
        check_interval=1 if smoke else 20,lrs=[.001] if smoke else [.001,.003],
        weight_decay=.001,strong_weight=.1,gnn_weight=.5,lnc_absolute_tolerance=.001,
        selection=parent['selection'],gate=parent['gate'],success_scope=parent['success_scope'],
        boundary=parent['boundary'].split('Label-blind')[0]+' No expander edges. Node degree ordering is recomputed after query-fact deletion.',
        literature=[dict(title='Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces',arxiv='2402.00789v1'),
                    dict(title='Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality',arxiv='2405.21060v1')],
        adaptation='Typed graph-local attention plus bidirectional scalar-per-head SSD memory; node-dependent Delta/B/C, learned negative A, 8 state coordinates, sorted support degrees with fixed seeded tie keys per support fold. No sequence convolution or random reordering augmentation. Dense chunk-boundary contractions are used instead of the official fused kernel; no asymptotic runtime claim.',
        controls='Static input-independent Delta/B/C; no cross-node recurrence with otherwise identical input-dependent projections and pointwise update; selective recurrence with degree-independent random order. All arms have the same nominal modules; static BC and selective BC projections are respectively inactive where unused.',
        numerics='Float32 training. Deterministic graph reductions and triangular contractions replace nondeterministic CUDA cumsum in evaluation; float64 sigmoid/probability averaging. Same deterministic inference for checkpoint selection and final replay.',
        input_sha256_note='All target and reference inputs are the same historical development arrays as the block experiment.',
        source_sha256={n:common.base.digest(common.base.ROOT/n) for n in own},inputs={},
        no_original_test_access=True,no_original_outer_query_label_access=True,
        existing_default_replaced=False,confirmed_innovations=0)
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    for n in own:shutil.copy2(common.base.ROOT/n,out/'source_snapshot'/n)
    for task in p['tasks']:
        source=common.OUT/'inputs'/task/'arrays.npz'
        dest=out/'inputs'/task/'arrays.npz';dest.parent.mkdir(parents=True)
        shutil.copy2(source,dest)
        for path in (source,dest):p['inputs'][str(path)]=common.base.digest(path)
    common.base.write_json(out/'protocol.json',p)
    print('FROZEN',out,flush=True)


def setup(a,p,kind,device):
    n=int(a['node_counts'].sum());nr=int(a['relation'].max()+1)
    graphs,queries=[],[]
    for fold in range(p['folds']):
        ix=np.flatnonzero(a['folds']==fold)
        edge,relation=graph_lib.masked_physical(a,ix)
        graph=graph_lib.graph_inputs(edge,relation,np.empty((2,0),np.int64),n,nr,device)
        order=method.scan_order(graph[2].cpu().numpy(),p['order_seed']+fold,kind=='random_order')
        graphs.append((*graph,torch.as_tensor(order,device=device)))
        queries.append(torch.as_tensor(ix,device=device))
    xs=[torch.as_tensor(a['x_'+str(i)],device=device) for i in range(len(a['node_counts']))]
    model=method.StateSpaceGraph([x.shape[1] for x in xs],list(map(int,a['node_counts'])),nr,
        p['width'],p['depth'],p['heads'],p['state'],p['dropout'],kind,p['chunk']).to(device)
    return model,xs,graphs,queries


def train(a,p,kind,lr,seed,device):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model,xs,graphs,indices=setup(a,p,kind,device)
    pt=torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None],device=device)
    pv=torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=device)
    y=torch.as_tensor(a['y_train'],device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=p['weight_decay'])
    best=-np.inf;best_epoch=None;state=None;trace=[];start=time.monotonic()
    for epoch in range(1,p['epochs']+1):
        fold=(epoch-1)%len(graphs)
        model.train();optimizer.zero_grad(set_to_none=True)
        pred=model(xs,graphs[fold],pt[:,indices[fold]])
        loss=F.binary_cross_entropy_with_logits(pred,y[indices[fold]])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite state-memory objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if epoch%p['check_interval']==0 or epoch==p['epochs']:
            prediction=precision.geometry_prediction(model,xs,graphs,pv).cpu().numpy()
            mixed=(1-p['strong_weight'])*a['anchor_'+str(seed)]+p['strong_weight']*prediction
            mm=common.base.metrics(a['y_val'],mixed,a['pairs_val'])
            raw=common.base.metrics(a['y_val'],prediction,a['pairs_val'])
            criterion=(mm['auc']+mm['ap'])/2
            trace.append(dict(epoch=epoch,loss=float(loss.detach()),anchor_composed=mm,standalone=raw))
            if criterion>best:
                best,best_epoch=criterion,epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch%max(100,p['check_interval'])==0:
                print('TRAIN',kind,lr,seed,epoch,round(raw['auc'],6),round(raw['ap'],6),
                      round(time.monotonic()-start,1),flush=True)
    model.load_state_dict(state)
    prediction=precision.geometry_prediction(model,xs,graphs,pv).cpu().numpy()
    return model,prediction,trace,dict(epoch=best_epoch,seconds=time.monotonic()-start,
        parameters=sum(v.numel() for v in model.parameters()))


def run(out,task,device,stage):
    out=Path(out);p=common.protocol(out);a=common.load(out/'inputs'/task/'arrays.npz')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ph=common.base.digest(out/'protocol.json')
    selected=None if stage=='screen' else json.loads((out/'SELECTION.json').read_text())
    if selected is not None and selected['protocol_sha256']!=ph:raise ValueError('Selection protocol changed')
    seeds=[p['screen_seed']] if stage=='screen' else p['confirm_seeds']
    for kind in p['kinds']:
        lrs=p['lrs'] if stage=='screen' else [selected['chosen'][kind]['lr']]
        for lr in lrs:
            for seed in seeds:
                dest=out/'runs'/task/kind/('lr'+str(lr))/('s'+str(seed))
                if (dest/'result.json').exists():common.verify(dest,ph);continue
                dest.mkdir(parents=True,exist_ok=True)
                model,pred,trace,detail=train(a,p,kind,lr,seed,device)
                torch.save(model.state_dict(),dest/'model.pt')
                common.base.write_json(dest/'trace.json',trace)
                common.base.write_json(dest/'training_detail.json',detail)
                reloaded,xs,graphs,_=setup(a,p,kind,device)
                reloaded.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True),strict=True)
                pv=torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=device)
                replay=precision.geometry_prediction(reloaded,xs,graphs,pv).cpu().numpy()
                reverse=precision.geometry_prediction(reloaded,xs,graphs,pv.flip(1)).cpu().numpy()[::-1]
                chunk=np.concatenate([precision.geometry_prediction(reloaded,xs,graphs,pv[:,i:i+800]).cpu().numpy()
                                      for i in range(0,pv.shape[1],800)])
                for value in (replay,reverse,chunk):np.testing.assert_allclose(pred,value,atol=3e-5,rtol=3e-5)
                scores=dict(y=a['y_val'],pairs=a['pairs_val'],standalone=pred)
                for name in ('gnn','tabicl','anchor'):
                    scores[name]=a[name+'_'+str(seed)]
                    weight=p['gnn_weight'] if name=='gnn' else p['strong_weight']
                    scores[name+'_composed']=(1-weight)*scores[name]+weight*pred
                np.savez_compressed(dest/'scores.npz',**scores)
                result=dict(task=task,kind=kind,lr=lr,seed=seed,**detail,protocol_sha256=ph,
                    replay_max_abs=float(max(np.max(abs(pred-v)) for v in (replay,reverse,chunk))),
                    metrics={n:common.base.metrics(scores['y'],v,scores['pairs']) for n,v in scores.items() if n not in ('y','pairs')},
                    files={n:common.base.digest(dest/n) for n in ('model.pt','trace.json','training_detail.json','scores.npz')})
                common.base.write_json(dest/'result.json',result)
                print('VERIFIED',task,kind,lr,seed,result['epoch'],result['metrics']['anchor_composed'],flush=True)
                del model,reloaded,xs,graphs;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','screen','select','confirm','report'))
    parser.add_argument('--out',default=str(OUT));parser.add_argument('--task',choices=common.base.TASKS)
    parser.add_argument('--device',default='cuda:1');parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if a.command=='prepare':prepare(a.out,a.smoke)
    elif a.command in ('screen','confirm'):run(a.out,a.task,a.device,a.command)
    elif a.command=='select':common.select(a.out)
    else:common.report(a.out)
