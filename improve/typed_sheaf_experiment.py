"""Screen a new typed sheaf architecture with matched transport controls."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import non_tabular_experiment as common
from improve import typed_sheaf_models as models

OUT=common.ROOT/'results'/'typed_sheaf_20260914'


def prepare(out):
    # This preparation command runs only in the existing PyG environment.
    from improve.backbone_replacement import stage_inputs
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output directory')
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    own=('typed_sheaf_models.py','typed_sheaf_experiment.py','non_tabular_experiment.py','non_tabular_models.py')
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),tasks=list(common.TASKS),
        configs=models.configurations(),seeds=[666,0,1],source_sha256={n:common.digest(common.ROOT/n) for n in own},
        input_sha256={},check_interval=10,
        hypothesis='Non-scalar, edge-conditioned matrix transport helps typed association prediction beyond identity/scalar message passing with identical surrounding architecture.',
        literature='Deep Neural Sheaf Diffusion, arXiv:2605.19021 (2026 preprint; workshop author claim, not treated as main-conference acceptance).',
        adaptation='SO(2) edge transports, symmetric degree normalization, same typed node attributes, physical-edge relation counts, attributed link decoder; paper reproduction not claimed.',
        selection='Historical inner only; maximize GLOBAL (AUC+average_precision)/2 at one checkpoint, all controls and configurations reported.',
        graph_policy='Use prior corrected graph. Remove query positives from physical support before counts, restriction maps and all propagation. Undirected physical edges carry forward/transposed transport; no query label is given to the model.',
        controls='Identity and scalar projection use the same inputs, depth, residual gates, normalization, decoder and optimizer. Orthogonal maps must beat both controls to support the mechanism.',
        nominal_parameters_note='Identity has inactive map parameters; scalar starts with nonzero maps to avoid stationary cosine derivative at zero. Counts in all arms mean improvements vs old GNN cannot be attributed wholly to sheaf transport.',
        original_test_access=False,outer_query_labels_access=False,new_external_data=False)
    for n in own:shutil.copy2(common.ROOT/n,out/'source_snapshot'/n)
    for task in p['tasks']:
        a,g,_=stage_inputs(common.GRAPH,task,'inner')
        nts=list(g.node_types);offsets={};count=0
        for nt in nts:offsets[nt]=count;count+=g[nt].num_nodes
        edges=[];rel=[];keys=[]
        target_codes=[]
        rna='miRNA' if task=='MiDrug' else 'lncRNA'
        for key,ix in g.edge_index_dict.items():
            s,name,t=key
            if name.startswith('rev_'):continue
            index=ix.numpy()
            if s==t:index=index[:,index[0]<index[1]]
            if index.shape[1]==0:continue
            code=np.full(index.shape[1],-1,dtype=np.int64)
            if key==(rna,task,'drug'):code=index[0]*g['drug'].num_nodes+index[1]
            shifted=index+np.array([offsets[s],offsets[t]])[:,None]
            edges.append(shifted);rel.append(np.full(len(code),len(keys),dtype=np.int64))
            target_codes.append(code);keys.append(list(key))
        b={n:a[n].copy() for n in ('pairs_train','pairs_val','y_train','y_val')}
        b.update(edges=np.concatenate(edges,1),relation=np.concatenate(rel),target_codes=np.concatenate(target_codes),
                 node_counts=np.array([g[nt].num_nodes for nt in nts],dtype=np.int64),
                 pair_offset=np.array([offsets[rna],offsets['drug']],dtype=np.int64),nd=np.array(g['drug'].num_nodes))
        for i,nt in enumerate(nts):b['x_'+str(i)]=g[nt].x.numpy().copy()
        if set(b['target_codes'][b['target_codes']>=0])!=set(a['pairs_train'][:,a['y_train']==1][0]*int(b['nd'])+a['pairs_train'][:,a['y_train']==1][1]):
            raise ValueError('Target support mismatch')
        if np.isin(a['pairs_val'][0]*int(b['nd'])+a['pairs_val'][1],b['target_codes']).any():
            raise ValueError('Validation pair visible')
        dest=out/'inputs'/task;dest.mkdir(parents=True)
        np.savez_compressed(dest/'arrays.npz',**b)
        common.write_json(dest/'graph.json',dict(node_types=nts,physical_relations=keys,
                                              nodes=count,physical_edges=b['edges'].shape[1],target_overlap=0))
        for path in (dest/'arrays.npz',dest/'graph.json',common.GRAPH/'inputs'/task/'inner.npz',common.GRAPH/'inputs'/task/'train_graph.pth'):
            p['input_sha256'][str(path)]=common.digest(path)
    common.write_json(out/'protocol.json',p)
    print('PREPARED',out,flush=True)


def protocol(out):
    p=json.loads((Path(out)/'protocol.json').read_text())
    for n,h in p['source_sha256'].items():
        if common.digest(common.ROOT/n)!=h:raise ValueError('Source changed: '+n)
    for n,h in p['input_sha256'].items():
        if common.digest(n)!=h:raise ValueError('Input changed: '+n)
    return p


def model_for(a,cfg):
    return models.TypedSheaf([a['x_'+str(i)].shape[1] for i in range(len(a['node_counts']))],
                             list(map(int,a['node_counts'])),int(a['relation'].max()+1),
                             cfg['channels'],cfg['depth'],cfg['mode'])


def train(a,cfg,seed,device):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    model=model_for(a,cfg).to(device)
    # A scalar cos(theta) control has zero gradient at theta=0; avoid that artifact.
    if cfg['mode']=='scalar':
        for layer in model.layers:
            torch.nn.init.normal_(layer.maps[-1].weight,std=.01)
            torch.nn.init.normal_(layer.maps[-1].bias,std=.01)
    xs=[torch.as_tensor(a['x_'+str(i)],device=device) for i in range(len(a['node_counts']))]
    edge=torch.as_tensor(a['edges'],device=device);relation=torch.as_tensor(a['relation'],device=device)
    labels=torch.as_tensor(a['y_train'],device=device)
    train_pairs=torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None],device=device)
    query_pairs=torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    best=-np.inf;state=None;best_epoch=None;trace=[];start=time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        query=np.concatenate([rng.choice(np.flatnonzero(a['y_train']==label),
                             max(1,int((a['y_train']==label).sum()*cfg['query_fraction'])),replace=False)
                              for label in (0,1)])
        held=a['pairs_train'][:,query[a['y_train'][query]==1]]
        code=held[0]*int(a['nd'])+held[1]
        keep=~np.isin(a['target_codes'],code)
        assert not np.isin(a['target_codes'][keep],code).any()
        keep=torch.as_tensor(keep,device=device);ids=torch.as_tensor(query,device=device)
        model.train();opt.zero_grad(set_to_none=True)
        prediction=model(xs,edge[:,keep],relation[keep],train_pairs[:,ids])
        loss=F.binary_cross_entropy_with_logits(prediction,labels[ids])
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite sheaf loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        if epoch%10==0 or epoch==cfg['epochs']:
            model.eval()
            with torch.no_grad():pred=model(xs,edge,relation,query_pairs).cpu().numpy()
            m=common.metrics(a['y_val'],pred,a['pairs_val']);criterion=(m['auc']+m['ap'])/2
            trace.append(dict(epoch=epoch,loss=float(loss),**m))
            if criterion>best:
                best,best_epoch=criterion,epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state);model.eval()
    with torch.no_grad():pred=model(xs,edge,relation,query_pairs).cpu().numpy()
    return dict(seed=seed,config=cfg,epoch=best_epoch,seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in model.parameters()),**common.metrics(a['y_val'],pred,a['pairs_val'])),pred,state,trace


def run(args):
    out=Path(args.out_dir);p=protocol(out);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with np.load(out/'inputs'/args.task/'arrays.npz',allow_pickle=False) as z:a={n:z[n].copy() for n in z.files}
    for cfg in p['configs']:
        if args.config and cfg['id'] not in args.config:continue
        for seed in (args.seeds if args.seeds else p['seeds']):
            dest=out/'screen'/args.task/cfg['id']/str(seed)
            if (dest/'result.json').exists():
                r=json.loads((dest/'result.json').read_text())
                for n,h in r['files'].items():
                    if common.digest(dest/n)!=h:raise ValueError('Result changed')
                continue
            dest.mkdir(parents=True,exist_ok=True)
            row,pred,state,trace=train(a,cfg,seed,args.device)
            torch.save(state,dest/'model.pt')
            model=model_for(a,cfg).to(args.device)
            model.load_state_dict(torch.load(dest/'model.pt',map_location=args.device,weights_only=True));model.eval()
            xs=[torch.as_tensor(a['x_'+str(i)],device=args.device) for i in range(len(a['node_counts']))]
            pairs=torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=args.device)
            with torch.no_grad():
                h,c=model.encode(xs,torch.as_tensor(a['edges'],device=args.device),torch.as_tensor(a['relation'],device=args.device))
                replay=model.decode(h,c,pairs).cpu().numpy()
                reverse=model.decode(h,c,pairs.flip(1)).cpu().numpy()[::-1]
                batched=torch.cat([model.decode(h,c,q) for q in torch.tensor_split(pairs,7,dim=1)]).cpu().numpy()
            np.testing.assert_allclose(pred,replay,atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(pred,reverse,atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(pred,batched,atol=2e-5,rtol=2e-5)
            np.savez_compressed(dest/'scores.npz',pred=pred,y=a['y_val'],pairs=a['pairs_val'])
            common.write_json(dest/'trace.json',trace)
            row.update(task=args.task,replay_max_abs=float(max(np.max(np.abs(v-pred)) for v in (replay,reverse,batched))),
                       files={n:common.digest(dest/n) for n in ('model.pt','scores.npz','trace.json')})
            common.write_json(dest/'result.json',row)
            print('VERIFIED',args.task,cfg['id'],seed,row['epoch'],row['auc'],row['ap'],row['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','run'))
    ap.add_argument('--out-dir',default=str(OUT));ap.add_argument('--task',choices=common.TASKS)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--config',nargs='+')
    ap.add_argument('--seeds',type=int,nargs='+');args=ap.parse_args()
    if args.command=='prepare':prepare(args.out_dir)
    else:run(args)


if __name__=='__main__':main()
