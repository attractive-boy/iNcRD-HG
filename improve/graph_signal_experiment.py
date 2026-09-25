"""Two-axis spectral diffusion and annotation anchoring, with frozen controls."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from improve import non_tabular_experiment as base
from improve import graph_signal_diffusion as methods

OUT=base.ROOT/'results'/'graph_signal_diffusion_20260914'


def prepare(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new output directory')
    base.load_protocol(base.OUT)
    names=('graph_signal_diffusion.py','graph_signal_experiment.py','non_tabular_experiment.py','non_tabular_models.py')
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),tasks=list(base.TASKS),seeds=[666,0,1],
           configs=methods.configurations(),source_sha256={n:base.digest(base.ROOT/n) for n in names},input_sha256={},
           literature='Zhu et al. Graph Signal Diffusion Model for Collaborative Filtering. SIGIR 2024; DOI 10.1145/3626772.3657759; equations and public TensorFlow implementation checked.',
           architecture='Independent PyTorch adaptation: separate RNA-history and drug-history denoisers; fixed arithmetic mean of complete score matrices; 128-mode spectral approximation and optional equal annotation-kernel anchor.',
           primary='Three-step reverse diffusion, arithmetic mean across both orientations. One-step and each orientation are diagnostics at the SAME primary-selected checkpoint.',
           controls='Support-only graph vs annotation anchored graph vs permuted annotation control. Three-step vs one-step at same weights tests reverse refinement; variants are not separate innovative techniques.',
           selection='Historical inner only. One joint checkpoint maximizes GLOBAL (ROC AUC+average_precision)/2. No coefficients or checkpoints selected using future resplits.',
           graph_policy='Only fitting positives construct the matrix and its spectrum. Matrix reconstruction uses all missing entries as unobserved zeros under implicit-feedback convention; these are not true-negative biological labels. Validation and test labels do not construct the graph.',
           side_policy='Same frozen node kernels as the non-tabular screen. The prior is annotation identity plus original attributes, not a measured causal target response.',
           original_test_labels_access=False,outer_query_labels_access=False,
           failure_policy='Retain every control and failure. No success or novelty claim follows merely from successful code execution.')
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    for n in names:shutil.copy2(base.ROOT/n,out/'source_snapshot'/n)
    for task in p['tasks']:
        source=base.OUT/'inputs'/task/'arrays.npz'
        with np.load(source,allow_pickle=False) as z:
            a={n:z[n].copy() for n in ('pairs_train','y_train','pairs_val','y_val','kr','kd')}
        graph=np.zeros((len(a['kr']),len(a['kd'])),dtype=np.float32)
        positives=a['pairs_train'][:,a['y_train']==1];graph[positives[0],positives[1]]=1.
        if graph[a['pairs_val'][0],a['pairs_val'][1]].any():raise ValueError('Validation pair in context')
        a['graph']=graph
        for cfg in p['configs']:
            for direction,x,kernel in [('r',graph,a['kd']),('d',graph.T,a['kr'])]:
                vectors,values=methods.spectrum(x,kernel,cfg['variant'],cfg['rank'])
                a[cfg['id']+'_'+direction+'_vectors']=vectors
                a[cfg['id']+'_'+direction+'_values']=values
        dest=out/'inputs'/task;dest.mkdir(parents=True)
        np.savez_compressed(dest/'arrays.npz',**a)
        for path in (source,dest/'arrays.npz'):p['input_sha256'][str(path)]=base.digest(path)
    base.write_json(out/'protocol.json',p)
    print('PREPARED',out,flush=True)


def protocol(out):
    p=json.loads((Path(out)/'protocol.json').read_text())
    for n,h in p['source_sha256'].items():
        if base.digest(base.ROOT/n)!=h:raise ValueError('Source changed: '+n)
    for n,h in p['input_sha256'].items():
        if base.digest(n)!=h:raise ValueError('Input changed: '+n)
    return p


def make(a,cfg,device):
    models=[]
    for direction,n in [('r',a['graph'].shape[1]),('d',a['graph'].shape[0])]:
        models.append(methods.SignalDiffusion(n,a[cfg['id']+'_'+direction+'_vectors'],
            a[cfg['id']+'_'+direction+'_values'],cfg['width'],cfg['steps'],cfg['alpha']).to(device))
    return torch.nn.ModuleList(models)


def predict(models,x):
    with torch.no_grad():
        r=models[0].predict(x,True)
        d=models[1].predict(x.T,True).T
        one=.5*(models[0].predict(x,False)+models[1].predict(x.T,False).T)
    return {'reverse':(.5*r+.5*d).cpu().numpy(),'one_step':one.cpu().numpy(),
            'row_reverse':r.cpu().numpy(),'col_reverse':d.cpu().numpy()}


def train(a,cfg,seed,device):
    start=time.perf_counter();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    models=make(a,cfg,device)
    x=torch.as_tensor(a['graph'],device=device)
    optimizer=torch.optim.AdamW(models.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    best=-np.inf;state=None;trace=[];best_epoch=None
    q=a['pairs_val']
    for epoch in range(1,cfg['epochs']+1):
        models.train();optimizer.zero_grad(set_to_none=True)
        # Equal orientation weight; average squared error per matrix cell.
        loss=.5*((models[0].training_prediction(x)-x).square().mean()+
                 (models[1].training_prediction(x.T)-x.T).square().mean())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite graph diffusion loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(models.parameters(),5.);optimizer.step()
        if epoch%20==0 or epoch==cfg['epochs']:
            models.eval();scores=predict(models,x)
            m=base.metrics(a['y_val'],scores['reverse'][q[0],q[1]],q)
            trace.append(dict(epoch=epoch,loss=float(loss),**m));criterion=(m['auc']+m['ap'])/2
            if criterion>best:
                best,best_epoch=criterion,epoch
                state={k:v.detach().cpu().clone() for k,v in models.state_dict().items()}
    models.load_state_dict(state);models.eval();scores=predict(models,x)
    result=dict(seed=seed,config=cfg,epoch=best_epoch,seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in models.parameters()),
                metrics={n:base.metrics(a['y_val'],s[q[0],q[1]],q) for n,s in scores.items()})
    return result,scores,state,trace


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
                    if base.digest(dest/n)!=h:raise ValueError('Stored result changed')
                continue
            dest.mkdir(parents=True,exist_ok=True)
            row,scores,state,trace=train(a,cfg,seed,args.device)
            torch.save(state,dest/'model.pt')
            loaded=make(a,cfg,args.device)
            loaded.load_state_dict(torch.load(dest/'model.pt',map_location=args.device,weights_only=True));loaded.eval()
            replay=predict(loaded,torch.as_tensor(a['graph'],device=args.device))
            error=0.
            for n,s in scores.items():
                np.testing.assert_allclose(s,replay[n],atol=2e-6,rtol=2e-6)
                error=max(error,float(np.abs(s-replay[n]).max()))
            q=a['pairs_val'];selected={n:s[q[0],q[1]] for n,s in scores.items()}
            for n,s in scores.items():
                np.testing.assert_array_equal(selected[n],s[q[0,::-1],q[1,::-1]][::-1])
                np.testing.assert_array_equal(selected[n],np.concatenate([s[q[0,ix],q[1,ix]] for ix in np.array_split(np.arange(q.shape[1]),7)]))
            np.savez_compressed(dest/'scores.npz',**selected,y=a['y_val'],pairs=q)
            base.write_json(dest/'trace.json',trace)
            row.update(task=args.task,replay_max_abs=error,
                       files={n:base.digest(dest/n) for n in ('model.pt','scores.npz','trace.json')})
            base.write_json(dest/'result.json',row)
            print('VERIFIED',args.task,cfg['id'],seed,row['epoch'],row['metrics'],row['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','run'))
    ap.add_argument('--out-dir',default=str(OUT));ap.add_argument('--task',choices=base.TASKS)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--config',nargs='+')
    ap.add_argument('--seeds',type=int,nargs='+');args=ap.parse_args()
    if args.command=='prepare':prepare(args.out_dir)
    else:run(args)


if __name__=='__main__':main()
