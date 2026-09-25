"""Test fixed vs renewed query masking with matched inference support density.

This is a follow-up after the first association denoiser failed, not an
independent innovation. It tests a concrete proposed cause of that failure.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import non_tabular_experiment as base

OUT=base.ROOT/'results'/'graph_mask_resampling_20260914'


def holdout(labels,seed):
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(labels==c),int((labels==c).sum()*.5),replace=False)
                           for c in (0,1)])


def context(a,held,device):
    keep=np.ones(len(a['y_train']),dtype=bool);keep[held]=False
    shape=(len(a['ur']),len(a['ud']))
    positive=np.zeros(shape,dtype=np.float32);zero=np.zeros(shape,dtype=np.float32)
    for c,m in ((1,positive),(0,zero)):
        q=a['pairs_train'][:,keep&(a['y_train']==c)];m[q[0],q[1]]=1.
    q=a['pairs_train'][:,held]
    if positive[q[0],q[1]].any() or zero[q[0],q[1]].any():raise ValueError('Hidden query outcome entered context')
    return torch.as_tensor(positive,device=device),torch.as_tensor(zero,device=device)


def prepare(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new directory')
    base.load_protocol(base.OUT)
    own=('graph_mask_resampling.py','non_tabular_models.py','non_tabular_experiment.py')
    configs=[dict(id=f'{mode}_g{int(graph)}',mask_mode=mode,family='denoise',rank=32,width=64,
                  graph=graph,zeros=True,lr=.001,weight_decay=.01,epochs=600)
             for mode in ('fixed','resampled') for graph in (False,True)]
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),tasks=list(base.TASKS),seeds=[666,0,1],configs=configs,
           source_sha256={n:base.digest(base.ROOT/n) for n in own},
           input_sha256={str(base.OUT/'inputs'/t/'arrays.npz'):base.digest(base.OUT/'inputs'/t/'arrays.npz') for t in base.TASKS},
           hypothesis='Fixed query partitions may permit context memorization; resample queries every step, while controlling masked fraction and inference density.',
           primary='Mean logits from five predetermined 50%-support graphs; no new validation labels used in any support. Full support is a diagnostic at the same checkpoint.',
           budget='Four combinations of fixed/resampled masking and graph/no-graph; same model shapes, optimizer, 600 updates, query fraction 0.5, and five inference supports.',
           novelty='Diagnostic follow-up, never counted as a separate frontier technology.',
           selection='Historical inner mean of global AUC and AP at one checkpoint. All variants and seeds reported.',
           test_labels_access=False)
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    for n in own:shutil.copy2(base.ROOT/n,out/'source_snapshot'/n)
    base.write_json(out/'protocol.json',p);print('PREPARED',out,flush=True)


def protocol(out):
    p=json.loads((Path(out)/'protocol.json').read_text())
    for n,h in p['source_sha256'].items():
        if base.digest(base.ROOT/n)!=h:raise ValueError('Source changed')
    for n,h in p['input_sha256'].items():
        if base.digest(n)!=h:raise ValueError('Input changed')
    return p


def train(a,cfg,seed,device):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=base.model_for(a,cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    train_pairs=torch.as_tensor(a['pairs_train'],device=device);labels=torch.as_tensor(a['y_train'],device=device)
    q=torch.as_tensor(a['pairs_val'],device=device)
    fixed=holdout(a['y_train'],seed+557)
    complement=np.setdiff1d(np.arange(len(labels)),fixed)
    eval_context=[context(a,holdout(a['y_train'],202609140+i),device) for i in range(5)]
    full=context(a,np.array([],dtype=np.int64),device)
    best=-np.inf;state=None;trace=[];best_epoch=None;start=time.perf_counter()

    def score():
        with torch.no_grad():
            primary=torch.stack([model.full(*c,time=.5)[q[0],q[1]] for c in eval_context]).mean(0).cpu().numpy()
            full_score=model.full(*full,time=.5)[q[0],q[1]].cpu().numpy()
        return primary,full_score

    for epoch in range(1,cfg['epochs']+1):
        held=(holdout(a['y_train'],seed*1000003+epoch) if cfg['mask_mode']=='resampled'
              else fixed if epoch%2 else complement)
        pos,zero=context(a,held,device);ix=torch.as_tensor(held,device=device)
        model.train();opt.zero_grad(set_to_none=True)
        heat=float(np.random.default_rng(seed+epoch).uniform(0,1))
        logits=model.full(pos,zero,heat,True)[train_pairs[0,ix],train_pairs[1,ix]]
        loss=F.binary_cross_entropy_with_logits(logits,labels[ix])
        if not torch.isfinite(loss):raise FloatingPointError('Invalid denoising loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if epoch%10==0:
            model.eval();pred,_=score();m=base.metrics(a['y_val'],pred,a['pairs_val'])
            trace.append(dict(epoch=epoch,loss=float(loss),**m));criterion=(m['auc']+m['ap'])/2
            if criterion>best:
                best,best_epoch=criterion,epoch
                state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    model.load_state_dict(state);model.eval();pred,full_score=score()
    row=dict(config=cfg,seed=seed,epoch=best_epoch,seconds=time.perf_counter()-start,
             primary=base.metrics(a['y_val'],pred,a['pairs_val']),
             full_support=base.metrics(a['y_val'],full_score,a['pairs_val']))
    return row,dict(pred=pred,full_support=full_score,y=a['y_val'],pairs=a['pairs_val']),state,trace,eval_context,full


def run(args):
    out=Path(args.out_dir);p=protocol(out);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with np.load(base.OUT/'inputs'/args.task/'arrays.npz',allow_pickle=False) as z:a={n:z[n].copy() for n in z.files}
    for cfg in p['configs']:
        for seed in p['seeds']:
            dest=out/'screen'/args.task/cfg['id']/str(seed)
            if (dest/'result.json').exists():
                r=json.loads((dest/'result.json').read_text())
                for n,h in r['files'].items():
                    if base.digest(dest/n)!=h:raise ValueError('Stored result changed')
                continue
            dest.mkdir(parents=True,exist_ok=True)
            row,scores,state,trace,contexts,full=train(a,cfg,seed,args.device)
            torch.save(state,dest/'model.pt')
            model=base.model_for(a,cfg).to(args.device)
            model.load_state_dict(torch.load(dest/'model.pt',map_location=args.device,weights_only=True));model.eval()
            q=torch.as_tensor(a['pairs_val'],device=args.device)
            with torch.no_grad():
                replay=torch.stack([model.full(*c,time=.5)[q[0],q[1]] for c in contexts]).mean(0).cpu().numpy()
            np.testing.assert_array_equal(scores['pred'],replay)
            np.savez_compressed(dest/'scores.npz',**scores);base.write_json(dest/'trace.json',trace)
            row.update(task=args.task,replay_max_abs=0.,files={n:base.digest(dest/n) for n in ('model.pt','scores.npz','trace.json')})
            base.write_json(dest/'result.json',row)
            print('VERIFIED',args.task,cfg['id'],seed,row['epoch'],row['primary'],row['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','run'))
    ap.add_argument('--out-dir',default=str(OUT));ap.add_argument('--task',choices=base.TASKS)
    ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    if args.command=='prepare':prepare(args.out_dir)
    else:run(args)


if __name__=='__main__':main()
