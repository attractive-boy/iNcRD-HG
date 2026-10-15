"""Exact all-pair/cross-regime/within-regime training losses; original inference."""
import numpy as np
import torch
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction

CONFIGS={'all':{},'cross':{},'within':{}}


def support_groups(data):
    pairs=data['train_pairs'];y=data['train_y'];positive=pairs[y==1]
    dr=np.bincount(positive[:,0],minlength=605);dd=np.bincount(positive[:,1],minlength=216)
    # Training records are unique, so y exactly equals the queried adjacency cell.
    degree=np.stack((dr[pairs[:,0]]-y,dd[pairs[:,1]]-y),axis=-1)
    cuts=np.median(degree,axis=0)
    groups=(degree[:,0]>cuts[0]).astype(np.int64)*2+(degree[:,1]>cuts[1]).astype(np.int64)
    return groups,cuts


def pair_counts(labels,groups):
    p=torch.bincount(groups[labels==1],minlength=4);n=torch.bincount(groups[labels==0],minlength=4)
    within=int((p*n).sum());total=int(p.sum()*n.sum())
    return dict(all=total,cross=total-within,within=within)


def ranking(logits,labels,groups,kind,block=256):
    if kind not in CONFIGS:raise ValueError(kind)
    positive=logits[labels==1];negative=logits[labels==0]
    pg=groups[labels==1];ng=groups[labels==0];counts=pair_counts(labels,groups);denominator=counts[kind]
    if denominator==0:return logits.sum()*0
    result=logits.new_zeros(())
    for begin in range(0,len(positive),block):
        p=positive[begin:begin+block]
        loss=F.softplus(negative[None,:]-p[:,None])
        if kind!='all':
            mask=pg[begin:begin+block,None]==ng[None,:]
            if kind=='cross':mask=~mask
            loss=loss*mask
        result=result+loss.sum()/denominator
    return result


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    groups,cuts=support_groups(data);a.rank_groups=torch.as_tensor(groups,device=device,dtype=torch.long)
    a.rank_cuts=cuts;a.rank_counts=pair_counts(a.y,a.rank_groups)
    if name in CONFIGS:a.rank_kind=name
    return a


def mask_generator(device,seed):return torch.Generator(device=device).manual_seed(seed)


def make_optimizer(m):
    return torch.optim.RMSprop([
        dict(params=[p for n,p in m.named_parameters() if not n.startswith('reconstruction.')],lr=1e-4),
        dict(params=m.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4)],alpha=.99)


def step(a,opt,epoch,generator):
    m=a.model;m.train();opt.zero_grad(set_to_none=True)
    logits=a.logits('train');bce=F.binary_cross_entropy_with_logits(logits,a.y)
    rank=ranking(logits,a.y,a.rank_groups,a.rank_kind)
    indices=m.reconstruction.masks(a.graph,.5,generator);auxiliary,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
    rw=.1*min(1.,max(0.,(epoch-30)/30.));aw=10*min(1.,epoch/50.)
    loss=bce+rw*rank+aw*auxiliary
    assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
    return dict(bce=float(bce.detach()),ranking=float(rank.detach()),rank_weight=rw,
                reconstruction=float(auxiliary.detach()),reconstruction_weight=aw,loss=float(loss.detach()))
