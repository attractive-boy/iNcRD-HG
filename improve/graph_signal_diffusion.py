"""Spectral association-signal diffusion adapted from the GiffCF equations.

Reference: Zhu et al., SIGIR 2024, DOI 10.1145/3626772.3657759,
arXiv:2311.08744. This is an independent PyTorch implementation with
two-axis predictions and optional exogenous node-kernel anchoring, not an
official reproduction. The sparse benchmark's unobserved entries are
reconstruction zeros, not asserted biological negatives.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def spectrum(a, prior, variant='support', rank=128):
    """Training-positive graph spectrum, with a fixed annotation-kernel control."""
    a=np.asarray(a,dtype=np.float64)
    r=np.maximum(a.sum(1),1.)**-.25
    c=np.maximum(a.sum(0),1.)**-.5
    b=r[:,None]*a*c[None,:]
    support=b.T@b
    top=np.linalg.eigvalsh(support)[-1]
    support/=max(top,1e-12)
    if variant!='support':
        prior=np.asarray(prior,dtype=np.float64).copy()
        if variant=='shuffled':
            ids=np.random.default_rng(20260914).permutation(len(prior))
            prior=prior[ids][:,ids]
        prior/=max(np.linalg.eigvalsh(prior)[-1],1e-12)
        support=.5*support+.5*prior
    values,vectors=np.linalg.eigh((support+support.T)/2)
    order=np.argsort(values)[::-1][:min(rank,len(values))]
    values=np.maximum(values[order],0);vectors=vectors[:,order]
    return vectors.astype(np.float32),values.astype(np.float32)


class CoordinateMixer(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(n,2),nn.SiLU(),nn.Linear(2,1))

    def forward(self,*values):
        return self.layers(torch.stack(values,dim=-1)).squeeze(-1)


class SignalDiffusion(nn.Module):
    def __init__(self,n_items,vectors,values,width=64,steps=3,alpha=1.5):
        super().__init__()
        self.steps=steps;self.alpha=alpha;self.n_items=n_items
        self.item=nn.Parameter(torch.empty(n_items,width))
        nn.init.xavier_uniform_(self.item)
        self.register_buffer('vectors',torch.as_tensor(vectors))
        self.register_buffer('values',torch.as_tensor(values))
        f=10000.**(-torch.arange(0,20,2,dtype=torch.float32)/20.)
        angle=torch.arange(steps+1)[:,None]*f[None,:]
        self.register_buffer('time_basis',torch.cat((angle.sin(),angle.cos()),1))
        self.time_latent=nn.Sequential(nn.Linear(20,20),nn.SiLU(),nn.Linear(20,1))
        self.time_score=nn.Sequential(nn.Linear(20,20),nn.SiLU(),nn.Linear(20,1))
        self.latent_mix=CoordinateMixer(3)
        self.score_mix=CoordinateMixer(4)

    def smooth(self,x):
        return ((x@self.vectors)*self.values[None,:])@self.vectors.T

    def blur(self,x,t):
        return x+(self.alpha*t/self.steps)*(self.smooth(x)-x)

    def reconstruct(self,z,context,context_smooth,t):
        time=self.time_basis[t.reshape(-1).long()]
        history=context/context.abs().sum(1,keepdim=True).clamp_min(1.)
        latent=self.latent_mix(z@self.item/self.n_items,history@self.item,
                              self.time_latent(time).expand(-1,self.item.shape[1]))
        score=latent@self.item.T
        return self.score_mix(score,context,context_smooth,
                              self.time_score(time).expand(-1,self.n_items))

    def training_prediction(self,x):
        t=torch.randint(1,self.steps+1,(len(x),1),device=x.device)
        z=self.blur(x,t.to(x.dtype))
        context=F.dropout(x,p=.5,training=True)
        return self.reconstruct(z,context,self.smooth(context),t)

    def predict(self,x,reverse=True):
        smooth=self.smooth(x)
        z=self.blur(x,float(self.steps))
        if not reverse:
            t=torch.full((len(x),1),self.steps,device=x.device,dtype=torch.long)
            return self.reconstruct(z,x,smooth,t)
        for t in range(self.steps,0,-1):
            times=torch.full((len(x),1),t,device=x.device,dtype=torch.long)
            recovered=self.reconstruct(z,x,smooth,times)
            # Deterministic reverse update preserves the measured forward residual.
            z=self.blur(recovered,float(t-1))+(z-self.blur(recovered,float(t)))
        return z


def configurations():
    return [dict(id=variant,family='signal_diffusion',variant=variant,width=64,steps=3,
                 alpha=1.5,lr=.003,weight_decay=.0001,epochs=1000,rank=128)
            for variant in ('support','annotation','shuffled')]
