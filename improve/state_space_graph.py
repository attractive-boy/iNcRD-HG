"""Typed graph-local attention plus bidirectional selective state memory.

Inspired by Graph-Mamba (arXiv:2402.00789) and the Mamba-2 SSD recurrence
(arXiv:2405.21060). The small-graph PyTorch implementation below uses dense
chunk-boundary contractions, not the official fused linear-time CUDA kernel.
It omits sequence convolution and uses graph-local attention instead.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve.overlap_block_models import SparseTypedAttention


def causal_decay(log_a):
    """Entry (i,j) is product a[j+1:i+1], with a zero upper triangle.

    Direct segment summation avoids subtracting long prefix sums. The SSD
    factorization follows Gu and Dao's published Listing 1; this routine is
    expressed for our head/chunk layout and tested against scalar recurrence.
    """
    length=log_a.shape[-1]
    row=torch.arange(length,device=log_a.device)[:,None]
    column=torch.arange(length,device=log_a.device)[None,:]
    terms=torch.where(row>column,log_a.unsqueeze(-1),0.)
    # PyTorch 2.5 has no deterministic float CUDA cumsum. A triangular
    # contraction computes the same finite segment sums with deterministic
    # cuBLAS and avoids weakening the inference reproducibility requirement.
    prefix=log_a.new_ones((length,length)).tril()
    sums=prefix@terms
    return torch.where(row>=column,sums,torch.full_like(sums,-torch.inf)).exp()


def state_scan(x,log_a,b,c,chunk=64):
    """h[t]=exp(log_a[t])*h[t-1]+b[t] outer x[t]; y[t]=c[t]@h[t].

    x: N,H,P; log_a: N,H; b,c: N,H,S. Padding carries state unchanged.
    All within-chunk and between-chunk histories are retained exactly in real
    arithmetic; this is not a sliding window or reset-at-boundary recurrence.
    """
    n,heads,channels=x.shape;state=b.shape[-1]
    extra=(-n)%chunk;blocks=(n+extra)//chunk
    x=F.pad(x,(0,0,0,0,0,extra)).reshape(blocks,chunk,heads,channels).permute(0,2,1,3)
    b=F.pad(b,(0,0,0,0,0,extra)).reshape(blocks,chunk,heads,state).permute(0,2,1,3)
    c=F.pad(c,(0,0,0,0,0,extra)).reshape(blocks,chunk,heads,state).permute(0,2,1,3)
    a=F.pad(log_a,(0,0,0,extra)).reshape(blocks,chunk,heads).permute(0,2,1)
    decay=causal_decay(a)
    similarity=c@b.transpose(-1,-2)
    local=(similarity*decay)@x
    terminal=decay[:,:,-1,:]
    boundary=torch.einsum('khis,khip,khi->khsp',b,x,terminal)
    transitions=a.sum(-1).T
    block_decay=causal_decay(F.pad(transitions,(1,0)))
    padded_states=torch.cat([torch.zeros_like(boundary[:1]),boundary],dim=0).permute(1,0,2,3)
    incoming=torch.einsum('hkl,hlsp->hksp',block_decay,padded_states)[:,:-1].permute(1,0,2,3)
    prefix=a.new_ones((chunk,chunk)).tril()
    history=torch.einsum('khis,khsp->khip',c,incoming)*(a@prefix.T).exp()[...,None]
    return (local+history).permute(0,2,1,3).reshape(-1,heads,channels)[:n]


def scan_order(counts,seed,random_order=False):
    # A random tie key is attached to fixed node identities. The method is
    # equivariant in distribution, not an exact graph-isomorphism canonicalizer.
    degree=np.rint(np.expm1(counts)).sum(1)
    noise=np.random.default_rng(seed).random(len(degree))
    return np.argsort(noise) if random_order else np.lexsort((noise,degree))


class SelectiveMemory(nn.Module):
    def __init__(self,width,heads,state=8,mode='full',chunk=64):
        super().__init__()
        self.heads,self.channels,self.state=heads,width//heads,state
        self.mode,self.chunk=mode,chunk
        self.input=nn.Linear(width,2*width)
        self.bc=nn.Linear(width,2*heads*state,bias=False)
        self.dt=nn.Linear(width,heads)
        nn.init.uniform_(self.dt.weight,-.01,.01)
        nn.init.constant_(self.dt.bias,math.log(math.expm1(.01)))
        self.log_a=nn.Parameter(torch.linspace(1.,4.,heads).log())
        self.fixed_b=nn.Parameter(torch.randn(heads,state)*.2)
        self.fixed_c=nn.Parameter(torch.randn(heads,state)*.2)
        self.skip=nn.Parameter(torch.ones(heads))
        self.norm=nn.LayerNorm(width)
        self.output=nn.Linear(width,width)

    def forward(self,h):
        n=len(h)
        x,gate=self.input(h).chunk(2,dim=-1)
        x=x.reshape(n,self.heads,self.channels)
        if self.mode=='static':
            b=self.fixed_b.expand(n,-1,-1);c=self.fixed_c.expand(n,-1,-1)
            dt=F.softplus(self.dt.bias).expand(n,-1)
        else:
            b,c=self.bc(h).reshape(n,2,self.heads,self.state).unbind(1)
            dt=F.softplus(self.dt(h))
        log_a=-dt*self.log_a.clamp(-5.,5.).exp()
        b=b*dt[...,None]
        if self.mode=='no_state':
            y=(c*b).sum(-1)[...,None]*x
        else:
            forward=state_scan(x,log_a,b,c,self.chunk)
            backward=state_scan(x.flip(0),log_a.flip(0),b.flip(0),c.flip(0),self.chunk).flip(0)
            y=.5*(forward+backward)
        y=y+self.skip[None,:,None]*x
        return self.output(self.norm(y.flatten(1))*F.silu(gate))


class GraphMemoryLayer(nn.Module):
    def __init__(self,width,heads,edge_types,dropout,state,mode,chunk):
        super().__init__()
        self.local=SparseTypedAttention(width,heads,edge_types,dropout)
        self.pre=nn.LayerNorm(width)
        self.memory=SelectiveMemory(width,heads,state,mode,chunk)
        self.norm=nn.LayerNorm(width)
        self.dropout=nn.Dropout(dropout)

    def forward(self,h,edges,relations,order):
        local=self.local(h,edges,relations)
        memory=self.memory(self.pre(h)[order])
        restored=torch.empty_like(memory).index_copy(0,order,memory)
        return self.norm(local+self.dropout(restored))


class StateSpaceGraph(nn.Module):
    def __init__(self,dims,node_counts,relations,width=64,depth=3,heads=4,state=8,
                 dropout=.1,mode='full',chunk=64):
        super().__init__()
        if mode not in ('full','static','no_state','random_order'):raise ValueError('Unknown state-space control')
        self.node_counts=node_counts
        self.project=nn.ModuleList([nn.Linear(d+relations,width) for d in dims])
        self.input_norm=nn.LayerNorm(width)
        self.layers=nn.ModuleList([GraphMemoryLayer(width,heads,2*relations+2,dropout,state,
            'full' if mode=='random_order' else mode,chunk) for _ in range(depth)])
        self.rna_readout=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width//2))
        self.drug_readout=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width//2))
        self.rna_degree=nn.Sequential(nn.Linear(relations,32),nn.GELU(),nn.Linear(32,1))
        self.drug_degree=nn.Sequential(nn.Linear(relations,32),nn.GELU(),nn.Linear(32,1))
        self.scale=math.sqrt(width//2)

    def encode(self,xs,graph):
        edges,relations,counts,order=graph
        values=[];offset=0
        for x,project,n in zip(xs,self.project,self.node_counts):
            rms=x.square().mean(1,keepdim=True).clamp_min(1e-8).sqrt()
            values.append(project(torch.cat([x/rms,counts[offset:offset+n]],1)))
            offset+=n
        h=F.gelu(self.input_norm(torch.cat(values)))
        for layer in self.layers:h=layer(h,edges,relations,order)
        return h,counts

    def decode(self,h,counts,pairs):
        r,d=pairs
        interaction=(self.rna_readout(h[r])*self.drug_readout(h[d])).sum(-1)/self.scale
        return interaction+self.rna_degree(counts[r]).squeeze(-1)+self.drug_degree(counts[d]).squeeze(-1)

    def forward(self,xs,graph,pairs):
        return self.decode(*self.encode(xs,graph),pairs)
