"""Portable BioPathNet/NBFNet DistMult-PNA path layers for a frozen residual.

The message / aggregate / combine equations follow emyyue/BioPathNet at
b051a9003a1363b5655cdb7dfdad60609c0e4fe3. This adaptation uses ordinary PyTorch
scatter operations, no TorchDrug C++ extension, and a zero-initialized residual
head. See vendor/biopathnet_20260914/LICENSE and PROVENANCE.json.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def degree_preserving_shuffle(matrix,seed,swaps_per_edge=10):
    """Switch bipartite edges, preserving each RNA and each gene degree."""
    from scipy import sparse
    coo=matrix.tocoo();rows=coo.row.copy();cols=coo.col.copy()
    if len(np.unique(coo.data))!=1 or (len(coo.data) and coo.data[0]!=1):
        raise ValueError('This switch control requires a binary graph')
    edges=set(zip(rows.tolist(),cols.tolist()));original=edges.copy()
    if len(edges)!=len(rows):raise ValueError('Duplicate input edge')
    rng=np.random.default_rng(seed);accepted=0
    for _ in range(swaps_per_edge*len(rows)):
        i,j=rng.integers(len(rows),size=2)
        r,s=rows[i],rows[j];g,h=cols[i],cols[j]
        if r==s or g==h or (r,h) in edges or (s,g) in edges:continue
        edges.remove((r,g));edges.remove((s,h));edges.add((r,h));edges.add((s,g))
        cols[i],cols[j]=h,g;accepted+=1
    result=sparse.csr_matrix((np.ones(len(rows),dtype=np.float32),(rows,cols)),shape=matrix.shape)
    np.testing.assert_array_equal(np.asarray(matrix.sum(0)),np.asarray(result.sum(0)))
    np.testing.assert_array_equal(np.asarray(matrix.sum(1)),np.asarray(result.sum(1)))
    return result,dict(accepted_switches=accepted,edges=len(rows),
                       retained_original_edges=len(original&edges),changed_fraction=1-len(original&edges)/max(len(edges),1))


@dataclass
class PathContext:
    edge:torch.Tensor
    relation:torch.Tensor
    degree:torch.Tensor
    nodes:int


def context(a,fold,kind,device):
    query=a['pairs_train'][:,a['folds']==fold]
    omitted=query[0]*int(a['nd'])+query[1]
    keep=~np.isin(a['target_codes'],omitted)
    if kind=='no_brg':keep&=~a['background']
    edges=a['shuffled_edges'][:,keep] if kind=='shuffled_brg' else a['edges'][:,keep]
    rel=a['relation'][keep]
    if np.isin(a['target_codes'][keep],omitted).any():raise ValueError('Query fact remains before adding reverse edges')
    inverse=np.where(a['symmetric_relations'][rel],rel,rel+int(a['physical_relations']))
    e=torch.as_tensor(np.concatenate([edges,edges[::-1]],axis=1),device=device)
    r=torch.as_tensor(np.concatenate([rel,inverse]),device=device)
    n=int(a['base_nodes'] if kind=='no_brg' else a['nodes'])
    if e.numel() and (int(e.min())<0 or int(e.max())>=n):raise ValueError('Edge outside node dictionary')
    degree=torch.bincount(e[1],minlength=n).to(torch.float32)+1
    return PathContext(e,r,degree,n)


class PathConv(nn.Module):
    """DistMult messages, PNA moments/scalers, source boundary every layer."""
    def __init__(self,width,relations):
        super().__init__();self.width=width;self.relations=relations
        self.linear=nn.Linear(width*13,width)
        self.relation_linear=nn.Linear(width,relations*width)
        self.layer_norm=nn.LayerNorm(width)

    def forward(self,ctx,h,boundary,query):
        b=query.shape[0]
        relation=self.relation_linear(query).reshape(b,self.relations,self.width).transpose(0,1)
        messages=h[ctx.edge[0]]*relation[ctx.relation]
        indices=ctx.edge[1][:,None,None].expand_as(messages)
        total=boundary.clone().scatter_add_(0,indices,messages)
        square=boundary.square().scatter_add_(0,indices,messages.square())
        high=boundary.clone().scatter_reduce_(0,indices,messages,reduce='amax',include_self=True)
        low=boundary.clone().scatter_reduce_(0,indices,messages,reduce='amin',include_self=True)
        degree=ctx.degree[:,None,None]
        mean=total/degree
        std=(square/degree-mean.square()).clamp_min(1e-6).sqrt()
        features=torch.stack([mean,high,low,std],dim=-1).flatten(-2)
        log_degree=degree.log();scale=log_degree/log_degree.mean().clamp_min(1e-12)
        scalers=torch.cat([torch.ones_like(scale),scale,1/scale.clamp_min(1e-2)],dim=-1)
        update=(features.unsqueeze(-1)*scalers.unsqueeze(-2)).flatten(-2)
        return F.relu(self.layer_norm(self.linear(torch.cat([h,update],dim=-1))))


class PathResidual(nn.Module):
    def __init__(self,relations,width=32,depth=6):
        super().__init__();self.width=width
        self.query=nn.Embedding(relations,width)
        self.layers=nn.ModuleList([PathConv(width,relations) for _ in range(depth)])
        self.mlp=nn.Sequential(nn.Linear(2*width,2*width),nn.ReLU(),nn.Linear(2*width,1))
        nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)

    def forward(self,ctx,sources,query_relation,rna_offset,rna_count):
        indices=torch.full_like(sources,query_relation)
        query=self.query(indices)
        boundary=torch.zeros(ctx.nodes,len(sources),self.width,device=query.device,dtype=query.dtype)
        boundary[sources,torch.arange(len(sources),device=query.device)]=query
        h=boundary
        for layer in self.layers:h=h+layer(ctx,h,boundary,query)
        h=h[rna_offset:rna_offset+rna_count]
        features=torch.cat([h,query[None].expand(rna_count,-1,-1)],dim=-1)
        return self.mlp(features).squeeze(-1).transpose(0,1)
