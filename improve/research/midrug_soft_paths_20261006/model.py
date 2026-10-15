"""Two observed anchors and one softly completed bipartite edge."""
import math
import torch
from torch import nn
from common import Adapter,Reconstruction,make_graph
from incrd_hg_m123.model import Model

CONFIGS={f'{mode}{suffix}':dict(mode=mode,reconstruction_weight=weight,auxiliary_weight=.1)
    for mode in ('direct','uniform','soft') for suffix,weight in [('',0.),('_rec10',10.)]}

def rectangle_sum(X,Y,Z):
    """Sum X[r,j]*Y[i,j]*Z[i,d], excluding i=r and j=d, for every r,d."""
    return X@(Y.t()@Z)-(X*Y).sum(1,keepdim=True)*Z-X*(Y*Z).sum(0,keepdim=True)+X*Y*Z

@torch.no_grad()
def completion_fields(A,known,probability,mode):
    """Soft weights are detached, like the recursive prediction in NCNC.

    This is an independent bipartite adaptation; it does not use NCNC's
    recursive GNN predictor or its probability rescaling formula.
    """
    dtype=A.dtype;A=A.double();unknown=(~known).double()
    if mode=='direct':return A.new_zeros((*A.shape,3),dtype=dtype)
    weight=probability.double() if mode=='soft' else .5
    U=unknown*weight
    values=torch.stack((rectangle_sum(U,A,A),rectangle_sum(A,U,A),rectangle_sum(A,A,U)),dim=-1)
    # Roundoff alone can put theoretical zeros slightly below zero.
    return values.clamp_min(0).log1p().to(dtype)

class PairScorer(nn.Module):
    def __init__(self,mi_dim,drug_dim,rank=32):
        super().__init__();self.rank=rank
        self.mi=nn.Sequential(nn.LayerNorm(mi_dim),nn.Linear(mi_dim,128),nn.LeakyReLU(.1),nn.Linear(128,rank))
        self.drug=nn.Sequential(nn.LayerNorm(drug_dim),nn.Linear(drug_dim,128),nn.LeakyReLU(.1),nn.Linear(128,rank))
        self.mi_bias=nn.Linear(rank,1);self.drug_bias=nn.Linear(rank,1)
    def forward(self,mi,drug):
        r,d=self.mi(mi),self.drug(drug)
        return r@d.t()/math.sqrt(self.rank)+self.mi_bias(r)+self.drug_bias(d).t()

class SoftPaths(nn.Module):
    def __init__(self,mi_dim,drug_dim,mode):
        super().__init__();self.mode=mode;self.scorer=PairScorer(mi_dim,drug_dim)
        self.head=nn.Sequential(nn.Linear(4,64),nn.LeakyReLU(.1),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.last_pair_logits=None
    def forward(self,ps,dense,r,d):
        logits=self.scorer(dense[0],dense[1]);self.last_pair_logits=logits
        fields=completion_fields(ps.A,ps.known,logits.sigmoid(),self.mode)
        features=torch.cat((logits[r,d,None],fields[r,d]),dim=-1)
        return self.head(features).view(-1)

class SoftPathModel(Model):
    def forward(self,eli,ps,qis,ease,dense):
        return super().forward(eli,ps,qis,ease,dense)+self.soft_paths(ps,dense,*eli)

def configure(a,data,cfg):
    known=torch.zeros_like(a.ps.A,dtype=torch.bool);known[a.pairs['train'][0],a.pairs['train'][1]]=True
    assert bool(known[a.ps.A>0].all());a.ps.known=known
    a.model.__class__=SoftPathModel
    a.model.add_module('soft_paths',SoftPaths(a.dense[0].shape[1],a.dense[1].shape[1],cfg['mode']).to(a.y.device))
    return a

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    cfg=dict(reconstruction_weight=10.) if name=='unmasked10' else CONFIGS[name]
    if cfg['reconstruction_weight']:a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a if name=='unmasked10' else configure(a,data,cfg)

def auxiliary_loss(a):
    logits=a.model.soft_paths.last_pair_logits
    assert logits is not None
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[a.pairs['train'][0],a.pairs['train'][1]],a.y)

def mask_generator(device,fold):return torch.Generator(device=device).manual_seed(20261006+fold)
