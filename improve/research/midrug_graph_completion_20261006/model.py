"""First-layer GNN soft completion with exact query-dependent row/column repair."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,Reconstruction,make_graph
from incrd_hg_m123.model import Model

CONFIGS={f'{mode}{suffix}':dict(mode=mode,reconstruction_weight=weight,auxiliary_weight=.1)
    for mode in ('direct','uniform','soft') for suffix,weight in [('',0.),('_rec10',10.)]}

def rectangle_sum(X,Y,Z):
    return X@(Y.t()@Z)-(X*Y).sum(1,keepdim=True)*Z-X*(Y*Z).sum(0,keepdim=True)+X*Y*Z

@torch.no_grad()
def completion_fields(A,known,probability,mode):
    """Used for the uniform structural control and coverage only."""
    dtype=A.dtype;A=A.double();unknown=(~known).double()
    if mode=='direct':return A.new_zeros((*A.shape,3),dtype=dtype)
    U=unknown*(probability.double() if mode=='soft' else .5)
    values=torch.stack((rectangle_sum(U,A,A),rectangle_sum(A,U,A),rectangle_sum(A,A,U)),dim=-1)
    return values.clamp_min(0).log1p().to(dtype)

@torch.no_grad()
def query_fields(A,K,P,Prow,Pcol,r,d,mode):
    if mode!='soft':return completion_fields(A,K,P,mode)[r,d]
    dtype=A.dtype;A=A.double();P=P.double();row=Prow.double();col=Pcol.double()
    q=A[r,d,None];batch=torch.arange(len(r),device=A.device)
    # Remove the intermediate endpoint, then remove the other repeated endpoint.
    wl=(A.t()@A)[d]-q*A[r];wr=(A@A.t())[r]-q*A[:,d].t()
    wl[batch,d]=0;wr[batch,r]=0
    ul=(~K[r])*row;ur=(~K[:,d].t())*col;ul[batch,d]=0;ur[batch,r]=0
    left=(ul*wl).sum(1);right=(ur*wr).sum(1)
    # Both intermediate states are unchanged by deleting (r,d) in layer one.
    middle=rectangle_sum(A,(~K)*P,A)[r,d]
    return torch.stack((left,middle,right),dim=-1).clamp_min(0).log1p().to(dtype)

def first_states(encoder,dense,r,d,A):
    p=encoder.layer(encoder.original.conv1,encoder.graph.x_dict)
    hm,hd=F.leaky_relu(p['miRNA'],.1),F.leaky_relu(p['drug'],.1)
    c=encoder.original.conv1;q=A[r,d,None]
    qr=F.leaky_relu(p['miRNA'][r]-q*c['drug__rev_MiDrug__miRNA'].lin_msg(dense[1])[d]/encoder.counts['miRNA'],.1)
    qd=F.leaky_relu(p['drug'][d]-q*c['miRNA__MiDrug__drug'].lin_msg(dense[0])[r]/encoder.counts['drug'],.1)
    return hm,hd,qr,qd

class PairScorer(nn.Module):
    def __init__(self,hidden,rank=32):
        super().__init__();self.rank=rank
        self.mi=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,128),nn.LeakyReLU(.1),nn.Linear(128,rank))
        self.drug=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,128),nn.LeakyReLU(.1),nn.Linear(128,rank))
        self.mi_bias=nn.Linear(rank,1);self.drug_bias=nn.Linear(rank,1)
    def matrix(self,r,d):return r@d.t()/math.sqrt(self.rank)+self.mi_bias(r)+self.drug_bias(d).t()
    def pairs(self,r,d):return (r*d).sum(1)/math.sqrt(self.rank)+self.mi_bias(r)[:,0]+self.drug_bias(d)[:,0]

class SoftPaths(nn.Module):
    def __init__(self,hidden,mode):
        super().__init__();self.mode=mode;self.scorer=PairScorer(hidden)
        self.head=nn.Sequential(nn.Linear(4,64),nn.LeakyReLU(.1),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.last_query_logits=None
    def features(self,encoder,ps,dense,r,d):
        hm,hd,qr,qd=first_states(encoder,dense,r,d,ps.A);s=self.scorer
        mr,dr,qr,qd=s.mi(hm),s.drug(hd),s.mi(qr),s.drug(qd)
        direct=s.pairs(qr,qd)
        # No backward graph for predicted completion weights; direct and
        # auxiliary supervision still train the scorer and shared first layer.
        with torch.no_grad():
            P=s.matrix(mr,dr).sigmoid()
            if self.mode=='soft':row=s.matrix(qr,dr).sigmoid();col=s.matrix(mr,qd).t().sigmoid()
            else:row=col=None
            fields=query_fields(ps.A,ps.known,P,row,col,r,d,self.mode)
        return direct,fields
    def forward(self,encoder,ps,dense,r,d):
        direct,fields=self.features(encoder,ps,dense,r,d);self.last_query_logits=direct
        return self.head(torch.cat((direct[:,None],fields),dim=-1)).view(-1)

class SoftPathModel(Model):
    def forward(self,eli,ps,qis,ease,dense):
        return super().forward(eli,ps,qis,ease,dense)+self.soft_paths(self.encoder,ps,dense,*eli)

def configure(a,data,cfg):
    known=torch.zeros_like(a.ps.A,dtype=torch.bool);known[a.pairs['train'][0],a.pairs['train'][1]]=True
    assert bool(known[a.ps.A>0].all());a.ps.known=known
    a.model.__class__=SoftPathModel;a.model.add_module('soft_paths',SoftPaths(a.cfg.hidden,cfg['mode']).to(a.y.device))
    return a

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    cfg=dict(reconstruction_weight=10.) if name=='unmasked10' else CONFIGS[name]
    if cfg['reconstruction_weight']:a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a if name=='unmasked10' else configure(a,data,cfg)

def auxiliary_loss(a):
    assert a.model.soft_paths.last_query_logits is not None
    return F.binary_cross_entropy_with_logits(a.model.soft_paths.last_query_logits,a.y)

def mask_generator(device,fold):return torch.Generator(device=device).manual_seed(20261006+fold)
