"""CGLE-inspired smoothed directional group frequencies, exact query exclusion."""
import json
import numpy as np
import torch
from torch import nn
from common import HERE,Adapter,Reconstruction,make_graph,sha256
from incrd_hg_m123.model import Model

CONFIGS={f'{mode}{suffix}':dict(mode=mode,reconstruction_weight=weight)
    for mode in ('feature','shuffled') for suffix,weight in [('',0.),('_rec10',10.)]}

class GroupStats:
    def __init__(self,A,mi,drug,alpha=10.):
        self.A=A;self.mi=torch.as_tensor(mi,device=A.device,dtype=torch.long)
        self.drug=torch.as_tensor(drug,device=A.device,dtype=torch.long);self.alpha=alpha
        assert len(mi)==A.shape[0] and len(drug)==A.shape[1] and alpha>0
        self.nr=int(self.mi.max())+1;self.nd=int(self.drug.max())+1
        self.pr=torch.bincount(self.mi,minlength=self.nr).double()/len(mi)
        self.pd=torch.bincount(self.drug,minlength=self.nd).double()/len(drug)
        assert bool((self.pr>0).all()) and bool((self.pd>0).all())
        r,d=torch.where(A>0)
        self.table=torch.bincount(self.mi[r]*self.nd+self.drug[d],minlength=self.nr*self.nd).reshape(self.nr,self.nd).double()
        self.rows=self.table.sum(1);self.cols=self.table.sum(0)

    def features(self,r,d):
        cr,cd=self.mi[r],self.drug[d];edge=self.A[r,d].double()
        block=self.table[cr,cd]-edge;row=self.rows[cr]-edge;col=self.cols[cd]-edge
        # Dirichlet reference measure is the target group's node fraction.
        # Return log enrichment, not a claim of calibrated pair probability.
        forward=(block+self.alpha*self.pd[cd])/(row+self.alpha)/self.pd[cd]
        reverse=(block+self.alpha*self.pr[cr])/(col+self.alpha)/self.pr[cr]
        return torch.stack((forward.log(),reverse.log()),dim=-1).to(self.A.dtype)

class PriorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,64),nn.LeakyReLU(.1),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x).view(-1)

class PriorModel(Model):
    def forward(self,eli,ps,qis,ease,dense):
        return super().forward(eli,ps,qis,ease,dense)+self.class_prior(ps.class_stats.features(*eli))

def configure(a,cfg,groups=None):
    if groups is None:
        manifest=json.loads((HERE/'groups.json').read_text())
        assert sha256(HERE/'groups.npz')==manifest['groups_sha256']
        prefix='shuffled_' if cfg['mode']=='shuffled' else ''
        with np.load(HERE/'groups.npz') as z:groups=(z[prefix+'miRNA'],z[prefix+'drug'])
    a.ps.class_stats=GroupStats(a.ps.A,*groups)
    a.model.__class__=PriorModel;a.model.add_module('class_prior',PriorHead().to(a.y.device))
    return a

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    cfg=dict(reconstruction_weight=10.) if name=='unmasked10' else CONFIGS[name]
    if cfg['reconstruction_weight']:a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a if name=='unmasked10' else configure(a,cfg)

def mask_generator(device,fold):return torch.Generator(device=device).manual_seed(20261006+fold)
