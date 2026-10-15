"""Nonlinear query-masked interaction profiles as a residual M123 branch."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
RELEASE=HERE.parents[2]/'release/iNcRD-HG-M1M2M3'
sys.path.insert(0,str(RELEASE))
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.model import Model
from incrd_hg_m123.data import build_graph,load_fold,sha256
from incrd_hg_m123.engine import seed_all

CONFIGS={
    'control':dict(channels=0,hidden=128,drop=0.,standalone=False,aux=0.),
    'pos128_d0':dict(channels=1,hidden=128,drop=0.,standalone=False,aux=.1),
    'pos128_d2':dict(channels=1,hidden=128,drop=.2,standalone=False,aux=.1),
    'pos256_d5':dict(channels=1,hidden=256,drop=.5,standalone=False,aux=.1),
    'signed128_d2':dict(channels=2,hidden=128,drop=.2,standalone=False,aux=.1),
    'signed256_d5':dict(channels=2,hidden=256,drop=.5,standalone=False,aux=.1),
    'only_pos256_d5':dict(channels=1,hidden=256,drop=.5,standalone=True,aux=0.),
    'only_signed256_d5':dict(channels=2,hidden=256,drop=.5,standalone=True,aux=0.),
}


def negative_matrix(data,shape,device):
    n=torch.zeros(shape,device=device)
    idx=torch.as_tensor(data['train_pairs'][data['train_y']==0],device=device)
    n[idx[:,0],idx[:,1]]=1
    return n


class MaskedProfiles(nn.Module):
    def __init__(self,nr,nd,cfg):
        super().__init__()
        self.channels=cfg['channels'];self.drop=cfg['drop']
        h=cfg['hidden']
        def encoder(n):
            return nn.Sequential(nn.Linear(n*self.channels,h),nn.LayerNorm(h),nn.GELU(),
                nn.Dropout(.1),nn.Linear(h,h),nn.GELU())
        self.renc,self.denc=encoder(nd),encoder(nr)
        self.rout,self.dout=nn.Linear(h,nd),nn.Linear(h,nr)
        # Zero residual at initialization; same underlying M123 initialization.
        nn.init.zeros_(self.rout.weight);nn.init.zeros_(self.rout.bias)
        nn.init.zeros_(self.dout.weight);nn.init.zeros_(self.dout.bias)

    def masked_inputs(self,A,N,r,d):
        ix=torch.arange(len(r),device=A.device)
        xr,xd=A[r].clone(),A.T[d].clone()
        xr[ix,d]=0;xd[ix,r]=0
        if self.channels==2:
            nr,nd=N[r].clone(),N.T[d].clone()
            nr[ix,d]=0;nd[ix,r]=0
            xr,xd=torch.cat([xr,nr],1),torch.cat([xd,nd],1)
        return xr,xd

    def forward(self,A,N,r,d):
        xr,xd=self.masked_inputs(A,N,r,d)
        xr=F.dropout(xr,self.drop,self.training);xd=F.dropout(xd,self.drop,self.training)
        hr,hd=self.renc(xr),self.denc(xd)
        a=(hr*self.rout.weight[d]).sum(1)+self.rout.bias[d]
        b=(hd*self.dout.weight[r]).sum(1)+self.dout.bias[r]
        return .5*(a+b)


class ProfileModel(Model):
    def __init__(self,graph,ps,nq,ne,cfg,experiment,N):
        super().__init__(graph,ps,nq,ne,cfg)
        self.experiment=experiment
        self.profile=MaskedProfiles(*ps.A.shape,experiment) if experiment['channels'] else None
        self.register_buffer('N',N,persistent=False)
        self.profile_logits=None

    def forward(self,eli,ps,qis,ease,dense):
        if self.profile is None:return super().forward(eli,ps,qis,ease,dense)
        self.profile_logits=self.profile(ps.A,self.N,*eli)
        if self.experiment['standalone']:return self.profile_logits
        return super().forward(eli,ps,qis,ease,dense)+self.profile_logits


def make_adapter(data,cfg,device):
    inputs={k:data[k] for k in ('train_pairs','train_y','val_pairs')}
    inputs['test_pairs']=np.empty((0,2),dtype=np.int64)
    adapter=Adapter(inputs,build_graph(inputs,'MiDrug'),device)
    initial=adapter.model.state_dict()
    model=ProfileModel(adapter.graph,adapter.ps,adapter.qis['train'].shape[1],adapter.ease['train'].shape[1],
        adapter.cfg,cfg,negative_matrix(inputs,adapter.ps.A.shape,device)).to(device)
    missing,unexpected=model.load_state_dict(initial,strict=False)
    assert not unexpected and all(k.startswith('profile.') for k in missing)
    adapter.model=model
    return adapter
