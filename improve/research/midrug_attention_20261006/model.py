"""M2 query-conditioned path attention with exact query-edge exclusion."""
import sys
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
RELEASE=HERE.parents[2]/"release/iNcRD-HG-M1M2M3"
sys.path.insert(0,str(RELEASE))
from incrd_hg_m123.model import Model
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.pair_channels import PairStruct
from incrd_hg_m123.data import load_fold,build_graph,sha256
from incrd_hg_m123.engine import seed_all

CONFIGS={
    "baseline":dict(rank=0,power=0.,temperature=1.,reg=0.),
    "degree05":dict(rank=0,power=.5,temperature=1.,reg=0.),
    "attn16":dict(rank=16,power=0.,temperature=1.,reg=.0001),
    "attn16_deg":dict(rank=16,power=.5,temperature=1.,reg=.0001),
    "attn64_deg":dict(rank=64,power=.5,temperature=1.,reg=.0001),
    "attn16_t3":dict(rank=16,power=.5,temperature=3.,reg=.0001),
    "attn64_t3":dict(rank=64,power=.5,temperature=3.,reg=.0001),
    "attn16_reg":dict(rank=16,power=.5,temperature=1.,reg=.01),
}


class AttentivePairStruct(PairStruct):
    def __init__(self,data,msg_pos,cfg,power=0.):
        super().__init__(data,msg_pos,cfg)
        self.wr=self.deg_r.clamp(min=1).pow(-power)
        self.wd=self.deg_d.clamp(min=1).pow(-power)
        self.Cmw=(self.A*self.wd)@self.A.T
        self.Cdw=self.A.T@(self.wr[:,None]*self.A)
        self.gates=None

    def qncn(self,r,d,z,corr=None):
        q=self.qvals(r,d)[:,None]
        if corr is not None:
            cm_dm=.5*F.linear(corr["dh_d"],corr["W_dm"])
            cm_mm=.5*F.linear(corr["dh_m"],corr["W_mm"])
            cd_md=.5*F.linear(corr["dh_m"],corr["W_md"])
            cd_dd=.5*F.linear(corr["dh_d"],corr["W_dd"])
        def pool_m(w):
            den=w.sum(1,keepdim=True)+1
            value=w@z["miRNA"]/den
            if corr is not None:
                value=value+cm_dm*(w.sum(1,keepdim=True)/den)+cm_mm*((w*self.Sr[:,r].T).sum(1,keepdim=True)/den)
            return value
        def pool_d(w):
            den=w.sum(1,keepdim=True)+1
            value=w@z["drug"]/den
            if corr is not None:
                value=value+cd_md*(w.sum(1,keepdim=True)/den)+cd_dd*((w*self.Sd[:,d].T).sum(1,keepdim=True)/den)
            return value
        Ad,Ar=self.A[:,d].T,self.A[r]
        wm=Ad*(self.Cmw[r]-q*Ad*self.wd[d,None])*self.wr
        wd=Ar*(self.Cdw[:,d].T-q*Ar*self.wr[r,None])*self.wd
        ix=torch.arange(len(r),device=r.device)
        wm[ix,r],wd[ix,d]=0,0
        weights=[wm,wd,Ad*self.Sr[r]*self.wr,Ar*self.Sd[:,d].T*self.wd]
        if self.gates is not None:weights=[w*g for w,g in zip(weights,self.gates)]
        return [pool_m(weights[0]),pool_d(weights[1]),pool_m(weights[2]),pool_d(weights[3])]


class PathAttention(nn.Module):
    """Gates depend on covariates, never on the query edge or its label."""
    def __init__(self,nr,nd,rank,temperature):
        super().__init__()
        self.rank,self.temperature=rank,temperature
        self.rnorm,self.dnorm=nn.LayerNorm(nr),nn.LayerNorm(nd)
        self.qr,self.qd=nn.Linear(nr,4*rank,bias=False),nn.Linear(nd,4*rank,bias=False)
        self.kr,self.kd=nn.Linear(nr,2*rank,bias=False),nn.Linear(nd,2*rank,bias=False)
        # Start with unit gates, matching the static-path control exactly.
        nn.init.zeros_(self.qr.weight);nn.init.zeros_(self.qd.weight)

    def forward(self,xr,xd,r,d):
        xr,xd=self.rnorm(xr),self.dnorm(xd)
        q=(self.qr(xr)[r]+self.qd(xd)[d]).view(len(r),4,self.rank)
        kr=self.kr(xr).view(len(xr),2,self.rank)
        kd=self.kd(xd).view(len(xd),2,self.rank)
        logs=[self.temperature*torch.tanh((q[:,j]@key.T)/math.sqrt(self.rank))
              for j,key in enumerate((kr[:,0],kd[:,0],kr[:,1],kd[:,1]))]
        return [s.exp() for s in logs],sum(s.square().mean() for s in logs)/4


class AttentiveModel(Model):
    def __init__(self,graph,ps,n_qis,n_ease,cfg,experiment):
        super().__init__(graph,ps,n_qis,n_ease,cfg)
        self.attention=PathAttention(graph["miRNA"].x.shape[1],graph["drug"].x.shape[1],
            experiment["rank"],experiment["temperature"]) if experiment["rank"] else None
        self.attention_penalty=None

    def forward(self,eli,ps,qis,ease,dense):
        if self.attention is None:
            self.attention_penalty=qis.new_zeros(())
            return super().forward(eli,ps,qis,ease,dense)
        gates,self.attention_penalty=self.attention(dense[0],dense[1],*eli)
        ps.gates=gates
        try:return super().forward(eli,ps,qis,ease,dense)
        finally:ps.gates=None


def make_adapter(data,config,device):
    inputs={k:data[k] for k in ("train_pairs","train_y","val_pairs")}
    inputs["test_pairs"]=np.empty((0,2),dtype=np.int64)
    adapter=Adapter(inputs,build_graph(inputs,"MiDrug"),device)
    initial=adapter.model.state_dict()
    ps=AttentivePairStruct(adapter.D,adapter.graph["miRNA","MiDrug","drug"].edge_index.T,adapter.cfg,config["power"])
    model=AttentiveModel(adapter.graph,ps,adapter.qis["train"].shape[1],adapter.ease["train"].shape[1],adapter.cfg,config).to(device)
    missing,unexpected=model.load_state_dict(initial,strict=False)
    assert not unexpected and all(k.startswith("attention.") for k in missing)
    adapter.model,adapter.ps=model,ps
    return adapter
