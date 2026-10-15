"""Original M123 plus an auxiliary-supervised covariate interaction branch."""
import math
import torch
from torch import nn
from data import HERE,RELEASE,build_graph,load_inner,sha256
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.model import Model
from incrd_hg_m123.engine import seed_all

CONFIGS={
    'control':dict(rank=0,aux_weight=0.,shared=True),
    'raw32':dict(rank=32,aux_weight=0.,shared=True),
    'shared32_a01':dict(rank=32,aux_weight=.1,shared=True),
    'shared32_a03':dict(rank=32,aux_weight=.3,shared=True),
    'shared128_a01':dict(rank=128,aux_weight=.1,shared=True),
    'drugshared32_a01':dict(rank=32,aux_weight=.1,shared=False),
}


class CovariateBranch(nn.Module):
    def __init__(self,graph,exp):
        super().__init__();rank=exp['rank'];self.rank=rank;self.shared=exp['shared']
        assert graph['miRNA'].x.shape[1]==graph['lncRNA'].x.shape[1]
        def encoder(width):
            return nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,rank))
        self.rna=encoder(graph['miRNA'].x.shape[1])
        self.lnc=None if self.shared else encoder(graph['lncRNA'].x.shape[1])
        self.drug=encoder(graph['drug'].x.shape[1])
        self.rbias,self.dbias=nn.Linear(rank,1),nn.Linear(rank,1)
        self.task_bias=nn.Parameter(torch.full((2,),-math.log(2)))

    def forward(self,xr,xd,pairs,task):
        r,d=pairs
        er=(self.rna if task==0 or self.shared else self.lnc)(xr)
        ed=self.drug(xd)
        scores=(er[r]*ed[d]).sum(1)/math.sqrt(self.rank)
        scores=scores+self.rbias(er)[r,0]+self.dbias(ed)[d,0]+self.task_bias[task]
        return scores,er.square().mean()+ed.square().mean()


class MultiTaskModel(Model):
    def __init__(self,graph,ps,nq,ne,cfg,exp):
        super().__init__(graph,ps,nq,ne,cfg)
        self.branch=CovariateBranch(graph,exp) if exp['rank'] else None
        self.branch_penalty=None

    def forward(self,eli,ps,qis,ease,dense):
        base=super().forward(eli,ps,qis,ease,dense)
        if self.branch is None:
            self.branch_penalty=base.new_zeros(());return base
        score,self.branch_penalty=self.branch(dense[0],dense[1],eli,0)
        return base+.1*score

    def auxiliary(self,graph,pairs):
        scores,penalty=self.branch(graph['lncRNA'].x,graph['drug'].x,pairs,1)
        return scores,penalty


def make_adapter(data,exp,device):
    adapter=Adapter(data,build_graph(data,'MiDrug'),device)
    initial=adapter.model.state_dict()
    model=MultiTaskModel(adapter.graph,adapter.ps,adapter.qis['train'].shape[1],adapter.ease['train'].shape[1],adapter.cfg,exp).to(device)
    missing,unexpected=model.load_state_dict(initial,strict=False)
    assert not unexpected and all(k.startswith('branch.') for k in missing)
    adapter.model=model
    return adapter
