"""M123 plus a query-deleted enclosing-subgraph GNN, miRNA/drug only."""
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import HERE,CLEAN,Adapter,Model,load_inner,make_graph,seed_all,sha256

CONFIGS={
    'sg1_struct':dict(k=16,hops=1,hidden=32,layers=3,covariates=False,signed=False),
    'sg1_cov':dict(k=16,hops=1,hidden=32,layers=3,covariates=True,signed=False),
    'sg1_signed':dict(k=16,hops=1,hidden=32,layers=3,covariates=True,signed=True),
    'sg2_cov':dict(k=32,hops=2,hidden=32,layers=3,covariates=True,signed=False),
}


class GraphCache:
    def __init__(self,data,cfg,fold,device):
        path=HERE/'cache'/f"f{fold}_h{cfg['hops']}_k{cfg['k']}.npz"
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['source_sha256']==sha256(HERE/'extract.py') and meta['cache_sha256']==sha256(path)
        assert meta['inner_data_sha256']==sha256(CLEAN/'inner_data.npz')
        with np.load(path) as z:
            self.data={key:torch.as_tensor(np.concatenate([z['train_'+key],z['val_'+key]]),device=device,
                dtype=torch.long if key in ('rid','did') else torch.float32) for key in ('rid','did','features','pos','neg')}
        pairs=np.concatenate([data['train_pairs'],data['val_pairs']]);assert len(np.unique(pairs,axis=0))==len(pairs)
        self.lookup=torch.full((605,216),-1,dtype=torch.long,device=device)
        p=torch.as_tensor(pairs,device=device);self.lookup[p[:,0],p[:,1]]=torch.arange(len(p),device=device)

    def select(self,r,d):
        ix=self.lookup[r,d];assert (ix>=0).all()
        return {key:value[ix] for key,value in self.data.items()}


class LocalLayer(nn.Module):
    def __init__(self,h,signed):
        super().__init__();self.signed=signed
        self.rr,self.dd=nn.Linear(h,h),nn.Linear(h,h)
        self.dr,self.rd=nn.Linear(h,h,bias=False),nn.Linear(h,h,bias=False)
        self.nr,self.nd=(nn.Linear(h,h,bias=False),nn.Linear(h,h,bias=False)) if signed else (None,None)
        self.rnorm,self.dnorm=nn.LayerNorm(h),nn.LayerNorm(h)

    def forward(self,hr,hd,pos,neg,mr,md):
        pr=torch.bmm(pos,hd)/pos.sum(2,keepdim=True).clamp(min=1)
        pd=torch.bmm(pos.transpose(1,2),hr)/pos.sum(1).unsqueeze(2).clamp(min=1)
        r,d=self.rr(hr)+self.dr(pr),self.dd(hd)+self.rd(pd)
        if self.signed:
            nr=torch.bmm(neg,hd)/neg.sum(2,keepdim=True).clamp(min=1)
            nd=torch.bmm(neg.transpose(1,2),hr)/neg.sum(1).unsqueeze(2).clamp(min=1)
            r=r+self.nr(nr);d=d+self.nd(nd)
        return F.leaky_relu(self.rnorm(r),.1)*mr,F.leaky_relu(self.dnorm(d),.1)*md


class SubgraphGNN(nn.Module):
    def __init__(self,cfg,xr_dim,xd_dim):
        super().__init__();h=cfg['hidden'];self.k=cfg['k'];self.covariates=cfg['covariates']
        self.role=nn.Linear(20,h)
        self.rproj=nn.Sequential(nn.LayerNorm(xr_dim),nn.Linear(xr_dim,h)) if self.covariates else None
        self.dproj=nn.Sequential(nn.LayerNorm(xd_dim),nn.Linear(xd_dim,h)) if self.covariates else None
        self.layers=nn.ModuleList([LocalLayer(h,cfg['signed']) for _ in range(cfg['layers'])])
        self.head=nn.Sequential(nn.Linear(6*h,64),nn.LeakyReLU(.1),nn.Dropout(.1),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,batch,xr,xd):
        k=self.k;rid,did=batch['rid'],batch['did']
        mr,md=(rid<605).float().unsqueeze(2),(did<216).float().unsqueeze(2)
        h=self.role(batch['features']);hr,hd=h[:,:k],h[:,k:]
        if self.covariates:
            r0,d0=self.rproj(xr),self.dproj(xd)
            r0=torch.cat([r0,torch.zeros_like(r0[:1])]);d0=torch.cat([d0,torch.zeros_like(d0[:1])])
            hr,hd=hr+r0[rid],hd+d0[did]
        hr,hd=hr*mr,hd*md
        for layer in self.layers:
            hr,hd=layer(hr,hd,batch['pos'],batch['neg'],mr,md)
            hr=F.dropout(hr,.1,self.training);hd=F.dropout(hd,.1,self.training)
        pooled=[hr[:,0],hd[:,0],(hr*mr).sum(1)/mr.sum(1),(hd*md).sum(1)/md.sum(1),
            hr.masked_fill(mr==0,-torch.inf).max(1).values,hd.masked_fill(md==0,-torch.inf).max(1).values]
        return self.head(torch.cat(pooled,1)).view(-1)


class SubgraphModel(Model):
    def __init__(self,graph,ps,nq,ne,cfg,exp,cache):
        super().__init__(graph,ps,nq,ne,cfg);self.cache=cache
        self.subgraph=SubgraphGNN(exp,graph['miRNA'].x.shape[1],graph['drug'].x.shape[1])
    def forward(self,eli,ps,qis,ease,dense):
        return super().forward(eli,ps,qis,ease,dense)+self.subgraph(self.cache.select(*eli),dense[0],dense[1])


def make_adapter(data,name,fold,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    exp=CONFIGS[name];initial=a.model.state_dict();cache=GraphCache(data,exp,fold,device)
    model=SubgraphModel(a.graph,a.ps,a.qis['train'].shape[1],a.ease['train'].shape[1],a.cfg,exp,cache).to(device)
    missing,unexpected=model.load_state_dict(initial,strict=False)
    assert not unexpected and all(key.startswith('subgraph.') for key in missing)
    a.model=model
    return a
