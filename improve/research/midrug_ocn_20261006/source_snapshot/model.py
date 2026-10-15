"""M123 plus exactly query-deleted pooled 3/5-walk encoder states."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,CLEAN,Adapter,Model,load_inner,make_graph,seed_all,sha256
from features import CONFIGS

class Cache:
    def __init__(self,name,fold,device):
        path=HERE/'cache'/f'{name}_f{fold}.npz';meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['cache_sha256']==sha256(path) and meta['features_sha256']==sha256(HERE/'features.py')
        assert meta['data_sha256']==sha256(CLEAN/'inner_data.npz')
        with np.load(path) as z:
            p=torch.as_tensor(z['pairs'],device=device)
            self.weights=torch.as_tensor(z['weights'],device=device)
            self.summary=torch.as_tensor(z['summary'],device=device)
        self.lookup=torch.full((605,216),-1,dtype=torch.long,device=device)
        assert len(torch.unique(p,dim=0))==len(p)
        self.lookup[p[:,0],p[:,1]]=torch.arange(len(p),device=device)
    def select(self,r,d):
        i=self.lookup[r,d];assert (i>=0).all()
        return self.weights[i],self.summary[i]

def corrected_pool(w,r,d,zr,zd,corr,dense):
    """Arbitrary zero-root coefficient vectors pooled on the deleted graph."""
    _,_,Sr1,Sd1,A=dense;wr,wd=w[:,:,:605],w[:,:,605:]
    cm_dm=.5*F.linear(corr['dh_d'],corr['W_dm']);cm_mm=.5*F.linear(corr['dh_m'],corr['W_mm'])
    cd_md=.5*F.linear(corr['dh_m'],corr['W_md']);cd_dd=.5*F.linear(corr['dh_d'],corr['W_dd'])
    pr=wr@zr+(wr*A[:,d].T[:,None,:]).sum(2,keepdim=True)*cm_dm[:,None,:]
    pr=pr+(wr*Sr1[:,r].T[:,None,:]).sum(2,keepdim=True)*cm_mm[:,None,:]
    pd=wd@zd+(wd*A[r][:,None,:]).sum(2,keepdim=True)*cd_md[:,None,:]
    pd=pd+(wd*Sd1[:,d].T[:,None,:]).sum(2,keepdim=True)*cd_dd[:,None,:]
    return pr,pd

class OCNBranch(nn.Module):
    def __init__(self,hidden):
        super().__init__();self.rproj=nn.Linear(hidden,32,bias=False);self.dproj=nn.Linear(hidden,32,bias=False)
        self.head=nn.Sequential(nn.Linear(132,128),nn.LeakyReLU(.1),nn.Dropout(.1),nn.Linear(128,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,pr,pd,summary):
        h=torch.cat([self.rproj(pr).flatten(1),self.dproj(pd).flatten(1),summary],1)
        return self.head(h).view(-1)

class OCNModel(Model):
    def __init__(self,graph,ps,nq,ne,cfg,cache):
        super().__init__(graph,ps,nq,ne,cfg);self.cache=cache;self.ocn=OCNBranch(cfg.hidden)
    def components(self,eli,ps,qis,ease,dense):
        r,d=eli;xr,xd,Sr1,Sd1,A=dense
        zr,zd,zm,zdd,corr=self.encoder.forward_dense(xr,xd,Sr1,Sd1,A,r,d,ps.qvals(r,d))
        z={'miRNA':zr,'drug':zd}
        parts=[zm,zdd]+[lin(e) for lin,e in zip(self.ncn_lin,ps.qncn(r,d,z,corr))]
        parts += [self.pc(qis),self.em(self.eb(ease))]
        h=F.leaky_relu(self.lin1(torch.cat(parts,-1)),.1);h=F.leaky_relu(self.lin2(h),.1)
        base=self.lin3(h).view(-1)
        w,summary=self.cache.select(r,d);pr,pd=corrected_pool(w,r,d,zr,zd,corr,dense)
        return base,self.ocn(pr,pd,summary)
    def forward(self,*args):
        base,extra=self.components(*args);return base+extra

def make_adapter(data,name,fold,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    state=a.model.state_dict();m=OCNModel(a.graph,a.ps,a.qis['train'].shape[1],a.ease['train'].shape[1],a.cfg,Cache(name,fold,device)).to(device)
    missing,unexpected=m.load_state_dict(state,strict=False)
    assert not unexpected and all(k.startswith('ocn.') for k in missing)
    a.model=m;return a
