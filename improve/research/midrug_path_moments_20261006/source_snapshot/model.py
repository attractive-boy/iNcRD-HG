"""Joint moments of the TWO intermediate nodes on actual simple 3-edge paths."""
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,Reconstruction,make_graph
from incrd_hg_m123.model import Model

CONFIGS={f'{mode}{suffix}':dict(mode=mode,reconstruction_weight=weight)
    for mode in ('independent','joint','covariance') for suffix,weight in [('',0.),('_rec10',10.)]}


def moments(A,Sr,Sd,r,d,phi,psi,a,b,c,e,chunk=512):
    """Linear projected node states plus exact M1 shifts on intermediate nodes.

    phi_i(q)=phi_i+a_q+Sr[i,r]*b_q; psi_j(q)=psi_j+c_q+Sd[j,d]*e_q.
    Path measure is A[r,j]*A[i,j]*A[i,d], with i!=r and j!=d.
    """
    output_dtype=phi.dtype
    # Covariance subtracts potentially large nearly equal moments. Accumulate
    # in float64; the branch head still receives the encoder's original dtype.
    A,Sr,Sd,phi,psi,a,b,c,e=[v.to(torch.float64) for v in (A,Sr,Sd,phi,psi,a,b,c,e)]
    outputs={k:[] for k in ('independent','joint','covariance')};counts=[]
    for start in range(0,len(r),chunk):
        stop=min(start+chunk,len(r));rr,dd=r[start:stop],d[start:stop];ix=torch.arange(len(rr),device=A.device)
        mr=A[:,dd].t().clone();md=A[rr].clone();mr[ix,rr]=0;md[ix,dd]=0
        hr=phi[None]+a[start:stop,None]+Sr[:,rr].t()[:,:,None]*b[start:stop,None]
        hd=psi[None]+c[start:stop,None]+Sd[:,dd].t()[:,:,None]*e[start:stop,None]
        wr=mr*(md@A.t());wd=md*(mr@A);count=wr.sum(1,keepdim=True);den=count.clamp_min(1)
        mean_r=(wr[:,:,None]*hr).sum(1)/den;mean_d=(wd[:,:,None]*hd).sum(1)/den
        # Avoid a [query, miRNA, drug, rank] tensor: contract the miRNA side first.
        left=(mr[:,:,None]*hr).transpose(1,2)@A
        joint=(left.transpose(1,2)*hd*md[:,:,None]).sum(1)/den
        independent=mean_r*mean_d;reliability=count/(count+1)
        for name,value in [('independent',independent),('joint',joint),('covariance',joint-independent)]:
            outputs[name].append(reliability*value)
        counts.append(count[:,0])
    if not len(r):return {k:phi.new_empty((0,phi.shape[1]),dtype=output_dtype) for k in outputs},phi.new_empty((0,),dtype=output_dtype)
    return {k:torch.cat(v).to(output_dtype) for k,v in outputs.items()},torch.cat(counts).to(output_dtype)


class PathMoments(nn.Module):
    def __init__(self,hidden,mode,rank=16):
        super().__init__();self.mode=mode
        self.rproj=nn.Linear(hidden,rank,bias=False);self.dproj=nn.Linear(hidden,rank,bias=False)
        self.head=nn.Sequential(nn.LayerNorm(rank),nn.Linear(rank,64),nn.LeakyReLU(.1),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def features(self,ps,r,d,zr,zd,corr):
        phi,psi=self.rproj(zr),self.dproj(zd)
        a=self.rproj(.5*F.linear(corr['dh_d'],corr['W_dm']))
        b=self.rproj(.5*F.linear(corr['dh_m'],corr['W_mm']))
        c=self.dproj(.5*F.linear(corr['dh_m'],corr['W_md']))
        e=self.dproj(.5*F.linear(corr['dh_d'],corr['W_dd']))
        return moments(ps.A,ps.Sr,ps.Sd,r,d,phi,psi,a,b,c,e)

    def forward(self,ps,r,d,zr,zd,corr):
        values,count=self.features(ps,r,d,zr,zd,corr)
        # No structural evidence means exactly zero residual, including head bias.
        return self.head(values[self.mode]).view(-1)*(count>0)


class MomentModel(Model):
    def forward(self,eli,ps,qis,ease,dense):
        r,d=eli
        zr,zd,zm,zdd,corr=self.encoder.forward_dense(*dense,r,d,ps.qvals(r,d))
        parts=[zm,zdd]+[lin(x) for lin,x in zip(self.ncn_lin,ps.qncn(r,d,{'miRNA':zr,'drug':zd},corr))]
        parts += [self.pc(qis),self.em(self.eb(ease))]
        h=F.leaky_relu(self.lin1(torch.cat(parts,-1)),.1);h=F.leaky_relu(self.lin2(h),.1)
        return self.lin3(h).view(-1)+self.path_moment(ps,r,d,zr,zd,corr)


def configure(a,cfg):
    assert torch.equal(a.ps.Sr,a.ps.Sr.t()) and torch.equal(a.ps.Sd,a.ps.Sd.t()),'Pinned M123 uses symmetric same-type graphs'
    a.model.__class__=MomentModel
    a.model.add_module('path_moment',PathMoments(a.cfg.hidden,cfg['mode']).to(a.y.device))
    return a


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    cfg=dict(reconstruction_weight=10.) if name=='unmasked10' else CONFIGS[name]
    if cfg['reconstruction_weight']:a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a if name=='unmasked10' else configure(a,cfg)


def mask_generator(device,fold):return torch.Generator(device=device).manual_seed(20261006+fold)
