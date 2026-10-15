"""Receiver-degree message scaling with exact query deletion at both GNN layers."""
import torch
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction
from incrd_hg_m123.encoder import QueryHollowEncoder
from incrd_hg_m123.pair_channels import PairStruct

CONFIGS={
    'target05':dict(alpha=.5,scope='target',reconstruction_weight=0.),
    'target10':dict(alpha=1.,scope='target',reconstruction_weight=0.),
    'all05':dict(alpha=.5,scope='all',reconstruction_weight=0.),
    'all10':dict(alpha=1.,scope='all',reconstruction_weight=0.),
    'target05_rec10':dict(alpha=.5,scope='target',reconstruction_weight=10.),
    'all05_rec10':dict(alpha=.5,scope='all',reconstruction_weight=10.),
}


class DegreeEncoder(QueryHollowEncoder):
    def configure(self,alpha,scope):
        self.alpha,self.scope=alpha,scope
        self.degrees={e:torch.sparse.sum(a,dim=1).to_dense() for e,a in self.adj.items()}
        self.scales={e:self.scale(e,d) for e,d in self.degrees.items()}

    def scale(self,edge,degree):
        exponent=self.alpha if self.scope=='all' or edge[0]!=edge[2] else 0.
        return degree.clamp(min=1).pow(-exponent)

    def layer(self,convs,x):
        out={}
        for e in self.graph.edge_types:
            c=convs['__'.join(e)]
            msg=torch.sparse.mm(self.adj[e],c.lin_msg(x[e[0]]))
            value=self.scales[e][:,None]*msg+c.lin_self(x[e[2]])
            out[e[2]]=out.get(e[2],0)+value
        return {n:v/self.counts[n] for n,v in out.items()}

    def forward(self,x,edges):
        # The graph was frozen at construction; independent deletion checks rebuild it.
        h={n:F.leaky_relu(v,.1) for n,v in self.layer(self.original.conv1,x).items()}
        return self.layer(self.original.conv2,h)

    def forward_dense(self,xr,xd,Sr1,Sd1,A,r=None,d=None,q=None):
        c1,c2=self.original.conv1,self.original.conv2
        mm=('miRNA','MiMi','miRNA');dd=('drug','DrugDrug','drug')
        md=('miRNA','MiDrug','drug');dm=('drug','rev_MiDrug','miRNA')
        nr,nd=self.counts['miRNA'],self.counts['drug']
        p1=self.layer(c1,self.graph.x_dict)
        h1={n:F.leaky_relu(v,.1) for n,v in p1.items()}
        z=self.layer(c2,h1)
        if r is None:return z['miRNA'],z['drug']
        qq=q[:,None]
        def target_change(convs,edge,x,receiver,sender):
            messages=convs['__'.join(edge)].lin_msg(x)
            total=torch.sparse.mm(self.adj[edge],messages)[receiver]
            old=self.scales[edge][receiver,None]
            new=self.scale(edge,self.degrees[edge][receiver]-q)[:,None]
            return (new*(total-qq*messages[sender])-old*total)/self.counts[edge[2]]
        dh_m=F.leaky_relu(p1['miRNA'][r]+target_change(c1,dm,xd,r,d),.1)-h1['miRNA'][r]
        dh_d=F.leaky_relu(p1['drug'][d]+target_change(c1,md,xr,d,r),.1)-h1['drug'][d]
        def self_shift(node,index,delta,edge,mat):
            c=c2['__'.join(edge)]
            shift=(self.scales[edge][index]*mat.diagonal()[index])[:,None]*F.linear(delta,c.lin_msg.weight)
            for e in self.graph.edge_types:
                if e[2]==node:shift=shift+F.linear(delta,c2['__'.join(e)].lin_self.weight)
            return shift/self.counts[node]
        zm=z['miRNA'][r]+self_shift('miRNA',r,dh_m,mm,Sr1)+target_change(c2,dm,h1['drug'],r,d)
        zd=z['drug'][d]+self_shift('drug',d,dh_d,dd,Sd1)+target_change(c2,md,h1['miRNA'],d,r)
        corr=dict(dh_m=dh_m,dh_d=dh_d,
            W_dm=c2['__'.join(dm)].lin_msg.weight*(2/nr),W_mm=c2['__'.join(mm)].lin_msg.weight*(2/nr),
            W_md=c2['__'.join(md)].lin_msg.weight*(2/nd),W_dd=c2['__'.join(dd)].lin_msg.weight*(2/nd),
            scale_dm=self.scales[dm],scale_mm=self.scales[mm],scale_md=self.scales[md],scale_dd=self.scales[dd])
        return z['miRNA'],z['drug'],zm,zd,corr


class DegreePairStruct(PairStruct):
    def qncn(self,r,d,z,corr=None):
        if corr is None:return super().qncn(r,d,z,None)
        q=self.qvals(r,d)[:,None]
        cm_dm=.5*F.linear(corr['dh_d'],corr['W_dm'])
        cm_mm=.5*F.linear(corr['dh_m'],corr['W_mm'])
        cd_md=.5*F.linear(corr['dh_m'],corr['W_md'])
        cd_dd=.5*F.linear(corr['dh_d'],corr['W_dd'])
        def pool_m(w):
            den=w.sum(1,keepdim=True)+1
            e=w@z['miRNA']/den
            e=e+cm_dm*((w*self.A[:,d].t()*corr['scale_dm'][None]).sum(1,keepdim=True)/den)
            return e+cm_mm*((w*self.Sr[:,r].t()*corr['scale_mm'][None]).sum(1,keepdim=True)/den)
        def pool_d(w):
            den=w.sum(1,keepdim=True)+1
            e=w@z['drug']/den
            e=e+cd_md*((w*self.A[r]*corr['scale_md'][None]).sum(1,keepdim=True)/den)
            return e+cd_dd*((w*self.Sd[:,d].t()*corr['scale_dd'][None]).sum(1,keepdim=True)/den)
        Ad=self.A[:,d].t();Ar=self.A[r]
        wr=Ad*(self.Cm[r]-q*Ad);wd=Ar*(self.Cd[:,d].t()-q*Ar)
        ix=torch.arange(len(r),device=r.device);wr[ix,r]=0;wd[ix,d]=0
        return [pool_m(wr),pool_d(wd),pool_m(Ad*self.Sr[r]),pool_d(Ar*self.Sd[:,d].t())]


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    cfg=dict(alpha=0.,scope='all',reconstruction_weight=0.) if name=='identity' else CONFIGS[name]
    assert a.ps.A.max()<=1,'The clean target relation must have unique pairs'
    # Change behavior while retaining every original initialized parameter and cache.
    a.model.encoder.__class__=DegreeEncoder
    a.model.encoder.configure(cfg['alpha'],cfg['scope'])
    a.ps.__class__=DegreePairStruct
    if cfg['reconstruction_weight']:
        a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a


def mask_generator(device,fold):
    return torch.Generator(device=device).manual_seed(20261006+fold)
