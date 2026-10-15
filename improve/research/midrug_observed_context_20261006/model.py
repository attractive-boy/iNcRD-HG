"""Observed-label context in M123, with exact deletion of positive AND zero queries.

No fixed biological sign or repulsive force is imposed on sampled zero labels.
The original positive encoder and M2/M3 definitions are preserved; extra linear
messages use their own parameters, initialized at zero, before each layer sum.
"""
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,Reconstruction,make_graph
from incrd_hg_m123.encoder import QueryHollowEncoder
from incrd_hg_m123.pair_channels import PairStruct

CONFIGS={f'{mode}{suffix}':dict(support=support,alpha=alpha,reconstruction_weight=weight)
    for mode,support,alpha in [('positive_mean',1,1.),('zero_mean',0,1.),('positive_sqrt',1,.5),('zero_sqrt',0,.5)]
    for suffix,weight in [('',0.),('_rec10',10.)]}


class ObservedEncoder(QueryHollowEncoder):
    def configure(self,data,support,alpha):
        self.alpha=alpha;self.support_label=support
        xr,xd=self.graph['miRNA'].x,self.graph['drug'].x
        selected=data['train_pairs'][data['train_y']==support]
        ids=torch.as_tensor(selected,device=xr.device,dtype=torch.long)
        self.context=xr.new_zeros((len(xr),len(xd)))
        self.context.index_put_((ids[:,0],ids[:,1]),torch.ones(len(ids),device=xr.device,dtype=xr.dtype),accumulate=True)
        assert self.context.max()<=1,'Unique observed support required'
        # Geometry is reconstructed from each fit's training records. It is
        # deliberately not a checkpoint buffer, just like original adjacency.
        self.matrices={'miRNA':self.context,'drug':self.context.t()}
        self.degrees={n:c.sum(1) for n,c in self.matrices.items()}
        self.scales={n:self.scale(d) for n,d in self.degrees.items()}
        self.extra1=nn.ModuleDict();self.extra2=nn.ModuleDict()
        for node,edge in [('miRNA','drug__rev_MiDrug__miRNA'),('drug','miRNA__MiDrug__drug')]:
            for number,convs in [(1,self.original.conv1),(2,self.original.conv2)]:
                c=convs[edge].lin_msg
                layer=nn.Linear(c.weight.shape[1],c.weight.shape[0],bias=False).to(c.weight)
                nn.init.zeros_(layer.weight);getattr(self,'extra'+str(number))[node]=layer

    def scale(self,degree):return degree.clamp_min(1).pow(-self.alpha)

    def layer(self,convs,x):
        out=super().layer(convs,x)
        modules=self.extra1 if convs is self.original.conv1 else self.extra2
        for receiver,source in [('miRNA','drug'),('drug','miRNA')]:
            out[receiver]=out[receiver]+self.scales[receiver][:,None]*(self.matrices[receiver]@modules[receiver](x[source]))
        return out

    def forward(self,x,edges):
        # Graph and observed support are fixed at construction. Independent
        # deletion checks construct a separate adapter with the record removed.
        h={n:F.leaky_relu(v,.1) for n,v in self.layer(self.original.conv1,x).items()}
        return self.layer(self.original.conv2,h)

    def forward_dense(self,xr,xd,Sr1,Sd1,A,r=None,d=None,q=None):
        c1,c2=self.original.conv1,self.original.conv2
        p1=self.layer(c1,self.graph.x_dict);h1={n:F.leaky_relu(v,.1) for n,v in p1.items()};z=self.layer(c2,h1)
        if r is None:return z['miRNA'],z['drug']
        qp=q[:,None];qc=self.context[r,d];endpoint={'miRNA':r,'drug':d}
        def change(convs,extra,x,node,source,edge):
            dst,src=endpoint[node],endpoint[source]
            direct=-qp*convs[edge].lin_msg(x[source])[src]/self.counts[node]
            values=extra[node](x[source]);total=self.matrices[node]@values
            old=self.scales[node][dst,None];new=self.scale(self.degrees[node][dst]-qc)[:,None]
            return direct+new*(total[dst]-qc[:,None]*values[src])-old*total[dst]
        routes=[('miRNA','drug','drug__rev_MiDrug__miRNA'),('drug','miRNA','miRNA__MiDrug__drug')]
        dh={n:F.leaky_relu(p1[n][endpoint[n]]+change(c1,self.extra1,self.graph.x_dict,n,s,e),.1)-h1[n][endpoint[n]] for n,s,e in routes}
        result={}
        for node,source,edge in routes:
            index=endpoint[node];same='miRNA__MiMi__miRNA' if node=='miRNA' else 'drug__DrugDrug__drug'
            mat=Sr1 if node=='miRNA' else Sd1
            root=mat.diagonal()[index,None]*F.linear(dh[node],c2[same].lin_msg.weight)
            for e in self.graph.edge_types:
                if e[2]==node:root=root+F.linear(dh[node],c2['__'.join(e)].lin_self.weight)
            # Both query-specific cross edges are absent after deletion, so
            # no changed-source term remains at the opposite query endpoint.
            result[node]=z[node][index]+root/self.counts[node]+change(c2,self.extra2,h1,node,source,edge)
        corr=dict(dh_m=dh['miRNA'],dh_d=dh['drug'],context=self.context,context_scales=self.scales,
            U_dm=self.extra2['miRNA'].weight,U_md=self.extra2['drug'].weight)
        for short,node,edge in [('dm','miRNA','drug__rev_MiDrug__miRNA'),('mm','miRNA','miRNA__MiMi__miRNA'),
                                ('md','drug','miRNA__MiDrug__drug'),('dd','drug','drug__DrugDrug__drug')]:
            corr['W_'+short]=c2[edge].lin_msg.weight*(2/self.counts[node])
        return z['miRNA'],z['drug'],result['miRNA'],result['drug'],corr


class ObservedPairStruct(PairStruct):
    def qncn(self,r,d,z,corr=None):
        if corr is None:return super().qncn(r,d,z,None)
        q=self.qvals(r,d)[:,None];C=corr['context'];scale=corr['context_scales']
        cm_dm=.5*F.linear(corr['dh_d'],corr['W_dm']);cm_mm=.5*F.linear(corr['dh_m'],corr['W_mm'])
        cd_md=.5*F.linear(corr['dh_m'],corr['W_md']);cd_dd=.5*F.linear(corr['dh_d'],corr['W_dd'])
        cm_extra=F.linear(corr['dh_d'],corr['U_dm']);cd_extra=F.linear(corr['dh_m'],corr['U_md'])
        def pool_m(w):
            den=w.sum(1,keepdim=True)+1;v=w@z['miRNA']/den
            v=v+cm_dm*((w*self.A[:,d].t()).sum(1,keepdim=True)/den)
            v=v+cm_mm*((w*self.Sr[:,r].t()).sum(1,keepdim=True)/den)
            return v+cm_extra*((w*C[:,d].t()*scale['miRNA'][None]).sum(1,keepdim=True)/den)
        def pool_d(w):
            den=w.sum(1,keepdim=True)+1;v=w@z['drug']/den
            v=v+cd_md*((w*self.A[r]).sum(1,keepdim=True)/den)
            v=v+cd_dd*((w*self.Sd[:,d].t()).sum(1,keepdim=True)/den)
            return v+cd_extra*((w*C[r]*scale['drug'][None]).sum(1,keepdim=True)/den)
        Ad=self.A[:,d].t();Ar=self.A[r];wr=Ad*(self.Cm[r]-q*Ad);wd=Ar*(self.Cd[:,d].t()-q*Ar)
        ix=torch.arange(len(r),device=r.device);wr[ix,r]=0;wd[ix,d]=0
        # Every M2 intermediate excludes both query endpoints; hence neither
        # its root projection nor its context degree changes under deletion.
        return [pool_m(wr),pool_d(wd),pool_m(Ad*self.Sr[r]),pool_d(Ar*self.Sd[:,d].t())]


def configure(a,data,cfg):
    assert a.ps.A.max()<=1
    a.model.encoder.__class__=ObservedEncoder
    a.model.encoder.configure(data,cfg['support'],cfg['alpha']);a.ps.__class__=ObservedPairStruct
    if cfg['support']==0:assert not bool((a.model.encoder.context*a.ps.A).any())
    return a


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    cfg=dict(reconstruction_weight=10.) if name=='unmasked10' else CONFIGS[name]
    if cfg['reconstruction_weight']:a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a if name=='unmasked10' else configure(a,data,cfg)


def mask_generator(device,fold):return torch.Generator(device=device).manual_seed(20261006+fold)
