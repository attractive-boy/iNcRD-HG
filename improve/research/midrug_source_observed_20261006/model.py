"""Two observation relations in exact three-layer drug-source propagation.

The additional relation holds either sampled-zero or duplicate-positive records.
Both query records are deleted; zero observations have no biological sign meaning.
"""
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction,base,read_module,HERE

prior=read_module('positive_source_propagation',HERE.parent/'midrug_source_conditioned_20261006/model.py')
propagate,neighbor_list=prior.propagate,prior.neighbor_list
CONFIGS={n+s:dict(observation=n,hidden=32,reconstruction_weight=w,branch_lr=.003)
         for n in ['zero','positive'] for s,w in [('',0.),('_rec10',10.)]}

class ObservedSourcePropagation(prior.SourcePropagation):
    def add_observation(self):
        self.cdm,self.cmd=[nn.ModuleList([nn.Linear(self.hidden,self.hidden,bias=False) for _ in range(3)]) for _ in range(2)]
        for p in list(self.cdm.parameters())+list(self.cmd.parameters()):nn.init.zeros_(p)

    def states(self,A,C,Sr,Sd):
        nr,nd=A.shape
        marker=torch.eye(nd,device=A.device,dtype=A.dtype)
        hm=A.new_zeros(nd,nr,self.hidden);hd=marker[:,:,None]*self.q
        saved=[]
        for layer in range(3):
            pm=propagate(A,self.dm[layer](hd))+propagate(C,self.cdm[layer](hd))+propagate(Sr,self.mm[layer](hm))+self.sm[layer](hm)
            pd=propagate(A.T,self.md[layer](hm))+propagate(C.T,self.cmd[layer](hm))+propagate(Sd,self.dd[layer](hd))+self.sd[layer](hd)+marker[:,:,None]*self.boundary[layer]
            saved.append((hm,hd,pm,pd))
            hm,hd=self.act(self.nm[layer],pm),self.act(self.nd[layer],pd)
        return hm,hd,saved

    def forward(self,A,C,Sr,Sd,nbr,r,d):
        hm,hd,st=self.states(A,C,Sr,Sd)
        out=torch.cat([st[2][0][d,r],hm[d,r]],1)
        qa,qc=A[r,d],C[r,d];ix=torch.nonzero((qa!=0)|(qc!=0),as_tuple=False).flatten()
        if not len(ix):return out
        rr,dd=r[ix],d[ix];qa,qc=qa[ix,None],qc[ix,None];batch=torch.arange(len(rr),device=A.device)
        (h0m,h0d,p1m,p1d),(h1m,h1d,p2m,p2d),(h2m,h2d,p3m,p3d)=st
        # Both initial miRNA states and Sr's diagonal are zero. Only r changes at layer1.
        p1r=p1m[dd,rr]-qa*self.dm[0](h0d[dd,dd])-qc*self.cdm[0](h0d[dd,dd])
        h1r=self.act(self.nm[0],p1r);delta1=h1r-h1m[dd,rr]
        p2r=p2m[dd,rr]-qa*self.dm[1](h1d[dd,dd])-qc*self.cdm[1](h1d[dd,dd])+F.linear(delta1,self.sm[1].weight)
        h2r=self.act(self.nm[1],p2r);delta2r=h2r-h2m[dd,rr]
        # Drug d is excluded from both final relations. Its layer2 correction is unused.
        p2drugs=p2d[dd]+A[rr,:,None]*F.linear(delta1,self.md[1].weight)[:,None,:]+C[rr,:,None]*F.linear(delta1,self.cmd[1].weight)[:,None,:]
        wa,wc=A[rr].clone(),C[rr].clone();wa[batch,dd]=0;wc[batch,dd]=0
        delta2d=(self.act(self.nd[1],p2drugs)-h2d[dd])*((wa!=0)|(wc!=0))[:,:,None]
        neighbors=nbr[rr];valid=(neighbors>=0).to(A.dtype);nn_=neighbors.clamp_min(0);rgrid=rr[:,None].expand_as(nn_)
        p2neighbors=p2m[dd[:,None],nn_]+(Sr[nn_,rgrid]*valid)[:,:,None]*F.linear(delta1,self.mm[1].weight)[:,None,:]
        delta2m=self.act(self.nm[1],p2neighbors)-h2m[dd[:,None],nn_]
        p3r=p3m[dd,rr]-qa*self.dm[2](h2d[dd,dd])-qc*self.cdm[2](h2d[dd,dd])
        p3r=p3r+F.linear((wa[:,:,None]*delta2d).sum(1),self.dm[2].weight)+F.linear((wc[:,:,None]*delta2d).sum(1),self.cdm[2].weight)
        p3r=p3r+F.linear(((Sr[rgrid,nn_]*valid)[:,:,None]*delta2m).sum(1),self.mm[2].weight)+F.linear(delta2r,self.sm[2].weight)
        return out.index_copy(0,ix,torch.cat([h2r,self.act(self.nm[2],p3r)],1))

class SourceModel(base.Model):
    def forward(self,eli,ps,qis,ease,dense):
        r,d=eli
        z=self.source_path(ps.A,self.source_c,self.source_sr,self.source_sd,self.source_nbr,r,d)
        addition=self.source_projection(z)
        handle=self.lin1.register_forward_hook(lambda module,args,output:output+addition)
        try:return super().forward(eli,ps,qis,ease,dense)
        finally:handle.remove()

def make_adapter(data,name,device):
    if name not in CONFIGS:return prior.make_adapter(data,name,device)
    cfg=CONFIGS[name]
    # Construct the complete old source model first, preserving its exact RNG/init.
    a=prior.make_adapter(data,'source_rec10' if cfg['reconstruction_weight'] else 'source',device)
    m=a.model;m.__class__=SourceModel;m.source_path.__class__=ObservedSourcePropagation
    m.source_path.add_observation();m.source_path.to(device)
    labels=data['train_y'];pairs=data['train_pairs'][labels==(0 if cfg['observation']=='zero' else 1)]
    c=torch.zeros_like(a.ps.A);ij=torch.as_tensor(pairs,device=device)
    c.index_put_((ij[:,0],ij[:,1]),torch.ones(len(ij),device=device),accumulate=True)
    # Derived from the current support, never overwritten by checkpoint state.
    m.source_c=c
    return a

mask_generator,make_optimizer=prior.mask_generator,prior.make_optimizer
