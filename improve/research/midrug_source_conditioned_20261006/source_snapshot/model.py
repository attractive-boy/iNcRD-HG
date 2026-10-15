"""Complete-graph drug-source propagation with exact three-layer query deletion.

Derived from the local PLMF NBF class, not imported from its executable loader.
Uses only original miRNA/drug positive adjacency and same-type relations.
"""
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction,base

CONFIGS={n+s:dict(conditioned=n=='source',hidden=32,reconstruction_weight=w,branch_lr=.003)
         for n in ['source','uniform'] for s,w in [('',0.),('_rec10',10.)]}

def propagate(matrix,states):
    """Receiver-by-sender sparse operator, independently for every source context."""
    ns,nnodes,width=states.shape
    result=torch.sparse.mm(matrix.to_sparse(),states.transpose(0,1).reshape(nnodes,ns*width))
    return result.reshape(matrix.shape[0],ns,width).transpose(0,1)

def neighbor_list(matrix):
    rows=[torch.nonzero(row,as_tuple=False).flatten() for row in matrix]
    width=max(1,max(map(len,rows)));out=torch.full((len(rows),width),-1,dtype=torch.long,device=matrix.device)
    for i,row in enumerate(rows):out[i,:len(row)]=row
    return out

class SourcePropagation(nn.Module):
    def __init__(self,hidden,conditioned):
        super().__init__();self.hidden=hidden;self.conditioned=conditioned
        self.dm,self.md,self.mm,self.dd=[nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(3)]) for _ in range(4)]
        self.sm,self.sd=[nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(3)]) for _ in range(2)]
        self.q=nn.Parameter(torch.randn(hidden)*.1);self.boundary=nn.Parameter(torch.randn(3,hidden)*.1)
        self.nm,self.nd=[nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)]) for _ in range(2)]

    @staticmethod
    def act(norm,value):return F.leaky_relu(norm(value),.1)

    def states(self,A,Sr,Sd):
        nr,nd=A.shape
        # Same total boundary L2 norm in both arms; uniform arm has no query-source marker.
        marker=torch.eye(nd,device=A.device,dtype=A.dtype) if self.conditioned else torch.full((nd,nd),nd**-.5,device=A.device,dtype=A.dtype)
        hm=A.new_zeros(nd,nr,self.hidden);hd=marker[:,:,None]*self.q
        saved=[]
        for layer in range(3):
            pm=propagate(A,self.dm[layer](hd))+propagate(Sr,self.mm[layer](hm))+self.sm[layer](hm)
            pd=propagate(A.T,self.md[layer](hm))+propagate(Sd,self.dd[layer](hd))+self.sd[layer](hd)+marker[:,:,None]*self.boundary[layer]
            saved.append((hm,hd,pm,pd))
            hm,hd=self.act(self.nm[layer],pm),self.act(self.nd[layer],pd)
        return hm,hd,saved

    def forward(self,A,Sr,Sd,nbr,r,d):
        """Read the query miRNA after layers2/3; remove all copies A[r,d]."""
        hm,hd,st=self.states(A,Sr,Sd)
        out=torch.cat([st[2][0][d,r],hm[d,r]],1)
        q=A[r,d];indices=torch.nonzero(q!=0,as_tuple=False).flatten()
        if not len(indices):return out
        rr,dd=r[indices],d[indices];qq=q[indices,None];batch=torch.arange(len(rr),device=A.device)
        (h0m,h0d,p1m,p1d),(h1m,h1d,p2m,p2d),(h2m,h2d,p3m,p3d)=st
        # At layer1 only r changes: all initial miRNA states are zero in BOTH arms.
        p1r=p1m[dd,rr]-qq*self.dm[0](h0d[dd,dd])
        h1r=self.act(self.nm[0],p1r);delta1=h1r-h1m[dd,rr]
        # Sr has zero diagonal, so r's layer2 correction is its self map + deleted edge.
        p2r=p2m[dd,rr]-qq*self.dm[1](h1d[dd,dd])+F.linear(delta1,self.sm[1].weight)
        h2r=self.act(self.nm[1],p2r);delta2r=h2r-h2m[dd,rr]
        # Other drugs hear r's delta. The queried drug d is not needed at layer3
        # because the edge d->r is deleted; explicitly zero its final weight.
        p2drugs=p2d[dd]+A[rr,:,None]*F.linear(delta1,self.md[1].weight)[:,None,:]
        w3=A[rr].clone();w3[batch,dd]=0
        delta2d=(self.act(self.nd[1],p2drugs)-h2d[dd])*(w3!=0)[:,:,None]
        # Only incoming same-type neighbors of r can affect its final readout.
        neighbors=nbr[rr];valid=(neighbors>=0).to(A.dtype);nn_=neighbors.clamp_min(0);rgrid=rr[:,None].expand_as(nn_)
        incoming_delta=Sr[nn_,rgrid]*valid
        p2neighbors=p2m[dd[:,None],nn_]+incoming_delta[:,:,None]*F.linear(delta1,self.mm[1].weight)[:,None,:]
        delta2m=self.act(self.nm[1],p2neighbors)-h2m[dd[:,None],nn_]
        outgoing=Sr[rgrid,nn_]*valid
        p3r=p3m[dd,rr]-qq*self.dm[2](h2d[dd,dd])
        p3r=p3r+F.linear((w3[:,:,None]*delta2d).sum(1),self.dm[2].weight)
        p3r=p3r+F.linear((outgoing[:,:,None]*delta2m).sum(1),self.mm[2].weight)+F.linear(delta2r,self.sm[2].weight)
        replacement=torch.cat([h2r,self.act(self.nm[2],p3r)],1)
        return out.index_copy(0,indices,replacement)

class SourceModel(base.Model):
    def forward(self,eli,ps,qis,ease,dense):
        r,d=eli
        z=self.source_path(ps.A,self.source_sr,self.source_sd,self.source_nbr,r,d)
        addition=self.source_projection(z)
        # Algebraically equivalent to appending z to lin1's input. Keeps all original
        # parameter/state keys and the exact original forward implementation intact.
        handle=self.lin1.register_forward_hook(lambda module,args,output:output+addition)
        try:return super().forward(eli,ps,qis,ease,dense)
        finally:handle.remove()

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    if name=='unmasked10' or CONFIGS[name]['reconstruction_weight']:
        a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name=='unmasked10':return a
    cfg=CONFIGS[name];m=a.model;m.__class__=SourceModel
    m.source_path=SourcePropagation(cfg['hidden'],cfg['conditioned']).to(device)
    m.source_projection=nn.Linear(2*cfg['hidden'],m.lin1.out_features,bias=False).to(device);nn.init.zeros_(m.source_projection.weight)
    # PairStruct stores sender-by-receiver same-type operators; use graph direction.
    m.source_sr=a.ps.Sr.T;m.source_sd=a.ps.Sd.T;m.source_nbr=neighbor_list(m.source_sr)
    assert not torch.diagonal(m.source_sr).any() and not torch.diagonal(m.source_sd).any()
    return a

def mask_generator(device,seed):
    g=torch.Generator(device=device);g.manual_seed(seed);return g

def make_optimizer(m,cfg):
    groups=[dict(params=[p for n,p in m.named_parameters() if not n.startswith(('source_path.','source_projection.','reconstruction.'))],lr=1e-4),
            dict(params=list(m.source_path.parameters())+list(m.source_projection.parameters()),lr=cfg['branch_lr'],weight_decay=1e-4)]
    if cfg['reconstruction_weight']:groups.append(dict(params=list(m.reconstruction.parameters()),lr=1e-4,weight_decay=1e-4))
    return torch.optim.RMSprop(groups,alpha=.99)
