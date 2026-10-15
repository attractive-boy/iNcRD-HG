"""Drug-conditioned expression features before complete two-layer propagation.

The original M123 remains intact. A matched mean-drug condition has identical
parameters, initialization and computation. This is a FiLM transfer test.
"""
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction,base

CONFIGS={n+s:dict(conditioned=n=='conditioned',hidden=32,reconstruction_weight=w,branch_lr=3e-4)
         for n in ['conditioned','mean'] for s,w in [('',0.),('_rec10',10.)]}

def propagate(matrix,states):
    contexts,nodes,width=states.shape
    out=torch.sparse.mm(matrix.to_sparse(),states.transpose(0,1).reshape(nodes,contexts*width))
    return out.reshape(matrix.shape[0],contexts,width).transpose(0,1)

def receiver_matrix(graph,edge):
    idx=graph[edge].edge_index
    m=graph[edge[0]].x.new_zeros(graph[edge[2]].num_nodes,graph[edge[0]].num_nodes)
    m.index_put_((idx[1],idx[0]),torch.ones(idx.shape[1],device=idx.device),accumulate=True)
    return m

class ConditionedExpression(nn.Module):
    def __init__(self,nr_features,nd_features,hidden,conditioned):
        super().__init__();self.hidden=hidden;self.conditioned=conditioned
        self.rinput=nn.Sequential(nn.LayerNorm(nr_features),nn.Linear(nr_features,hidden),nn.LeakyReLU(.1))
        self.dinput=nn.Sequential(nn.LayerNorm(nd_features),nn.Linear(nd_features,hidden),nn.LeakyReLU(.1))
        self.modulator=nn.Sequential(nn.Linear(hidden,hidden),nn.LeakyReLU(.1),nn.Linear(hidden,2*hidden))
        nn.init.zeros_(self.modulator[-1].weight);nn.init.zeros_(self.modulator[-1].bias)
        self.dm,self.md,self.mm,self.dd=[nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(2)]) for _ in range(4)]
        self.sm,self.sd=[nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(2)]) for _ in range(2)]
        self.nm,self.nd=[nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)]) for _ in range(2)]

    @staticmethod
    def act(norm,x):return F.leaky_relu(norm(x),.1)

    def initial(self,xr,xd):
        hr,hd=self.rinput(xr),self.dinput(xd)
        condition=hd if self.conditioned else hd.mean(0,keepdim=True).expand_as(hd)
        gamma,beta=self.modulator(condition).chunk(2,dim=1)
        hm=F.leaky_relu((1+gamma.tanh())[:,None,:]*hr[None,:,:]+beta[:,None,:],.1)
        return hm,hd[None,:,:].expand(len(xd),-1,-1)

    def states(self,xr,xd,A,Sr,Sd):
        hm,hd=self.initial(xr,xd);saved=[]
        for layer in range(2):
            pm=.5*(propagate(A,self.dm[layer](hd))+propagate(Sr,self.mm[layer](hm)))+self.sm[layer](hm)
            pd=.5*(propagate(A.T,self.md[layer](hm))+propagate(Sd,self.dd[layer](hd)))+self.sd[layer](hd)
            saved.append((hm,hd,pm,pd))
            hm,hd=self.act(self.nm[layer],pm),self.act(self.nd[layer],pd)
        return hm,hd,saved

    def forward(self,xr,xd,A,Sr,Sd,r,d):
        hm,hd,states=self.states(xr,xd,A,Sr,Sd)
        (h0m,h0d,p1m,p1d),(h1m,h1drugs,p2m,p2drugs)=states
        q=A[r,d,None]
        h1r=self.act(self.nm[0],p1m[d,r]-.5*q*self.dm[0](h0d[d,d]))
        h1d=self.act(self.nd[0],p1d[d,d]-.5*q*self.md[0](h0m[d,r]))
        dr,dd=h1r-h1m[d,r],h1d-h1drugs[d,d]
        # At layer1 only the two endpoints change. At layer2 their connecting
        # edge is absent; only self and same-type diagonal messages carry deltas.
        p2r=p2m[d,r]-.5*q*self.dm[1](h1drugs[d,d])
        p2r=p2r+F.linear(dr,self.sm[1].weight)+.5*Sr[r,r,None]*F.linear(dr,self.mm[1].weight)
        p2d=p2drugs[d,d]-.5*q*self.md[1](h1m[d,r])
        p2d=p2d+F.linear(dd,self.sd[1].weight)+.5*Sd[d,d,None]*F.linear(dd,self.dd[1].weight)
        return torch.cat([self.act(self.nm[1],p2r),self.act(self.nd[1],p2d)],1)

class ConditionedModel(base.Model):
    def forward(self,eli,ps,qis,ease,dense):
        r,d=eli
        z=self.expression_context(dense[0],dense[1],ps.A,self.context_sr,self.context_sd,r,d)
        addition=self.context_projection(z)
        handle=self.lin1.register_forward_hook(lambda module,args,out:out+addition)
        try:return super().forward(eli,ps,qis,ease,dense)
        finally:handle.remove()

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    if name=='unmasked10' or CONFIGS[name]['reconstruction_weight']:
        a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name=='unmasked10':return a
    cfg=CONFIGS[name];m=a.model;m.__class__=ConditionedModel
    m.expression_context=ConditionedExpression(a.graph['miRNA'].x.shape[1],a.graph['drug'].x.shape[1],cfg['hidden'],cfg['conditioned']).to(device)
    m.context_projection=nn.Linear(2*cfg['hidden'],m.lin1.out_features,bias=False).to(device);nn.init.zeros_(m.context_projection.weight)
    m.context_sr=receiver_matrix(a.graph,('miRNA','MiMi','miRNA'))
    m.context_sd=receiver_matrix(a.graph,('drug','DrugDrug','drug'))
    return a

def mask_generator(device,seed):return torch.Generator(device=device).manual_seed(seed)

def make_optimizer(m,cfg):
    groups=[dict(params=[p for n,p in m.named_parameters() if not n.startswith(('expression_context.','context_projection.','reconstruction.'))],lr=1e-4),
            dict(params=list(m.expression_context.parameters())+list(m.context_projection.parameters()),lr=cfg['branch_lr'],weight_decay=1e-4)]
    if cfg['reconstruction_weight']:groups.append(dict(params=m.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4))
    return torch.optim.RMSprop(groups,alpha=.99)
