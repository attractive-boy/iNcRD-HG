"""Attribute-only keys and query-excluded nonlinear label-value retrieval."""
import types
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,make_graph,Reconstruction

CONFIGS={'label':dict(retrieve=True,labels=True),'features':dict(retrieve=True,labels=False),'direct':dict(retrieve=False,labels=False)}

class PairMemory(nn.Module):
    def __init__(self,graph,data,config,width=64,neighbors=32):
        super().__init__();self.config=config;self.neighbors=neighbors;self.nd=graph['drug'].num_nodes
        self.enc=nn.ModuleDict()
        for n in ['miRNA','drug']:
            x=graph[n].x.detach();z=(x-x.mean(0))/x.std(0,unbiased=False).clamp(min=.05)
            self.register_buffer('x_'+n,z,persistent=False)
            self.enc[n]=nn.Sequential(nn.Linear(x.shape[1],width),nn.ReLU(),nn.Linear(width,width))
        self.K=nn.Linear(width,width)
        self.T=nn.Sequential(nn.Linear(width,2*width),nn.ReLU(),nn.Linear(2*width,width,bias=False))
        self.label=nn.Embedding(2,width);nn.init.uniform_(self.label.weight,-1.,1.)
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        pairs=torch.as_tensor(data['train_pairs'],device=x.device,dtype=torch.long)
        y=torch.as_tensor(data['train_y'],device=x.device,dtype=torch.long)
        ids=pairs[:,0]*self.nd+pairs[:,1];assert ids.unique().numel()==len(ids)
        self.register_buffer('support_pairs',pairs,persistent=False)
        self.register_buffer('support_y',y,persistent=False)
        self.register_buffer('support_ids',ids,persistent=False)

    def nodes(self):
        return {n:self.enc[n](getattr(self,'x_'+n)) for n in ['miRNA','drug']}

    def context(self,pairs,encoded=None):
        encoded=self.nodes() if encoded is None else encoded
        # Compute node terms once: the same pair receives the same key in any query batch.
        keys={n:F.linear(h,self.K.weight,None) for n,h in encoded.items()}
        qh=encoded['miRNA'][pairs[0]]+encoded['drug'][pairs[1]]
        qk=keys['miRNA'][pairs[0]]+keys['drug'][pairs[1]]+self.K.bias
        if not self.config['retrieve']:
            return qh,torch.zeros_like(qh),None,None
        sp=self.support_pairs;sk=keys['miRNA'][sp[:,0]]+keys['drug'][sp[:,1]]+self.K.bias
        qids=pairs[0]*self.nd+pairs[1]
        # Discrete selection is detached; recomputed selected distances carry gradients.
        with torch.no_grad():
            q,s=qk.double(),sk.double()
            distance=q.square().sum(1,keepdim=True)+s.square().sum(1)[None,:]-2*q@s.T
            own=qids[:,None]==self.support_ids[None,:];distance.masked_fill_(own,float('inf'))
            count=min(self.neighbors,len(sp)-1)
            assert count>0
            # Stable sort resolves identical distances by frozen support order.
            ix=torch.argsort(distance,dim=1,stable=True)[:,:count]
            assert not own.gather(1,ix).any()
        difference=qk[:,None]-sk[ix]
        weights=torch.softmax(-difference.square().sum(-1),dim=1)
        values=self.T(difference)
        if self.config['labels']:values=values+self.label(self.support_y[ix])
        pooled=(weights[...,None]*values).sum(1)
        return qh,pooled,ix,weights

    def forward(self,pairs):
        qh,pooled,_,_=self.context(pairs)
        return self.head(qh+pooled).squeeze(-1)

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name in CONFIGS:
        m=a.model;m.add_module('case_memory',PairMemory(a.graph,data,CONFIGS[name]).to(device))
        original=m.forward
        def forward(self,pairs,*args,**kwargs):return original(pairs,*args,**kwargs)+self.case_memory(pairs)
        m.forward=types.MethodType(forward,m)
    return a

def make_optimizer(model):
    main=[p for n,p in model.named_parameters() if not n.startswith(('reconstruction.','case_memory.'))]
    aux=[p for n,p in model.named_parameters() if n.startswith(('reconstruction.','case_memory.'))]
    return torch.optim.RMSprop([dict(params=main,lr=1e-4),dict(params=aux,lr=1e-4,weight_decay=1e-4)],alpha=.99)

def mask_generator(device,seed):
    g=torch.Generator(device=device);g.manual_seed(seed);return g
