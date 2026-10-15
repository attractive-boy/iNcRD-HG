"""Label-free typed global feature routing BEFORE the exact M123 graph encoder.

Every arm allocates the same modules. Disabled paths are excluded from the
optimizer; allocated parameter equality is not an effective-capacity claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,Reconstruction,make_graph
from incrd_hg_m123.encoder import QueryHollowEncoder

CONFIGS={f'{mode}{suffix}':dict(mode=mode,reconstruction_weight=weight)
         for mode in ('local','pool','route') for suffix,weight in [('',0.),('_rec10',10.)]}


class GlobalFeatures(nn.Module):
    def __init__(self,graph,mode,width=64,routes=8):
        super().__init__();self.mode,self.width,self.routes=mode,width,routes
        self.input=nn.ModuleDict();self.key=nn.ModuleDict();self.value=nn.ModuleDict()
        self.query=nn.ModuleDict();self.output=nn.ModuleDict();self.tokens=nn.ParameterDict()
        for name in ('miRNA','drug'):
            x=graph[name].x
            self.register_buffer(name+'_mean',x.mean(0))
            self.register_buffer(name+'_scale',x.std(0).clamp(min=.05))
            self.input[name]=nn.Sequential(nn.Linear(x.shape[1],width),nn.LeakyReLU(.1),nn.LayerNorm(width))
            self.key[name]=nn.Linear(width,width,bias=False)
            self.value[name]=nn.Linear(width,width,bias=False)
            self.query[name]=nn.Linear(width,width,bias=False)
            self.tokens[name]=nn.Parameter(torch.randn(routes,width)/math.sqrt(width))
            self.output[name]=nn.Sequential(nn.Linear(2*width,width),nn.LeakyReLU(.1),nn.Linear(width,x.shape[1]))
            nn.init.zeros_(self.output[name][-1].weight);nn.init.zeros_(self.output[name][-1].bias)
        if mode!='route':
            self.key.requires_grad_(False);self.query.requires_grad_(False);self.tokens.requires_grad_(False)
        if mode=='local':self.value.requires_grad_(False)

    def forward(self,x):
        h={name:self.input[name]((v-getattr(self,name+'_mean'))/getattr(self,name+'_scale')) for name,v in x.items()}
        if self.mode!='local':
            tokens=[]
            for name in ('miRNA','drug'):
                value=self.value[name](h[name])
                if self.mode=='route':
                    weights=torch.softmax(self.tokens[name]@self.key[name](h[name]).t()/math.sqrt(self.width),dim=1)
                    tokens.append(weights@value)
                else:tokens.append(value.mean(0,keepdim=True).expand(self.routes,-1))
            context=torch.cat(tokens)
        result={}
        for name,v in h.items():
            if self.mode=='local':c=torch.zeros_like(v)
            elif self.mode=='pool':c=context.mean(0,keepdim=True).expand_as(v)
            else:c=torch.softmax(self.query[name](v)@context.t()/math.sqrt(self.width),dim=1)@context
            delta=self.output[name](torch.cat([v,c],dim=-1))
            result[name]=x[name]+.1*getattr(self,name+'_scale')*delta
        return result


class GlobalFeatureEncoder(QueryHollowEncoder):
    def layer(self,convs,x):
        if convs is self.original.conv1:x=self.refiner(x)
        return super().layer(convs,x)

    def forward(self,x,edges):
        return self.original(self.refiner(x),edges)

    def forward_dense(self,xr,xd,Sr1,Sd1,A,r=None,d=None,q=None):
        # Parent layer() transforms the full-graph inputs. Its explicit removed
        # messages must use the same transformed endpoint inputs. Recomputing
        # this small deterministic function avoids stale graph/autograd caches.
        x=self.refiner(self.graph.x_dict)
        return super().forward_dense(x['miRNA'],x['drug'],Sr1,Sd1,A,r,d,q)


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='control':return a
    cfg=dict(reconstruction_weight=10.) if name=='unmasked10' else CONFIGS[name]
    if cfg['reconstruction_weight']:a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name=='unmasked10':return a
    a.model.encoder.__class__=GlobalFeatureEncoder
    a.model.encoder.add_module('refiner',GlobalFeatures(a.graph,cfg['mode']).to(device))
    return a


def mask_generator(device,fold):return torch.Generator(device=device).manual_seed(20261006+fold)
