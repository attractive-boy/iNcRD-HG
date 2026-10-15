"""Auxiliary feature reconstruction shares the M123 encoder, not its inference head."""
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter,make_graph

CONFIGS={
    'masked001':dict(mask=True,weight=.01,ratio=.5),
    'masked01':dict(mask=True,weight=.1,ratio=.5),
    'masked1':dict(mask=True,weight=1.,ratio=.5),
    'unmasked01':dict(mask=False,weight=.1,ratio=.5),
}

class Reconstruction(nn.Module):
    def __init__(self,graph,hidden):
        super().__init__();self.decoder=nn.ModuleDict();self.token=nn.ParameterDict()
        for name in ('miRNA','drug'):
            x=graph[name].x;mean=x.mean(0);scale=x.std(0).clamp(min=.05)
            self.register_buffer(name+'_mean',mean);self.register_buffer(name+'_scale',scale)
            self.token[name]=nn.Parameter(mean.clone())
            self.decoder[name]=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,128),nn.LeakyReLU(.1),nn.Linear(128,x.shape[1]))

    def masks(self,graph,ratio,generator):
        return {name:torch.randperm(graph[name].num_nodes,device=graph[name].x.device,generator=generator)[:max(1,round(ratio*graph[name].num_nodes))]
                for name in ('miRNA','drug')}

    def loss(self,encoder,graph,indices,masked):
        x={name:graph[name].x for name in ('miRNA','drug')}
        inputs={name:value.clone() for name,value in x.items()}
        if masked:
            for name,ix in indices.items():inputs[name][ix]=self.token[name]
        # No graph/input mutation: auxiliary masking cannot change association inference.
        h=encoder.layer(encoder.original.conv1,inputs)
        h={name:F.leaky_relu(value,.1) for name,value in h.items()}
        h=encoder.layer(encoder.original.conv2,h)
        losses={}
        for name,ix in indices.items():
            target=(x[name][ix]-getattr(self,name+'_mean'))/getattr(self,name+'_scale')
            losses[name]=F.mse_loss(self.decoder[name](h[name][ix]),target)
        return .5*(losses['miRNA']+losses['drug']),losses

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a

def mask_generator(device,seed):
    g=torch.Generator(device=device);g.manual_seed(seed);return g

