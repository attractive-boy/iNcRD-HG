"""Original shared name fragments enter the intact M123 encoder; zero initial change."""
import json
import numpy as np
import torch
from torch import nn
from common import HERE, Adapter, Reconstruction, make_graph, sha256
from incrd_hg_m123.encoder import QueryHollowEncoder

CONFIGS={name:dict(name_features=name,reconstruction_weight=10.) for name in ['aligned','permuted','zero']}

class NameFeatures(nn.Module):
    def __init__(self,features,output_width):
        super().__init__();self.register_buffer('features',features)
        self.projection=nn.Linear(features.shape[1],output_width,bias=False)
        nn.init.zeros_(self.projection.weight)
    def forward(self,x):
        return dict(miRNA=x['miRNA'],drug=x['drug']+self.projection(self.features))

class NameEncoder(QueryHollowEncoder):
    def layer(self,convs,x):
        if convs is self.original.conv1:x=self.refiner(x)
        return super().layer(convs,x)
    def forward(self,x,edges):return self.original(self.refiner(x),edges)
    def forward_dense(self,xr,xd,Sr1,Sd1,A,r=None,d=None,q=None):
        x=self.refiner(self.graph.x_dict)
        return super().forward_dense(x['miRNA'],x['drug'],Sr1,Sd1,A,r,d,q)

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    if name=='baseline':return a
    assert name in CONFIGS
    meta=json.loads((HERE/'input_manifest.json').read_text())
    assert sha256(HERE/'covariates.npz')==meta['covariates_sha256']
    with np.load(HERE/'covariates.npz') as z:
        values=z['features'].copy()
        if name=='permuted':values=values[z['permutation']]
        if name=='zero':values[:]=0
    a.model.encoder.__class__=NameEncoder
    a.model.encoder.add_module('refiner',NameFeatures(torch.from_numpy(values),a.graph['drug'].x.shape[1]).to(device))
    return a

def mask_generator(device,fold):return torch.Generator(device=device).manual_seed(20261006+fold)
