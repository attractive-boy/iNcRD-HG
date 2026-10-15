"""Change miRNA expression coordinates, preserve reconstruction targets and graph."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE, Adapter, Reconstruction, make_graph, sha256

CONFIGS={name:dict(representation=name,reconstruction_weight=10.)
         for name in ['cell_gaussian','cell_minmax','row_gaussian']}

class OriginalReconstruction(Reconstruction):
    def __init__(self,graph,hidden):
        super().__init__(graph,hidden)
        for name in ('miRNA','drug'):self.register_buffer(name+'_target',graph[name].x.clone())
    def loss(self,encoder,graph,indices,masked):
        assert not masked
        h=encoder.layer(encoder.original.conv1,graph.x_dict)
        h={name:F.leaky_relu(value,.1) for name,value in h.items()}
        h=encoder.layer(encoder.original.conv2,h)
        losses={}
        for name,ix in indices.items():
            target=(getattr(self,name+'_target')[ix]-getattr(self,name+'_mean'))/getattr(self,name+'_scale')
            losses[name]=F.mse_loss(self.decoder[name](h[name][ix]),target)
        return .5*(losses['miRNA']+losses['drug']),losses

def make_adapter(data,name,device):
    assert name in CONFIGS
    original=make_graph(data);graph=original.clone()
    manifest=json.loads((HERE/'input_manifest.json').read_text())
    assert sha256(HERE/'covariates.npz')==manifest['covariates_sha256']
    with np.load(HERE/'covariates.npz') as z:graph['miRNA'].x=torch.from_numpy(z[name].copy())
    a=Adapter(data,graph,device)
    a.model.add_module('reconstruction',OriginalReconstruction(original,a.cfg.hidden).to(device))
    return a

def mask_generator(device,fold):return torch.Generator(device=device).manual_seed(20261006+fold)
