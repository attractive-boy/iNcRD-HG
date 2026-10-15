"""M123 inference retained; compare ranking alone with reconstruction plus ranking."""
import torch
from common import Adapter,make_graph,Reconstruction

CONFIGS={
    'bpr010':dict(rank_weight=.1,reconstruction_weight=0.),
    'bpr020':dict(rank_weight=.2,reconstruction_weight=0.),
    'rec3_bpr010':dict(rank_weight=.1,reconstruction_weight=3.),
    'rec10_bpr010':dict(rank_weight=.1,reconstruction_weight=10.),
    'rec10_bpr020':dict(rank_weight=.2,reconstruction_weight=10.),
}

def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name!='control' and CONFIGS[name]['reconstruction_weight']:
        a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    return a

def mask_generator(device,fold):
    return torch.Generator(device=device).manual_seed(20261006+fold)

