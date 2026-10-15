"""The already-audited M123 and unmasked10 recipes, without a new model branch."""
import torch
from common import Adapter,make_graph,Reconstruction

CONFIGS={'control':dict(reconstruction_weight=0.),'unmasked10':dict(reconstruction_weight=10.)}
SIZES={'half':.5,'three_quarter':.75}


def make_adapter(data,name,device):
    a=Adapter(data,make_graph(data),device)
    if name=='unmasked10':a.model.add_module('reconstruction',Reconstruction(a.graph,a.cfg.hidden).to(device))
    elif name!='control':raise ValueError(name)
    return a


def make_optimizer(m):
    groups=[dict(params=[p for n,p in m.named_parameters() if not n.startswith('reconstruction.')],lr=1e-4)]
    if hasattr(m,'reconstruction'):groups.append(dict(params=m.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4))
    return torch.optim.RMSprop(groups,alpha=.99)


def mask_generator(device,seed):
    return torch.Generator(device=device).manual_seed(seed)
