"""Frozen pure-miRNA inputs; original M123 plus unmasked reconstruction."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
STRICT=HERE.parent/'midrug_subgraph_20261006/common.py'
RECON=HERE.parent/'midrug_masked_reconstruction_20261006'
def read_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
base=read_module('strict_independent_ensemble_inputs',STRICT)
CLEAN,RELEASE=base.CLEAN,base.RELEASE
Adapter,seed_all,sha256=base.Adapter,base.seed_all,base.sha256
load_inner,make_graph=base.load_inner,base.make_graph
original=read_module('original_reconstruction_independent_ensemble',RECON/'model.py')
Reconstruction,make_adapter,mask_generator=original.Reconstruction,original.make_adapter,original.mask_generator
MEMBERS=8
CONFIGS={'unmasked10':dict(mask=False,weight=10.,ratio=.5)}

def member_seed(root,index):
    assert root>=0 and 0<=index<MEMBERS
    return root if index==0 else int(np.random.SeedSequence([root,index]).generate_state(1,dtype=np.uint32)[0])

def probability_mean(values):
    values=np.asarray(values)
    assert values.ndim==2 and values.shape[0]==MEMBERS
    assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
    return values.mean(axis=0,dtype=np.float64)

def make_optimizer(model):
    base=[p for n,p in model.named_parameters() if not n.startswith('reconstruction.')]
    aux=[p for n,p in model.named_parameters() if n.startswith('reconstruction.')]
    return torch.optim.RMSprop([dict(params=base,lr=1e-4),dict(params=aux,lr=1e-4,weight_decay=1e-4)],alpha=.99)
