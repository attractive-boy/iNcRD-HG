"""Strict independent miRNA/drug data and verified original molecular descriptors."""
import importlib.util,json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
STRICT=HERE.parent/'midrug_subgraph_20261006/common.py'
RECON=HERE.parent/'midrug_masked_reconstruction_20261006'
def read_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=read_module('strict_molecular_inputs',STRICT)
CLEAN,RELEASE=base.CLEAN,base.RELEASE
Adapter,Model,seed_all,sha256=base.Adapter,base.Model,base.seed_all,base.sha256
load_inner,make_graph=base.load_inner,base.make_graph
Reconstruction=read_module('verified_reconstruction_module',RECON/'model.py').Reconstruction


def load_molecules(device):
    meta=json.loads((HERE/'input_manifest.json').read_text())
    assert sha256(HERE/'molecular_inputs.npz')==meta['molecular_inputs_sha256']
    assert sha256(CLEAN/'graph.pt')==meta['clean_graph_sha256']
    with np.load(HERE/'molecular_inputs.npz') as z:
        data={k:torch.as_tensor(z[k].copy(),dtype=torch.long,device=device)
              for k in ['atom_features','bond_features','edge_index','atom_drug','atom_offsets']}
    assert torch.equal(data['atom_drug'][data['edge_index'][0]],data['atom_drug'][data['edge_index'][1]])
    return data,meta
