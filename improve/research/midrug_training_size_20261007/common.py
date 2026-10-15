"""Strict original-training, miRNA/drug-only inputs."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
STRICT=HERE.parent/'midrug_subgraph_20261006/common.py'
RECON=HERE.parent/'midrug_masked_reconstruction_20261006'
DEGREE=HERE.parent/'midrug_degree_encoder_20261006'
def read_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
base=read_module('strict_differentiable_completion_inputs',STRICT)
CLEAN,RELEASE=base.CLEAN,base.RELEASE
Adapter,seed_all,sha256=base.Adapter,base.seed_all,base.sha256
load_full_inner,make_graph=base.load_inner,base.make_graph
load_inner=load_full_inner
Reconstruction=read_module('original_reconstruction_differentiable_completion',RECON/'model.py').Reconstruction


def load_subset(fold,size):
    import json
    import numpy as np
    data=load_full_inner(fold)
    if size=='full':return data
    manifest=json.loads((HERE/'subset_manifest.json').read_text())
    assert sha256(HERE/'subsets.npz')==manifest['subsets_sha256']
    assert sha256(CLEAN/'inner_data.npz')==manifest['original_data_sha256']
    with np.load(HERE/'subsets.npz') as z:
        ix=z[f'{size}_local_f{fold}'];global_indices=z[f'{size}_global_f{fold}']
    np.testing.assert_array_equal(global_indices,data['train_indices'][ix])
    assert not np.intersect1d(global_indices,data['val_indices']).size
    return {**data,'train_pairs':data['train_pairs'][ix], 'train_y':data['train_y'][ix],
            'train_indices':global_indices,'omitted_train_indices':np.setdiff1d(data['train_indices'],global_indices)}
