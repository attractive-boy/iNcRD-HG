"""Strict frozen miRNA/drug inputs and original reconstruction objective."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
STRICT=HERE.parent/'midrug_subgraph_20261006/common.py'
RECON=HERE.parent/'midrug_masked_reconstruction_20261006'
DEGREE=HERE.parent/'midrug_degree_encoder_20261006'
def read_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
base=read_module('strict_variance_covariance_inputs',STRICT)
CLEAN,RELEASE=base.CLEAN,base.RELEASE
Adapter,seed_all,sha256=base.Adapter,base.seed_all,base.sha256
load_inner,make_graph=base.load_inner,base.make_graph
Reconstruction=read_module('original_reconstruction_variance_covariance',RECON/'model.py').Reconstruction
