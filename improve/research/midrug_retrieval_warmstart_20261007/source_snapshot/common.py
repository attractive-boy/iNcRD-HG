"""Frozen original miRNA/drug inputs and reconstruction module."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
STRICT=HERE.parent/'midrug_subgraph_20261006/common.py'
RECON=HERE.parent/'midrug_masked_reconstruction_20261006'
PREVIOUS=HERE.parent/'midrug_label_retrieval_20261007'
def read_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
base=read_module('strict_retrieval_warmstart_inputs',STRICT)
CLEAN,RELEASE=base.CLEAN,base.RELEASE
Adapter,seed_all,sha256=base.Adapter,base.seed_all,base.sha256
load_inner,make_graph=base.load_inner,base.make_graph
Reconstruction=read_module('original_reconstruction_retrieval_warmstart',RECON/'model.py').Reconstruction
