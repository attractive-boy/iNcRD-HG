"""Strict original miRNA/drug loader and immutable reconstruction module."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
STRICT=HERE.parent/'midrug_subgraph_20261006/common.py'
RECON=HERE.parent/'midrug_masked_reconstruction_20261006/model.py'
def read_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=read_module('strict_feedback_geometry',STRICT)
CLEAN,RELEASE=base.CLEAN,base.RELEASE
Adapter,seed_all,sha256=base.Adapter,base.seed_all,base.sha256
make_graph=base.make_graph
def load_inner(fold):return {**base.load_inner(fold),"inner_fold":fold}
Reconstruction=read_module('original_feedback_reconstruction',RECON).Reconstruction
