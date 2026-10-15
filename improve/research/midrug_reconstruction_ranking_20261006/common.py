"""Audited miRNA/drug-only loaders and the pre-existing training-only BPR sampler."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
RECON=HERE.parent/'midrug_masked_reconstruction_20261006'
RANKING=HERE.parent/'midrug_ranking_20261006/run_ranking.py'
def read_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=read_module('strict_recon_common',RECON/'common.py')
CLEAN,RELEASE,PREVIOUS=base.CLEAN,base.RELEASE,base.PREVIOUS
Adapter,Model,seed_all,sha256=base.Adapter,base.Model,base.seed_all,base.sha256
load_inner,make_graph=base.load_inner,base.make_graph
ranking=read_module('verified_training_ranking',RANKING)
SameDrugRanking,rank_scale=ranking.SameDrugRanking,ranking.rank_scale
# Historical loaders are not part of this experiment's permitted input path.
def unavailable(*args,**kwargs):raise RuntimeError('Use independent miRNA-only inputs, not historical graph/split loaders')
ranking.load_fold=ranking.load_graph=ranking.make_graph=ranking.make_adapter=unavailable
Reconstruction=read_module('verified_feature_reconstruction',RECON/'model.py').Reconstruction

