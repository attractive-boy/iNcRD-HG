"""Load only independently frozen miRNA/drug features and inner training folds."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'midrug_subgraph_20261006'
spec=importlib.util.spec_from_file_location('verified_mirna_common',PREVIOUS/'common.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
CLEAN,RELEASE=source.CLEAN,source.RELEASE
Adapter,Model,seed_all,sha256=source.Adapter,source.Model,source.seed_all,source.sha256
load_inner,make_graph=source.load_inner,source.make_graph

