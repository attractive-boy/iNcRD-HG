"""Reuse the verified, strictly miRNA/drug-only loader."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'midrug_subgraph_20261006'
spec=importlib.util.spec_from_file_location('clean_midrug_common',PREVIOUS/'common.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
CLEAN,Adapter,Model,seed_all,sha256=module.CLEAN,module.Adapter,module.Model,module.seed_all,module.sha256
load_inner,make_graph=module.load_inner,module.make_graph

