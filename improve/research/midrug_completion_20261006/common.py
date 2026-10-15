"""Independent pure-miRNA inputs shared with the audited subgraph experiment."""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
STRICT = HERE.parent/'midrug_subgraph_20261006/common.py'
spec = importlib.util.spec_from_file_location('strict_completion_inputs', STRICT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
load_inner, make_graph, sha256 = base.load_inner, base.make_graph, base.sha256
CLEAN, RELEASE = base.CLEAN, base.RELEASE
RECON = HERE.parent/'midrug_masked_reconstruction_20261006'
