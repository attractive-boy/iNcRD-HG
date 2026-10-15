"""Reuse the audited reconstruction-round loader without modifying its frozen code."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location('reconstruction_round_common',PARENT/'common.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
PREVIOUS,CLEAN,RELEASE=prior.PREVIOUS,prior.CLEAN,prior.RELEASE
Adapter,Model,seed_all,sha256=prior.Adapter,prior.Model,prior.seed_all,prior.sha256
load_inner,make_graph=prior.load_inner,prior.make_graph

