"""Strict miRNA/drug inputs and five nested partitions within each training fold."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'midrug_subgraph_20261006'
spec=importlib.util.spec_from_file_location('strict_inner_common',PREVIOUS/'common.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
CLEAN,Adapter,Model,seed_all,sha256=module.CLEAN,module.Adapter,module.Model,module.seed_all,module.sha256
load_inner,make_graph=module.load_inner,module.make_graph

def nested_data(fold,k):
    outer=load_inner(fold);n=len(outer['train_y'])
    if k==-1:
        return outer,np.arange(n)
    splits=list(StratifiedKFold(5,shuffle=True,random_state=20261006+fold).split(outer['train_pairs'],outer['train_y']))
    tr,va=splits[k]
    assert not np.intersect1d(tr,va).size and len(tr)+len(va)==n
    data=dict(train_pairs=outer['train_pairs'][tr],train_y=outer['train_y'][tr],
        val_pairs=outer['train_pairs'][va],val_y=outer['train_y'][va],test_pairs=np.empty((0,2),dtype=np.int64),
        train_indices=outer['train_indices'][tr],val_indices=outer['train_indices'][va])
    assert not np.intersect1d(data['train_indices'],outer['val_indices']).size
    assert not np.intersect1d(data['val_indices'],outer['val_indices']).size
    return data,va

def import_source(name,file):
    spec=importlib.util.spec_from_file_location(name,PREVIOUS/file)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

Extractor=import_source('verified_extractor','extract.py').Extractor
SubgraphGNN=import_source('verified_subgraph','model.py').SubgraphGNN
SG=dict(k=16,hops=1,hidden=32,layers=3,covariates=False,signed=False)

