"""Frozen pure-miRNA inputs, original reconstruction and fixed nested folds."""
import importlib.util
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
HERE=Path(__file__).resolve().parent
STRICT=HERE.parent/'midrug_subgraph_20261006/common.py'
RECON=HERE.parent/'midrug_masked_reconstruction_20261006'
RETRIEVAL=HERE.parent/'midrug_label_retrieval_20261007'
def read_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
base=read_module('strict_retrieval_crossfit_inputs',STRICT)
CLEAN,RELEASE=base.CLEAN,base.RELEASE
Adapter,seed_all,sha256=base.Adapter,base.seed_all,base.sha256
load_inner,make_graph=base.load_inner,base.make_graph
rec=read_module('original_reconstruction_crossfit',RECON/'model.py')
Reconstruction,mask_generator=rec.Reconstruction,rec.mask_generator

def nested_data(fold,k):
    outer=load_inner(fold)
    if k==-1:return outer,np.arange(len(outer['val_y']))
    tr,va=list(StratifiedKFold(5,shuffle=True,random_state=20261006+fold).split(outer['train_pairs'],outer['train_y']))[k]
    data=dict(train_pairs=outer['train_pairs'][tr],train_y=outer['train_y'][tr],
        val_pairs=outer['train_pairs'][va],val_y=outer['train_y'][va],test_pairs=np.empty((0,2),dtype=np.int64),
        train_indices=outer['train_indices'][tr],val_indices=outer['train_indices'][va])
    assert not np.intersect1d(data['train_indices'],data['val_indices']).size
    return data,va

def make_base(data,device):return rec.make_adapter(data,'unmasked10',device)
