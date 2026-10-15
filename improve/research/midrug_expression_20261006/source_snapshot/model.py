"""Original M123, train supervision and graph; original-expression transforms."""
from prepare import HERE,RELEASE,sha256
import json
import numpy as np
import torch
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.data import load_fold,build_graph
from incrd_hg_m123.engine import seed_all

CONFIGS={
    'original':dict(transform='original',nodes=[]),
    'mi_global':dict(transform='global',nodes=['miRNA']),
    'mi_column':dict(transform='column',nodes=['miRNA']),
    'rna_global':dict(transform='global',nodes=['miRNA','lncRNA']),
    'rna_column':dict(transform='column',nodes=['miRNA','lncRNA']),
    'rna_rowz':dict(transform='rowz',nodes=['miRNA','lncRNA']),
    'rna_columnz':dict(transform='columnz',nodes=['miRNA','lncRNA']),
    'rna_stats':dict(transform='stats',nodes=['miRNA','lncRNA']),
}


def make_adapter(data,cfg,device):
    info=json.loads((HERE/'provenance.json').read_text())
    assert info['expression_sha256']==sha256(HERE/'expression.npz')
    inputs={k:data[k] for k in ('train_pairs','train_y','val_pairs')}
    inputs['test_pairs']=np.empty((0,2),dtype=np.int64)
    graph=build_graph(inputs,'MiDrug')
    with np.load(HERE/'expression.npz') as z:
        for node in cfg['nodes']:graph[node].x=torch.from_numpy(z[node+'_'+cfg['transform']].copy())
    return Adapter(inputs,graph,device)
