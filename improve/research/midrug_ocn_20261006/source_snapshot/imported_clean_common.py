"""Load only the independently frozen miRNA/drug inputs."""
import sys,json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
CLEAN=HERE.parent/'midrug_mirna_only_20261006'
RELEASE=HERE.parents[2]/'release/iNcRD-HG-M1M2M3'
sys.path.insert(0,str(RELEASE))
from incrd_hg_m123.data import sha256
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.model import Model
from incrd_hg_m123.engine import seed_all


def load_inner(fold):
    meta=json.loads((CLEAN/'manifest.json').read_text());assert sha256(CLEAN/'inner_data.npz')==meta['data_sha256']
    with np.load(CLEAN/'inner_data.npz') as z:
        tr,va=z[f'train_{fold}'],z[f'val_{fold}'];pairs,y=z['pairs'],z['y']
    return dict(train_pairs=pairs[tr],train_y=y[tr],val_pairs=pairs[va],val_y=y[va],
        test_pairs=np.empty((0,2),dtype=np.int64),train_indices=tr,val_indices=va)


def make_graph(data):
    meta=json.loads((CLEAN/'manifest.json').read_text());assert sha256(CLEAN/'graph.pt')==meta['graph_sha256']
    g=torch.load(CLEAN/'graph.pt',map_location='cpu')
    assert set(g.node_types)=={'miRNA','drug'} and all('lnc' not in str(e).lower() for e in g.edge_types)
    for node in g.node_types:assert set(g[node].keys())=={'x'}
    pos=data['train_pairs'][data['train_y']==1];e=torch.as_tensor(pos.T.copy(),dtype=torch.long)
    g['miRNA','MiDrug','drug'].edge_index=e;g['drug','rev_MiDrug','miRNA'].edge_index=e.flip(0)
    return g
