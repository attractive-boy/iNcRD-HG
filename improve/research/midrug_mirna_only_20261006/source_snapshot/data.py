"""Strict miRNA/drug inputs; training never loads lncRNA-containing graphs."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.model_selection import StratifiedKFold
HERE=Path(__file__).resolve().parent
RELEASE=HERE.parents[2]/'release/iNcRD-HG-M1M2M3'
sys.path.insert(0,str(RELEASE))
from incrd_hg_m123.data import sha256,split_path,load_graph

ALLOWED_NODES={'miRNA','drug'}
ALLOWED_EDGES={('miRNA','MiMi','miRNA'),('drug','DrugDrug','drug'),('miRNA','MiDrug','drug'),('drug','rev_MiDrug','miRNA')}


def assert_clean(graph):
    assert set(graph.node_types)==ALLOWED_NODES
    assert set(graph.edge_types)==ALLOWED_EDGES
    assert not list(graph._global_store.keys())
    for node in graph.node_types:assert set(graph[node].keys())=={'x'}
    for edge in graph.edge_types:assert set(graph[edge].keys())=={'edge_index'}


def prepare():
    # One-time allowlist extraction; no lncRNA store is copied to the clean file.
    src=load_graph('MiDrug');graph=HeteroData()
    for node in ('miRNA','drug'):graph[node].x=src[node].x.clone()
    for edge in (('miRNA','MiMi','miRNA'),('drug','DrugDrug','drug')):graph[edge].edge_index=src[edge].edge_index.clone()
    for edge in (('miRNA','MiDrug','drug'),('drug','rev_MiDrug','miRNA')):graph[edge].edge_index=torch.empty((2,0),dtype=torch.long)
    assert_clean(graph)
    path=HERE/'graph.pt'
    if path.exists():raise FileExistsError(path)
    torch.save(graph,path)
    arrays={}
    with np.load(split_path('MiDrug',0)) as z:arrays['pairs'],arrays['y']=z['train_pairs'],z['train_y']
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=20261006)
    for f,(tr,va) in enumerate(cv.split(arrays['pairs'],arrays['y'])):arrays[f'train_{f}'],arrays[f'val_{f}']=tr,va
    np.savez_compressed(HERE/'inner_data.npz',**arrays)
    manifest=dict(status='prepared',allowed_node_types=sorted(ALLOWED_NODES),allowed_edge_types=sorted(ALLOWED_EDGES),
        graph_sha256=sha256(path),data_sha256=sha256(HERE/'inner_data.npz'),source_sha256=sha256(__file__),
        original_graph_sha256=sha256(RELEASE/'data/frozen/MiDrug_graph.pt'),original_split_sha256=sha256(split_path('MiDrug',0)),
        lncrna_features_relations_supervision_used=False,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


def load_inner(fold):
    meta=json.loads((HERE/'manifest.json').read_text());assert sha256(HERE/'inner_data.npz')==meta['data_sha256']
    with np.load(HERE/'inner_data.npz') as z:
        tr,va=z[f'train_{fold}'],z[f'val_{fold}'];pairs,y=z['pairs'],z['y']
    return dict(train_pairs=pairs[tr],train_y=y[tr],val_pairs=pairs[va],val_y=y[va],
        test_pairs=np.empty((0,2),dtype=np.int64),train_indices=tr,val_indices=va)


def make_graph(data):
    meta=json.loads((HERE/'manifest.json').read_text());assert sha256(HERE/'graph.pt')==meta['graph_sha256']
    graph=torch.load(HERE/'graph.pt',map_location='cpu');assert_clean(graph)
    pos=data['train_pairs'][data['train_y']==1];idx=torch.as_tensor(pos.T.copy(),dtype=torch.long)
    graph['miRNA','MiDrug','drug'].edge_index=idx;graph['drug','rev_MiDrug','miRNA'].edge_index=idx.flip(0)
    return graph

if __name__=='__main__':prepare()
