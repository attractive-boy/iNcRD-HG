"""Training-internal folds and original LncDrug auxiliary labels only."""
import json,sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
HERE=Path(__file__).resolve().parent
RELEASE=HERE.parents[2]/'release/iNcRD-HG-M1M2M3'
sys.path.insert(0,str(RELEASE))
from incrd_hg_m123.data import split_path,sha256,load_graph,build_graph


def prepare():
    path=split_path('MiDrug',0)
    with np.load(path) as z:
        # Outer validation/test labels and pairs are not loaded by this experiment.
        pairs=z['train_pairs'];y=z['train_y']
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=20261006)
    arrays={}
    for f,(tr,va) in enumerate(cv.split(pairs,y)):
        arrays[f'train_{f}']=tr;arrays[f'val_{f}']=va
    g=load_graph('MiDrug')
    pos=np.unique(g['lncRNA','LncDrug','drug'].edge_index.T.numpy(),axis=0)
    nr,nd=g['lncRNA'].num_nodes,g['drug'].num_nodes
    mask=np.ones((nr,nd),dtype=bool);mask[pos[:,0],pos[:,1]]=False
    candidates=np.argwhere(mask)
    rng=np.random.default_rng(271828)
    neg=candidates[rng.choice(len(candidates),size=2*len(pos),replace=False)]
    arrays['aux_pairs']=np.concatenate([pos,neg]);arrays['aux_y']=np.r_[np.ones(len(pos)),np.zeros(len(neg))].astype(np.int64)
    output=HERE/'inner_data.npz'
    if output.exists():raise FileExistsError(output)
    np.savez_compressed(output,**arrays)
    provenance=dict(status='prepared',source_split_sha256=sha256(path),source_graph_sha256=sha256(RELEASE/'data/frozen/MiDrug_graph.pt'),
        data_sha256=sha256(output),source_sha256=sha256(__file__),outer_validation_read=False,test_read=False,
        train_count=len(y),aux_positive_count=len(pos),aux_sampled_negative_count=len(neg),
        auxiliary_source='Original frozen LncDrug relation only; complement negatives with seed 271828',
        folds=[dict(fold=f,train_count=len(arrays[f'train_{f}']),val_count=len(arrays[f'val_{f}'])) for f in range(3)])
    (HERE/'provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps(provenance),flush=True)


def load_inner(fold):
    info=json.loads((HERE/'provenance.json').read_text())
    assert sha256(HERE/'inner_data.npz')==info['data_sha256']
    path=split_path('MiDrug',0);assert sha256(path)==info['source_split_sha256']
    with np.load(path) as z:pairs,y=z['train_pairs'],z['train_y']
    with np.load(HERE/'inner_data.npz') as z:
        tr,va=z[f'train_{fold}'],z[f'val_{fold}']
        aux_pairs,aux_y=z['aux_pairs'],z['aux_y']
    assert not np.intersect1d(tr,va).size
    return dict(train_pairs=pairs[tr],train_y=y[tr],val_pairs=pairs[va],val_y=y[va],
        test_pairs=np.empty((0,2),dtype=np.int64),train_indices=tr,val_indices=va,aux_pairs=aux_pairs,aux_y=aux_y)

if __name__=='__main__':prepare()
