"""Freeze feature-only groups; no pair labels or target relations are inspected."""
import json
import numpy as np
import torch
from sklearn.cluster import KMeans
from common import HERE,CLEAN,sha256

def main():
    manifest=json.loads((CLEAN/'manifest.json').read_text())
    assert sha256(CLEAN/'graph.pt')==manifest['graph_sha256']
    g=torch.load(CLEAN/'graph.pt',map_location='cpu')
    assert set(g.node_types)=={'miRNA','drug'}
    assert not g['miRNA','MiDrug','drug'].edge_index.numel()
    arrays={};records=[]
    for node,k in [('miRNA',16),('drug',8)]:
        x=g[node].x.numpy().astype(np.float64)
        assert np.isfinite(x).all()
        x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        fit=KMeans(n_clusters=k,n_init=10,random_state=20261006,algorithm='lloyd').fit(x)
        labels=fit.labels_;assert len(np.unique(labels))==k
        shuffled=np.random.default_rng(20261006+(node=='drug')).permutation(labels)
        np.testing.assert_array_equal(np.bincount(labels),np.bincount(shuffled))
        arrays[node]=labels;arrays['shuffled_'+node]=shuffled
        records.append(dict(node=node,groups=k,sizes=np.bincount(labels).tolist(),inertia=fit.inertia_))
    path=HERE/'groups.npz'
    if path.exists():
        with np.load(path) as old:
            for key,val in arrays.items():np.testing.assert_array_equal(old[key],val)
    else:np.savez_compressed(path,**arrays)
    info=dict(status='prepared',graph_sha256=sha256(CLEAN/'graph.pt'),groups_sha256=sha256(path),
        records=records,features='original node features, row L2 normalization, separate fixed KMeans16/8',
        no_target_graph_aggregation=True,no_labels=True,no_external_data=True,no_lncrna=True)
    (HERE/'groups.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))

if __name__=='__main__':main()
