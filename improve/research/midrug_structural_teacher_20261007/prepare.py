"""Freeze same-type graph-clustered and random masks without labels or new data."""
from collections import deque
import json
import numpy as np
import torch
from common import HERE,CLEAN,sha256


def clustered(neighbors,quota,rng):
    n=len(neighbors);priority=rng.permutation(n);rank=np.argsort(priority)
    seen=set();chosen=[];queue=deque()
    for seed in priority:
        if len(chosen)==quota:break
        if int(seed) in seen:continue
        seen.add(int(seed));queue.append(int(seed))
        while queue and len(chosen)<quota:
            node=queue.popleft();chosen.append(node)
            for other in sorted(neighbors[node],key=lambda j:rank[j]):
                if other not in seen:seen.add(other);queue.append(other)
    mask=np.zeros(n,dtype=bool);mask[chosen]=True
    assert mask.sum()==quota
    return mask


def main():
    graph=torch.load(CLEAN/'graph.pt',map_location='cpu');assert set(graph.node_types)=={'miRNA','drug'}
    arrays={};records=[]
    for fold in range(3):
        rng=np.random.RandomState(20261007+fold)
        for node,relation in [('miRNA','MiMi'),('drug','DrugDrug')]:
            n=graph[node].num_nodes;edges=graph[node,relation,node].edge_index.numpy();quota=round(.5*n)
            neighbors=[set() for _ in range(n)]
            for a,b in edges.T:
                if a!=b:neighbors[a].add(int(b));neighbors[b].add(int(a))
            assert n-2*quota in [0,1]
            schedules={'block':[],'random':[]}
            for pair in range(400):
                first=clustered(neighbors,quota,rng)
                available=np.ones(n,dtype=bool)
                if n%2:available[rng.choice(np.flatnonzero(~first))]=False
                second=available&~first
                random=np.zeros(n,dtype=bool);random[rng.permutation(np.flatnonzero(available))[:quota]]=True
                random_second=available&~random
                if rng.rand()<.5:first,second=second,first
                schedules['block'].extend([first,second]);schedules['random'].extend([random,random_second])
            for kind in ['block','random']:
                value=np.stack(schedules[kind]);arrays[f'{kind}_{fold}_{node}']=value
                assert (value.sum(1)==quota).all()
                nonself=edges[:,edges[0]!=edges[1]]
                internal=(value[:,nonself[0]]&value[:,nonself[1]]).sum(1)
                boundary=(value[:,nonself[0]]^value[:,nonself[1]]).sum(1)
                records.append(dict(fold=fold,node=node,kind=kind,target_count=quota,
                    mean_internal_edge_fraction=float(np.mean(internal/nonself.shape[1])),mean_boundary_edge_fraction=float(np.mean(boundary/nonself.shape[1])),
                    min_node_exposures=int(value.sum(0).min()),max_node_exposures=int(value.sum(0).max())))
            block=arrays[f'block_{fold}_{node}'];random=arrays[f'random_{fold}_{node}']
            np.testing.assert_array_equal(block.reshape(400,2,n).sum(1),random.reshape(400,2,n).sum(1))
    path=HERE/'masks.npz';assert not path.exists();np.savez_compressed(path,**arrays)
    (HERE/'mask_manifest.json').write_text(json.dumps(dict(status='frozen',records=records,masks_sha256=sha256(path),
        source_sha256=sha256(HERE/'prepare.py'),graph_sha256=sha256(CLEAN/'graph.pt'),epochs=800,
        uses_association_labels=False,uses_same_type_original_graphs=True,paired_node_exposures_matched=True,
        unbalanced_precheck_retained='precheck_unbalanced_masks',no_lncrna=True),indent=2))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
