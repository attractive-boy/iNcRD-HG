"""Structural eligibility only; not a performance or model-selection test."""
import json
import numpy as np
import torch
from common import HERE,load_inner,make_graph
from model import rectangle_sum,completion_fields

def main():
    rows=[]
    for fold in range(3):
        data=load_inner(fold);g=make_graph(data)
        A=torch.zeros((g['miRNA'].num_nodes,g['drug'].num_nodes),dtype=torch.float64)
        pos=g['miRNA','MiDrug','drug'].edge_index;A[pos[0],pos[1]]=1
        K=torch.zeros_like(A,dtype=torch.bool);r,d=torch.as_tensor(data['train_pairs'].T);K[r,d]=True
        r,d=torch.as_tensor(data['val_pairs'].T);assert not bool(K[r,d].any())
        counts=rectangle_sum(A,A,A)[r,d];fields=completion_fields(A,K,torch.full_like(A,.5),'uniform')[r,d]
        soft=(fields>0).any(1);observed=counts>0
        rows.append(dict(fold=fold,queries=len(r),with_observed_path=int(observed.sum()),with_one_unknown_path=int(soft.sum()),
            newly_covered=int((~observed&soft).sum()),neither=int((~observed&~soft).sum()),
            by_missing_edge=(fields>0).sum(0).tolist()))
    out=dict(records=rows,read_heldout_labels=False,used_for_selection=False,
        interpretation='Existence of two observed anchors plus one unobserved edge, not evidence that the predicted missing edge is real.')
    (HERE/'coverage.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
