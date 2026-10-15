"""Check split isolation, auxiliary label provenance, deletion and gradients."""
import json
import numpy as np
import torch
from model import HERE,CONFIGS,load_inner,make_adapter,seed_all,sha256
from data import load_graph
import check_modules


def main():
    all_val=[]
    g=load_graph('MiDrug')
    known=np.unique(g['lncRNA','LncDrug','drug'].edge_index.T.numpy(),axis=0)
    for fold in range(3):
        d=load_inner(fold);all_val.extend(d['val_indices'].tolist())
        np.testing.assert_array_equal(d['aux_pairs'][d['aux_y']==1],known)
        assert len(np.unique(d['aux_pairs'],axis=0))==len(d['aux_pairs'])
        assert not np.intersect1d(d['train_indices'],d['val_indices']).size
        assert len(d['test_pairs'])==0
    assert sorted(all_val)==list(range(len(all_val)))
    d=load_inner(0);seed_all(42);a=make_adapter(d,CONFIGS['shared32_a01'],'cuda:0')
    np.testing.assert_array_equal(a.graph['miRNA','MiDrug','drug'].edge_index.T.cpu().numpy(),d['train_pairs'][d['train_y']==1])
    checks=check_modules.check(a,count=12)
    a.model.train();aux_pairs=torch.as_tensor(d['aux_pairs'].T,device='cuda:0');aux_y=torch.as_tensor(d['aux_y'],dtype=torch.float32,device='cuda:0')
    aux,pen=a.model.auxiliary(a.graph,aux_pairs)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(aux,aux_y)+.001*pen;loss.backward()
    for name in ('rna','drug'):
        params=list(getattr(a.model.branch,name).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert any(p.grad.abs().max()>0 for p in params)
    # Auxiliary transfer cannot backpropagate to the message-graph encoder.
    assert all(p.grad is None for p in a.model.encoder.parameters())
    # The extra score itself is invariant to the query's graph membership.
    a.model.eval()
    with torch.no_grad():
        score,_=a.model.branch(a.dense[0],a.dense[1],a.pairs['val'],0)
        aux,_=a.model.auxiliary(a.graph,aux_pairs)
        changed=a.graph.clone();changed['miRNA','MiDrug','drug'].edge_index=changed['miRNA','MiDrug','drug'].edge_index[:,:0]
        other,_=a.model.auxiliary(changed,aux_pairs)
        torch.testing.assert_close(aux,other,rtol=0,atol=0)
    result=dict(status='passed',three_inner_folds_partition_original_training=True,auxiliary_positives_exactly_original_graph=True,
        auxiliary_negative_pairs_unique_and_disjoint=True,target_message_graph_inner_train_only=True,
        final_logit_deletion=checks,auxiliary_shared_rna_drug_gradients=True,auxiliary_logits_graph_independent=True,
        outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2));print('PASS split isolation, auxiliary provenance, deletion and shared gradients',flush=True)

if __name__=='__main__':main()
