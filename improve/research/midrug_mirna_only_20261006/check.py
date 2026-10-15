"""Strict scope, entity alignment, self-label exclusion and gradient checks."""
import json
import numpy as np
import torch
from model import HERE,CONFIGS,GroupEvidence,GroupPairStruct,make_adapter,load_inner,seed_all,assert_clean
import check_modules


def main():
    d=load_inner(0);seed_all(42);control=make_adapter(d,'control','cuda:0');control.model.eval()
    seed_all(42);a=make_adapter(d,'combined','cuda:0');a.model.eval();assert_clean(a.graph)
    for name,_ in a.model.named_parameters():assert 'lnc' not in name.lower()
    for edge in a.graph.edge_types:torch.testing.assert_close(a.graph[edge].edge_index,control.graph[edge].edge_index,rtol=0,atol=0)
    with torch.no_grad():p0,p1=control.predict('val'),a.predict('val')
    np.testing.assert_allclose(p0,p1,rtol=2e-5,atol=2e-5)
    errors=[]
    ix=np.r_[np.flatnonzero(d['train_y']==1)[[0,111,411]],np.flatnonzero(d['train_y']==0)[[0,111,411]]]
    for i in ix:
        keep=np.arange(len(d['train_y']))!=i
        deleted={**d,'train_pairs':d['train_pairs'][keep],'train_y':d['train_y'][keep]}
        ev=GroupEvidence(deleted,'cuda:0');pair=a.pairs['train'][:,i:i+1]
        expected=a.ps.evidence.features(*pair,'combined');actual=ev.features(*pair,'combined')
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        errors.append(dict(index=int(i),label=int(d['train_y'][i]),max_group_feature_error=float((expected-actual).abs().max())))
    original=check_modules.PairStruct
    try:
        check_modules.PairStruct=lambda D,p,c:GroupPairStruct(D,p,c,a.ps.evidence,'combined')
        checks=check_modules.check(a,count=12)
    finally:check_modules.PairStruct=original
    a.model.train();loss=torch.nn.functional.binary_cross_entropy_with_logits(a.logits('train'),a.y);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
    assert a.model.pc[0].weight.grad[:,17:].abs().max()>0
    val=[]
    for f in range(3):
        dd=load_inner(f);assert not np.intersect1d(dd['train_indices'],dd['val_indices']).size
        val.extend(dd['val_indices'].tolist())
    assert sorted(val)==list(range(len(val)))
    rec=dict(status='passed',allowed_node_types=sorted(a.graph.node_types),allowed_edge_types=sorted(a.graph.edge_types),
        no_lncrna_parameters_features_relations_or_supervision=True,inner_folds_partition_original_train=True,
        initial_control_max_error=float(np.max(np.abs(p0-p1))),group_record_deletion=errors,final_logit_deletion=checks,
        finite_nonzero_group_channel_gradients=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(rec,indent=2));print('PASS strict miRNA-only scope, split isolation, group query deletion and model checks',flush=True)

if __name__=='__main__':main()
