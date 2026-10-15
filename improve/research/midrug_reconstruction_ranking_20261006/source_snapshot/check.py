"""Verify train-only pair sampling, monotonic ranking loss and combined gradients."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,seed_all,SameDrugRanking
from model import make_adapter,mask_generator
import check_modules

def main():
    data=load_inner(0);seed_all(42);a=make_adapter(data,'rec10_bpr010','cuda:0');a.model.eval()
    rank=SameDrugRanking(data['train_pairs'],data['train_y'],'cuda:0',42,'mixed_middle')
    logits=a.logits('train');negative=rank.sample(logits);anchors=rank.anchors
    pairs=torch.as_tensor(data['train_pairs'],device='cuda:0');y=a.y
    assert y[anchors].eq(1).all() and y[negative].eq(0).all()
    assert (pairs[negative,1]==pairs[anchors,1,None]).all()
    assert negative.min()>=0 and negative.max()<len(y)
    old=rank.generator.get_state();loss0=rank.loss(logits);rank.generator.set_state(old)
    improved=logits+1.0*y;loss1=rank.loss(improved);assert loss1<loss0
    rec=a.model.reconstruction;indices=rec.masks(a.graph,.5,mask_generator('cuda:0',0))
    aux,_=rec.loss(a.model.encoder,a.graph,indices,False)
    combined=F.binary_cross_entropy_with_logits(logits,y)+.1*loss0+10*aux
    combined.backward();assert torch.isfinite(combined)
    assert rec.decoder['miRNA'][-1].weight.grad.abs().max()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
    seed_all(42);b=make_adapter(data,'control','cuda:0');b.model.eval()
    for name,x in b.model.state_dict().items():torch.testing.assert_close(x,a.model.state_dict()[name],atol=0,rtol=0)
    with torch.no_grad():p0,p1=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(p0,p1,atol=1e-6,rtol=1e-5)
    checks=check_modules.check(a,count=6)
    r=dict(status='passed',training_only_same_drug_negative_sampling=True,ranking_coverage=rank.coverage,
        original_ranking_loss=float(loss0),positive_scores_increased_loss=float(loss1),finite_combined_gradients=True,
        initial_base_weights_equal=True,initial_prediction_max_error=float(abs(p0-p1).max()),full_logit_deletion=checks,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

if __name__=='__main__':main()

