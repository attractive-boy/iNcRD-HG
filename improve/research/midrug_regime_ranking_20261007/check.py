"""Exact pair enumeration, explicit gradients and original-model query checks."""
import json
import numpy as np
import torch
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,ranking,pair_counts,make_adapter,make_optimizer,step,mask_generator

QUERY_CHECK_SOURCE=DEGREE/'check.py'
query=read_module('regime_ranking_deletion',QUERY_CHECK_SOURCE);query.make_adapter=make_adapter
deletion_check=query.deletion_check


def oracle(scores,labels,groups,kind):
    pairs=[(i,j) for i in range(len(scores)) for j in range(len(scores))
           if labels[i]==1 and labels[j]==0 and (kind=='all' or ((groups[i]!=groups[j])==(kind=='cross')))]
    grad=np.zeros_like(scores);value=0.
    if not pairs:return value,grad
    for i,j in pairs:
        delta=scores[j]-scores[i];value+=np.logaddexp(0.,delta)/len(pairs)
        derivative=1./(1.+np.exp(-delta))/len(pairs);grad[j]+=derivative;grad[i]-=derivative
    return value,grad


def main():
    scores=torch.tensor([-.7,.4,.4,-1.,1.3,.2],dtype=torch.float64,requires_grad=True)
    labels=torch.tensor([1,0,1,0,0,1]);groups=torch.tensor([0,1,1,0,2,3])
    counts=pair_counts(labels,groups);computed={}
    for kind in CONFIGS:
        loss=ranking(scores,labels,groups,kind,block=2);grad=torch.autograd.grad(loss,scores)[0]
        value,expected=oracle(scores.detach().numpy(),labels.numpy(),groups.numpy(),kind)
        np.testing.assert_allclose(float(loss),value,atol=1e-14,rtol=1e-14)
        np.testing.assert_allclose(grad.numpy(),expected,atol=1e-14,rtol=1e-14)
        assert torch.autograd.gradcheck(lambda z:ranking(z,labels,groups,kind,block=2),(scores,),eps=1e-6,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(loss,ranking(scores+17,labels,groups,kind,block=3),atol=1e-14,rtol=1e-14)
        assert (grad[labels==1]<=0).all() and (grad[labels==0]>=0).all();computed[kind]=loss.detach()
    torch.testing.assert_close(computed['all'],(computed['cross']*counts['cross']+computed['within']*counts['within'])/counts['all'],atol=1e-14,rtol=1e-14)
    empty=ranking(scores,labels,torch.zeros_like(groups),'cross');assert empty==0
    torch.testing.assert_close(torch.autograd.grad(empty,scores)[0],torch.zeros_like(scores),atol=0,rtol=0)
    previous=json.loads((HERE.parent/'midrug_degree_regime_20261007/precheck.json').read_text());records=[]
    for fold in range(3):
        data=load_inner(fold);seed_all(42);base=make_adapter(data,'unmasked10','cuda:0');base.model.eval()
        initial={k:v.detach().clone() for k,v in base.model.state_dict().items()}
        with torch.no_grad():p0=base.predict('val')
        ref=next(r for r in previous['records'] if r['fold']==fold and r['config']=='unmasked10')
        np.testing.assert_array_equal(base.rank_cuts,ref['thresholds'])
        assert torch.bincount(base.rank_groups,minlength=4).cpu().tolist()==[r['count'] for r in ref['train']]
        assert base.rank_counts['all']==base.rank_counts['within']+base.rank_counts['cross']
        for name in CONFIGS:
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.eval()
            for key,value in initial.items():torch.testing.assert_close(value,a.model.state_dict()[key],atol=0,rtol=0)
            with torch.no_grad():p1=a.predict('val')
            np.testing.assert_allclose(p0,p1,atol=2e-5,rtol=2e-5)
            opt=make_optimizer(a.model);gen=mask_generator('cuda:0',20261006+fold)
            # Use the full-weight schedule for a single mechanism-check update;
            # this is not a formal training trajectory and advances BN exactly once.
            losses=step(a,opt,60,gen);assert int(a.model.eb.num_batches_tracked)==1
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
            seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0')
            b.model.load_state_dict(a.model.state_dict());a.model.eval();b.model.eval()
            torch.testing.assert_close(a.rank_groups,b.rank_groups,atol=0,rtol=0)
            with torch.no_grad():
                pa,pb=a.predict('val'),b.predict('val');la,lb=a.logits('train'),b.logits('train')
                ra,rb=ranking(la,a.y,a.rank_groups,name),ranking(lb,b.y,b.rank_groups,name)
            np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(ra,rb,atol=2e-5,rtol=2e-5)
            records.append(dict(config=name,fold=fold,rank_counts=a.rank_counts,initial_error=float(abs(p0-p1).max()),
                losses=losses,heldout_flip_error=float(abs(pa-pb).max()),explicit_deletions=deletion_check(a,data,name,count=4)))
            print('PASS',name,fold,flush=True);del a,b,opt;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,
        explicit_loss_gradient_oracle=True,float64_gradcheck=True,partition_identity=True,zero_comparison_loss_gradient=True,
        query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    print('ALL PASSED',flush=True)


if __name__=='__main__':main()
