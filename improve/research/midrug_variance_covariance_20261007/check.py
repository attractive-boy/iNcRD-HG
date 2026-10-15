"""Independent moment formulas, gradients, model scope and physical deletion."""
import json
import numpy as np
import torch
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,regularizers,training_forward,make_adapter,make_optimizer,step,mask_generator

QUERY_CHECK_SOURCE=DEGREE/'check.py'
query=read_module('variance_covariance_deletion',QUERY_CHECK_SOURCE);query.make_adapter=make_adapter
deletion_check=query.deletion_check


def oracle(h):
    centered=h-h.mean(1,keepdims=True)
    z=centered/np.sqrt((centered**2).mean(1,keepdims=True)+1e-5)
    c=np.cov(z,rowvar=False,ddof=1)
    var=np.maximum(0,1-np.sqrt(c.diagonal()+1e-4)).mean()
    cov=sum(c[i,j]**2 for i in range(c.shape[0]) for j in range(c.shape[1]) if i!=j)/c.shape[0]
    return np.array([var,cov])


def main():
    gen=torch.Generator().manual_seed(17)
    h=torch.randn(11,5,generator=gen,dtype=torch.float64,requires_grad=True)
    out=torch.stack(regularizers(h));np.testing.assert_allclose(out.detach().numpy(),oracle(h.detach().numpy()),atol=1e-13,rtol=1e-13)
    assert torch.autograd.gradcheck(lambda x:torch.stack(regularizers(x)),(h,),eps=1e-6,atol=1e-6,rtol=1e-5)
    for changed in [h.flip(0),h[:,torch.tensor([3,4,0,2,1])],h+torch.arange(11,dtype=h.dtype)[:,None]]:
        torch.testing.assert_close(out,torch.stack(regularizers(changed)),atol=1e-12,rtol=1e-12)
    constant=torch.ones(11,5,dtype=torch.float64,requires_grad=True);v,c=regularizers(constant)
    assert abs(float(v)-.99)<1e-12 and c==0
    assert torch.isfinite(torch.autograd.grad(v+c,constant)[0]).all()
    # Both antipodal and orthogonally balanced examples have zero mean on the sphere;
    # that alone does not distinguish rank1 from rank2 (reason not to copy HyperGRL).
    antipodal=np.array([[1.,0.],[-1.,0.],[1.,0.],[-1.,0.]])
    balanced=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    assert (antipodal.mean(0)**2).sum()==(balanced.mean(0)**2).sum()==0
    assert np.linalg.matrix_rank(antipodal)==1 and np.linalg.matrix_rank(balanced)==2
    records=[]
    for fold in range(3):
        data=load_inner(fold);seed_all(42);base=make_adapter(data,'unmasked10','cuda:0');base.model.eval()
        initial={k:v.detach().clone() for k,v in base.model.state_dict().items()}
        with torch.no_grad():p0=base.predict('val')
        for name in CONFIGS:
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.eval()
            for key,value in initial.items():torch.testing.assert_close(value,a.model.state_dict()[key],atol=0,rtol=0)
            with torch.no_grad():
                p1=a.predict('val');ordinary=a.logits('train');captured,v,c=training_forward(a)
            # Repeated CUDA sparse aggregation has roundoff-level differences.
            torch.testing.assert_close(ordinary,captured,atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(p0,p1,atol=2e-5,rtol=2e-5)
            # Make sure this objective can update the shared encoder at initialization.
            _,v,c=training_forward(a)
            auxiliary=CONFIGS[name]['var']*v+CONFIGS[name]['cov']*c
            grads=torch.autograd.grad(auxiliary,list(a.model.encoder.parameters()))
            assert all(torch.isfinite(g).all() for g in grads) and sum(float(g.square().sum()) for g in grads)>0
            opt=make_optimizer(a.model);losses=step(a,opt,60,mask_generator('cuda:0',20261006+fold))
            assert int(a.model.eb.num_batches_tracked)==1 and not a.model.lin1._forward_pre_hooks
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
            seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0')
            b.model.load_state_dict(a.model.state_dict());a.model.eval();b.model.eval()
            with torch.no_grad():
                pa,pb=a.predict('val'),b.predict('val');la,va,ca=training_forward(a);lb,vb,cb=training_forward(b)
            np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
            for x,y in [(la,lb),(va,vb),(ca,cb)]:torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-5)
            records.append(dict(config=name,fold=fold,initial_error=float(abs(p0-p1).max()),losses=losses,
                captured_forward_error=float(abs(ordinary-captured).max()),
                heldout_flip_error=float(abs(pa-pb).max()),explicit_deletions=deletion_check(a,data,name,count=4)))
            print('PASS',name,fold,flush=True);del a,b,opt,grads;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,independent_numpy_moments=True,
        float64_gradcheck=True,row_column_permutation_invariance=True,no_extra_inference_parameters=True,
        finite_constant_representation_loss_gradient=True,zero_mean_not_sufficient_counterexample=True,
        query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    print('ALL PASSED',flush=True)


if __name__=='__main__':main()
