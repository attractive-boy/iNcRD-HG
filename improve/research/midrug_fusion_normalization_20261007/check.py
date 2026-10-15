"""Check exact input decomposition and full query-local graph inference."""
import json
import numpy as np
import torch
from common import HERE,DEGREE,read_module,load_inner,seed_all,sha256
from model import CONFIGS,decompose,combine,make_adapter,make_optimizer,step,mask_generator

QUERY_CHECK_SOURCE=DEGREE/'check.py'
query=read_module('fusion_normalization_deletion',QUERY_CHECK_SOURCE);query.make_adapter=make_adapter
deletion_check=query.deletion_check


def oracle(h):
    mean=h.mean(-1,keepdims=True);scale=np.sqrt(((h-mean)**2).mean(-1,keepdims=True)+1e-5)
    return (h-mean)/scale,np.concatenate([np.arcsinh(mean),np.log1p(scale)],-1)


def moment_check(a):
    a.model.eval()
    with torch.no_grad():parts=a.model.parts(a.pairs['train'],a.ps,a.qis['train'],a.ease['train'],a.dense)
    errors=[];inverse=[]
    for part in parts[:-2]:
        x=part.double();u,t=decompose(x);eu,et=oracle(x.cpu().numpy())
        np.testing.assert_allclose(u.cpu(),eu,atol=1e-11,rtol=1e-11);np.testing.assert_allclose(t.cpu(),et,atol=1e-11,rtol=1e-11)
        reconstructed=u*torch.expm1(t[:,1:2])+torch.sinh(t[:,:1])
        torch.testing.assert_close(reconstructed,x,atol=1e-10,rtol=1e-11)
        errors.append(max(float(abs(u.cpu().numpy()-eu).max()),float(abs(t.cpu().numpy()-et).max())))
        inverse.append(float(abs(reconstructed-x).max()))
    full=combine(parts,a.model.fusion_mode)
    one=torch.cat([combine([p[i:i+1] for p in parts],a.model.fusion_mode) for i in [0,1,2]])
    torch.testing.assert_close(full[:3],one,atol=1e-5,rtol=1e-5)
    return dict(oracle_error=max(errors),inverse_error=max(inverse),transform_batch_error=float(abs(full[:3]-one).max()))


def main():
    gen=torch.Generator().manual_seed(17)
    h=torch.randn(11,5,generator=gen,dtype=torch.float64,requires_grad=True)
    u,t=decompose(h);eu,et=oracle(h.detach().numpy())
    np.testing.assert_allclose(u.detach(),eu,atol=1e-13,rtol=1e-13);np.testing.assert_allclose(t.detach(),et,atol=1e-13,rtol=1e-13)
    assert torch.autograd.gradcheck(decompose,(h,),eps=1e-6,atol=1e-6,rtol=1e-5)
    reconstructed=u*torch.expm1(t[:,1:2])+torch.sinh(t[:,:1])
    torch.testing.assert_close(reconstructed,h,atol=1e-13,rtol=1e-13)
    torch.testing.assert_close(torch.autograd.grad(reconstructed.sum(),h)[0],torch.ones_like(h),atol=1e-12,rtol=1e-12)
    perm=torch.tensor([3,4,0,2,1]);pu,pt=decompose(h[:,perm]);torch.testing.assert_close(pu,u[:,perm],atol=1e-13,rtol=1e-13);torch.testing.assert_close(pt,t,atol=1e-13,rtol=1e-13)
    ru,rt=decompose(h.flip(0));torch.testing.assert_close(ru,u.flip(0));torch.testing.assert_close(rt,t.flip(0))
    for value in [0.,5.,-5.]:
        c=torch.full((3,5),value,dtype=torch.float64,requires_grad=True);cu,ct=decompose(c)
        assert (cu==0).all() and torch.isfinite(ct).all()
        assert torch.isfinite(torch.autograd.grad(cu.sum()+ct.sum(),c)[0]).all()
    records=[]
    for fold in range(3):
        data=load_inner(fold);seed_all(42);base=make_adapter(data,'unmasked10','cuda:0');base.model.eval()
        initial={k:v.detach().clone() for k,v in base.model.state_dict().items()}
        with torch.no_grad():p0=base.predict('val')
        normalized_initial=None
        for name in CONFIGS:
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.eval()
            n=a.model.extra_width;assert n==12
            for key,value in initial.items():
                actual=a.model.state_dict()[key]
                if key=='lin1.weight':
                    assert (actual[:,-n:]==0).all();actual=actual[:,:-n]
                torch.testing.assert_close(value,actual,atol=0,rtol=0)
            with torch.no_grad():p1=a.predict('val')
            if name=='raw':reference=p0
            elif normalized_initial is None:normalized_initial=p1;reference=p1
            else:reference=normalized_initial
            np.testing.assert_allclose(reference,p1,atol=2e-5,rtol=2e-5)
            opt=make_optimizer(a.model);losses=step(a,opt,60,mask_generator('cuda:0',20261006+fold))
            assert int(a.model.eb.num_batches_tracked)==1
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
            assert any(p.grad is not None and p.grad.abs().max()>0 for p in a.model.encoder.parameters())
            extra=a.model.lin1.weight[:,-n:];gradient=a.model.lin1.weight.grad[:,-n:]
            if name=='direction_scale':assert extra.abs().max()>0 and gradient.abs().max()>0
            else:assert (extra==0).all() and (gradient==0).all()
            seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');b.model.load_state_dict(a.model.state_dict());a.model.eval();b.model.eval()
            with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
            np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
            records.append(dict(config=name,fold=fold,initial_error=float(abs(reference-p1).max()),losses=losses,
                heldout_flip_error=float(abs(pa-pb).max()),extra_weight_gradient_norm=float(gradient.norm()),
                moment_check=moment_check(a),explicit_deletions=deletion_check(a,data,name,count=4)))
            print('PASS',name,fold,flush=True);del a,b,opt;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,independent_numpy_transform=True,
        float64_gradcheck=True,inverse_output_and_gradient=True,row_column_permutation=True,constant_blocks_finite=True,
        raw_initial_matches_original=True,normalized_initial_arms_match=True,
        query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    print('ALL PASSED',flush=True)


if __name__=='__main__':main()
