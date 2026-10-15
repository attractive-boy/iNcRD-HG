"""Meaningful leakage/gradient tests for the nonlinear profile branch."""
import json
import numpy as np
import torch
from model import HERE,CONFIGS,make_adapter,load_fold,seed_all,negative_matrix
import check_modules


def check_profiles(adapter,data,count=12):
    m=adapter.model.eval();prof=m.profile
    if prof is None:return []
    ix=np.r_[np.flatnonzero(data['train_y']==1)[np.linspace(0,int((data['train_y']==1).sum())-1,count//2,dtype=int)],
             np.flatnonzero(data['train_y']==0)[np.linspace(0,int((data['train_y']==0).sum())-1,count//2,dtype=int)]]
    errors=[]
    with torch.no_grad():
        for i in ix:
            pair=adapter.pairs['train'][:,i:i+1];r,d=pair[:,0]
            A,N=adapter.ps.A,m.N
            before=prof(A,N,*pair)
            A2,N2=A.clone(),N.clone();A2[r,d]=0;N2[r,d]=0
            after=prof(A2,N2,*pair)
            # Independent input assembly after deleting the supervised record.
            keep=np.arange(len(data['train_y']))!=i
            remain={'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
            independent_N=negative_matrix(remain,A.shape,A.device)
            torch.testing.assert_close(independent_N,N2,rtol=0,atol=0)
            xr,xd=prof.masked_inputs(A,N,*pair)
            expected_r=A2[r:r+1];expected_d=A2[:,d].reshape(1,-1)
            if prof.channels==2:
                expected_r=torch.cat([expected_r,N2[r:r+1]],1)
                expected_d=torch.cat([expected_d,N2[:,d].reshape(1,-1)],1)
            torch.testing.assert_close(xr,expected_r,rtol=0,atol=0)
            torch.testing.assert_close(xd,expected_d,rtol=0,atol=0)
            torch.testing.assert_close(before,after,rtol=0,atol=0)
            errors.append(dict(index=int(i),label=int(data['train_y'][i]),branch_logit_deletion_error=float((before-after).abs().max())))
    return errors


def main():
    data=load_fold('MiDrug',0)
    seed_all(42);base=make_adapter(data,CONFIGS['control'],'cuda:0');base.model.eval()
    seed_all(42);a=make_adapter(data,CONFIGS['signed128_d2'],'cuda:0');a.model.eval()
    with torch.no_grad():
        base_p,zero_p=base.predict('val'),a.predict('val')
        np.testing.assert_allclose(base_p,zero_p,rtol=2e-6,atol=1e-6)
    # Nonzero readout ensures deletion is checked with an active branch.
    with torch.no_grad():
        a.model.profile.rout.weight.normal_(std=.03);a.model.profile.dout.weight.normal_(std=.03)
    records=check_profiles(a,data)
    checks=check_modules.check(a,count=12)
    a.model.train();loss=torch.nn.functional.binary_cross_entropy_with_logits(a.logits('train'),a.y)
    loss.backward()
    grads=[p.grad for p in a.model.profile.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert a.model.profile.renc[0].weight.grad.abs().max()>0
    altered={**data,'val_y':1-data['val_y'],'test_y':1-data['test_y']}
    seed_all(42);b=make_adapter(altered,CONFIGS['signed128_d2'],'cuda:0')
    torch.testing.assert_close(a.ps.A,b.ps.A,rtol=0,atol=0)
    torch.testing.assert_close(a.model.N,b.model.N,rtol=0,atol=0)
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',zero_branch_control_equivalence=True,
        control_max_prediction_error=float(np.max(np.abs(base_p-zero_p))),
        profile_deletion=records,m123_deletion=checks,heldout_label_invariance=True,finite_nonzero_gradients=True),indent=2))
    print('PASS initial profile deletion, M123 final-logit deletion, held-out label invariance, gradients',flush=True)

if __name__=='__main__':main()
