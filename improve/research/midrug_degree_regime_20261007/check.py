"""Independent routing, initial identity, gradients and physical query deletion."""
import json
import numpy as np
import torch
from common import HERE, DEGREE, read_module, load_inner, seed_all, sha256
from model import CONFIGS, make_adapter, make_optimizer, step, mask_generator, degree_group, pair_hash, N_DRUGS, PRIME

QUERY_CHECK_SOURCE=DEGREE/'check.py'
query=read_module('degree_regime_deletion',QUERY_CHECK_SOURCE);query.make_adapter=make_adapter
deletion_check=query.deletion_check


def route_reference(a,data):
    matrix=np.zeros((605,N_DRUGS),dtype=np.int64)
    pos=data['train_pairs'][data['train_y']==1];matrix[pos[:,0],pos[:,1]]=1
    cuts=a.model.degree_cuts.detach().cpu().numpy();rc=a.model.random_cuts.detach().cpu().numpy()
    result={}
    for part in ['train','val']:
        pairs=data[part+'_pairs'];r,d=pairs.T;own=matrix[r,d]
        values=np.stack((matrix.sum(1)[r]-own,matrix.sum(0)[d]-own),-1)
        expected=(values[:,0]>cuts[0]).astype(int)*2+(values[:,1]>cuts[1]).astype(int)
        actual=degree_group(a.ps,*a.pairs[part],a.model.degree_cuts).cpu().numpy()
        np.testing.assert_array_equal(expected,actual)
        hashed=((r*N_DRUGS+d)*1103515245+20261007)%PRIME
        np.testing.assert_array_equal(hashed,pair_hash(*a.pairs[part]).cpu().numpy())
        random=np.searchsorted(rc,hashed,side='left')
        result[part]=dict(degree_counts=np.bincount(expected,minlength=4).tolist(),random_counts=np.bincount(random,minlength=4).tolist())
        if part=='train':assert result[part]['degree_counts']==result[part]['random_counts']
        actual_gate=a.model.gate(a.pairs[part],a.ps).detach().cpu().numpy()
        weights=np.full((len(pairs),4),.25) if a.model.routing=='uniform' else np.eye(4)[expected if a.model.routing=='degree' else random]
        np.testing.assert_array_equal(actual_gate,weights)
    # Explicit physical deletion for every degree-boundary training pair, not
    # merely subtraction through the same helper used by forward.
    selected=np.r_[np.flatnonzero(data['train_y']==1)[:4],np.flatnonzero(data['train_y']==0)[:2]]
    for i in selected:
        r,d=data['train_pairs'][i];other=matrix.copy();other[r,d]=0
        expected=int(other[r,:].sum()>cuts[0])*2+int(other[:,d].sum()>cuts[1])
        assert int(degree_group(a.ps,a.pairs['train'][0,i:i+1],a.pairs['train'][1,i:i+1],a.model.degree_cuts))==expected
    return result


def main():
    records=[]
    for fold in range(3):
        data=load_inner(fold);seed_all(42);base=make_adapter(data,'unmasked10','cuda:0');base.model.eval()
        original={k:v.detach().clone() for k,v in base.model.state_dict().items()}
        with torch.no_grad():p0=base.predict('val')
        initial_experts=None
        for name in CONFIGS:
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.eval()
            for key,value in original.items():torch.testing.assert_close(value,a.model.state_dict()[key],atol=0,rtol=0)
            experts={k:v.detach().clone() for k,v in a.model.experts.state_dict().items()}
            if initial_experts is None:initial_experts=experts
            else:
                for k in experts:torch.testing.assert_close(experts[k],initial_experts[k],atol=0,rtol=0)
            with torch.no_grad():p1=a.predict('val')
            np.testing.assert_allclose(p0,p1,atol=2e-5,rtol=2e-5)
            counts=route_reference(a,data)
            opt=make_optimizer(a.model);gen=mask_generator('cuda:0',20261006+fold)
            for epoch in [1,2]:losses=step(a,opt,epoch,gen)
            assert int(a.model.eb.num_batches_tracked)==2
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
            assert all(any(p.grad is not None and p.grad.abs().max()>0 for p in expert.parameters()) for expert in a.model.experts)
            a.model.eval()
            with torch.no_grad():
                shared,residual,output,gate=a.model.components(a.pairs['val'],a.ps,a.qis['val'],a.ease['val'],a.dense)
                expected=sum(output[:,i]*gate[:,i] for i in range(4))
                torch.testing.assert_close(expected,residual,atol=1e-7,rtol=1e-6)
                torch.testing.assert_close(shared+expected,a.logits('val'),atol=2e-4,rtol=5e-5)
            deleted=deletion_check(a,data,name,count=4)
            seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0')
            b.model.load_state_dict(a.model.state_dict());b.model.eval();a.model.eval()
            with torch.no_grad():p,q=a.predict('val'),b.predict('val')
            np.testing.assert_allclose(p,q,atol=2e-5,rtol=2e-5)
            records.append(dict(config=name,fold=fold,initial_probability_error=float(abs(p0-p1).max()),counts=counts,
                explicit_deletions=deleted,heldout_flip_error=float(abs(p-q).max()),first_two_updates_finite=True))
            print('PASS',name,fold,flush=True);del a,b,opt,experts;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,
        query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    print('ALL PASSED',flush=True)


if __name__=='__main__':main()
