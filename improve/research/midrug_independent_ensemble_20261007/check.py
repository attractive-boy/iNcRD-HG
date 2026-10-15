"""Validate seed independence, fixed aggregation and original query semantics."""
import json
import numpy as np
import torch
from common import HERE,RECON,read_module,load_inner,seed_all,sha256,make_adapter,mask_generator,member_seed,probability_mean
QUERY_CHECK_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
query=read_module('independent_ensemble_deletion',QUERY_CHECK_SOURCE);query.make_adapter=make_adapter
deletion_check=query.deletion_check

@torch.no_grad()
def batch_check(a):
    batched=a.logits('val')[:3]
    alone=torch.cat([a.model(a.pairs['val'][:,i:i+1],a.ps,a.qis['val'][i:i+1],a.ease['val'][i:i+1],a.dense) for i in range(3)])
    torch.testing.assert_close(batched,alone,atol=2e-4,rtol=5e-5)
    return float(abs(batched-alone).max())

def main():
    seeds={str(r):[member_seed(r,j) for j in range(8)] for r in [42,43,44]}
    assert len(set(s for row in seeds.values() for s in row))==24
    values=np.array([[.01*j,.12*j,.04*j] for j in range(8)],dtype=np.float32)
    expected=np.array([sum(float(values[j,i]) for j in range(8))/8 for i in range(3)])
    np.testing.assert_array_equal(probability_mean(values),expected)
    np.testing.assert_array_equal(probability_mean(values[:,[2,0]]),expected[[2,0]])
    np.testing.assert_array_equal(probability_mean(values[::-1]),expected)
    prior=read_module('independent_original_reference',RECON/'model.py');records=[]
    for fold in range(3):
        data=load_inner(fold);assert len(data['test_pairs'])==0
        seed_all(42);a=make_adapter(data,'unmasked10','cuda:0');a.model.eval()
        seed_all(42);b=prior.make_adapter(data,'unmasked10','cuda:0');b.model.eval()
        for k,v in a.model.state_dict().items():torch.testing.assert_close(v,b.model.state_dict()[k],atol=0,rtol=0)
        seed_all(member_seed(42,1));c=make_adapter(data,'unmasked10','cuda:0');c.model.eval()
        different=sum(not torch.equal(v,dict(c.model.named_parameters())[k]) for k,v in a.model.named_parameters())
        assert different>0
        ga=mask_generator('cuda:0',20261006+fold);gc=mask_generator('cuda:0',20261006+fold)
        for _ in range(3):
            ma=a.model.reconstruction.masks(a.graph,.5,ga);mc=c.model.reconstruction.masks(c.graph,.5,gc)
            for n in ma:torch.testing.assert_close(ma[n],mc[n],atol=0,rtol=0)
        seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},'unmasked10','cuda:0');b.model.eval()
        with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
        np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
        records.append(dict(fold=fold,different_parameter_tensors=different,private_sampler_matches=True,
            original_initial_parameters_equal=True,heldout_label_flip_error=float(abs(pa-pb).max()),
            batch_max_logit_error=batch_check(a),explicit_deletions=deletion_check(a,data,'unmasked10',count=4)))
        print('PASS',fold,flush=True)
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',seeds=seeds,records=records,
        float64_probability_mean_matches_scalar_sum=True,query_subset_invariant=True,member_permutation_invariant=True,
        query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
