"""Verify nested folds against prior audited indices before fitting."""
import json
import numpy as np
from common import HERE,load_inner,nested_data,sha256

def main():
    prior=HERE.parent/'midrug_crossfit_20261006';v=json.loads((prior/'base_verification.json').read_text());assert v['status']=='passed'
    records=[]
    for fold in range(3):
        outer=load_inner(fold);counts=np.zeros(len(outer['train_y']),dtype=int)
        for k in range(-1,5):
            data,rows=nested_data(fold,k);p=prior/'bases'/f'base_f{fold}_k{k}.npz';r=json.loads(p.with_suffix('.json').read_text())
            assert sha256(p)==r['cache_sha256']
            with np.load(p) as z:
                np.testing.assert_array_equal(z['train_indices'],data['train_indices']);np.testing.assert_array_equal(z['prediction_indices'],data['val_indices'])
                np.testing.assert_array_equal(z['pairs'],data['val_pairs']);np.testing.assert_array_equal(z['prediction_rows'],rows)
            if k>=0:
                counts[rows]+=1
                assert not np.intersect1d(data['train_indices'],outer['val_indices']).size
                assert not np.intersect1d(data['val_indices'],outer['val_indices']).size
            records.append(dict(fold=fold,nested=k,n_train=len(data['train_y']),n_predict=len(rows),old_cache_sha256=sha256(p)))
        assert (counts==1).all()
    (HERE/'base_initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,oof_exactly_once=True,
        prior_verification_sha256=sha256(prior/'base_verification.json'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    print('PASS nested indices, disjointness and exactly-once coverage',flush=True)

if __name__=='__main__':main()
