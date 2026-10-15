"""Independently replay saved dual weights, without calling the LP or its builder."""
import json
from pathlib import Path
import numpy as np
from common import HERE,load_inner,sha256

def main():
    proof=json.loads((HERE/'mixture_certificate.json').read_text())
    prior=json.loads((HERE/'shared_error_bound.json').read_text())
    assert sha256(HERE/'mixture_certificate.py')==proof['source_sha256']
    assert sha256(HERE/'shared_error_bound.json')==proof['prior_bound_sha256']
    records=[]
    for rec in proof['records']:
        fold=rec['fold'];data=load_inner(fold);paths=[Path(p) for p in prior['prediction_hashes'] if f'_f{fold}_s42_inner.npz' in p]
        scores=[]
        for p in paths:
            assert sha256(p)==prior['prediction_hashes'][str(p)]
            with np.load(p) as z:
                np.testing.assert_array_equal(z['y'],data['val_y']);np.testing.assert_array_equal(z['pairs'],data['val_pairs'])
                scores.append(z['probability'].astype(np.float64))
        scores=np.stack(scores);assert np.isfinite(scores).all() and scores.min()>=0 and scores.max()<=1
        path=HERE/f'mixture_dual_f{fold}.npz';assert sha256(path)==rec['certificate_sha256']
        with np.load(path) as z:a=z['a'].copy()
        assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
        positive=np.flatnonzero(data['val_y']==1);negative=np.flatnonzero(data['val_y']==0)
        dots=np.zeros(len(scores),dtype=np.longdouble);cursor=wrong=ties=0
        for pos in positive:
            difference=scores[:,pos:pos+1]-scores[:,negative]
            minimum=difference.min(0);maximum=difference.max(0)
            wrong+=int((maximum<0).sum());ties+=int(((maximum==0)&(minimum==0)).sum())
            chosen=np.flatnonzero((minimum<0)&(maximum>=0));count=len(chosen)
            if count:
                scale=np.max(abs(difference[:,chosen]),axis=0)
                coefficients=(difference[:,chosen]/scale).astype(np.longdouble)
                dots+=(coefficients*a[cursor:cursor+count].astype(np.longdouble)).sum(1,dtype=np.longdouble)
                cursor+=count
        assert cursor==len(a)==rec['ambiguous_comparisons']
        assert wrong==rec['shared_strict_errors'] and ties==rec['all_model_ties']
        lam=float(-dots.max());assert abs(lam-rec['lambda_before_allowance'])<1e-8
        conservative=max(0.,lam-1e-12*float(a.sum())-1e-8*(len(a)+1))
        assert abs(conservative-rec['certified_extra_error_mass'])<1e-8
        upper=1-(wrong+.5*ties+conservative)/(len(positive)*len(negative))
        assert abs(upper-rec['certified_auc_upper_bound'])<1e-12
        records.append(dict(fold=fold,status='passed',dual_lambda_error=abs(lam-rec['lambda_before_allowance']),
            certified_auc_upper_bound=upper,ambiguous_comparisons=cursor))
    result=dict(status='passed',records=records,certificate_json_sha256=sha256(HERE/'mixture_certificate.json'),
        verifier_sha256=sha256(HERE/'verify_mixture_certificate.py'),no_optimizer_called=True,
        scope='Only the original saved probability vectors, with one fixed nonnegative weight vector per fold; excludes query-dependent gates, transformations to logits/ranks, newly trained models, and original-test claims.')
    (HERE/'mixture_certificate_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
