"""A dual certificate bounding AUC of fixed nonnegative score mixtures.

For ambiguous positive-negative comparisons, D[i,k] is model k's score margin.
R[i,k]=D[i,k]/max_k |D[i,k]|. For w in the probability simplex, the number of
misranked comparisons is at least sum_i max(0,-R[i]@w). For any a in [0,1]^m,
that sum is at least min_k -(R.T@a)[k]. This last expression is a certificate
for EVERY fixed mixture; it is not a fitted predictor and does not access test.

We optimize a via a small-row linear program, but certify any returned a after
clipping its bounds. Optimal solver status is not required for validity. We
subtract explicit coefficient and accumulation roundoff allowances. All-model
strict errors and all-model ties are counted separately, exactly as before.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from sklearn.metrics import roc_auc_score
from common import HERE,load_inner,sha256

COEFFICIENT_ALLOWANCE=1e-12

def build_margins(predictions,y,block=64):
    p=np.flatnonzero(y==1);n=np.flatnonzero(y==0);ambiguous=[];wrong=ties=0
    for start in range(0,len(p),block):
        # [models, positives-in-block, negatives] only, then retain mixed signs.
        D=predictions[:,p[start:start+block],None]-predictions[:,None,n]
        lo=D.min(0);hi=D.max(0);wrong+=int((hi<0).sum());ties+=int(((lo==0)&(hi==0)).sum())
        take=(lo<0)&(hi>=0)
        if take.any():
            M=np.maximum(abs(lo[take]),abs(hi[take]))
            ambiguous.append((D[:,take]/M[None]).T.copy())
    R=np.concatenate(ambiguous,axis=0) if ambiguous else np.zeros((0,len(predictions)))
    return R,dict(comparisons=int(len(p)*len(n)),shared_strict_errors=wrong,all_model_ties=ties)

def certify(R,time_limit=90):
    m,k=R.shape
    if not m:return dict(ambiguous_comparisons=0,certified_extra_error_mass=0.,solver_success=True),np.empty(0)
    assert m<10_000_000 and np.isfinite(R).all() and abs(R).max()<=1+1e-14
    # Max lambda subject to R.T@a + lambda <= 0, 0<=a<=1.
    # A constant allowance in the objective accommodates coefficient rounding.
    matrix=np.column_stack((R.T,np.ones(k)));objective=np.r_[np.full(m,COEFFICIENT_ALLOWANCE),-1.]
    started=time.time()
    result=linprog(objective,A_ub=matrix,b_ub=np.zeros(k),bounds=[(0.,1.)]*m+[(None,None)],
        method='highs',options={'time_limit':time_limit,'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    # Even a time-limited feasible iterate is useful. No iterate gives the
    # zero certificate; never substitute a primal objective for a lower bound.
    a=np.zeros(m) if result.x is None else np.clip(np.asarray(result.x[:m]),0.,1.)
    assert np.isfinite(a).all()
    dots=np.array([np.sum(R[:,j].astype(np.longdouble)*a.astype(np.longdouble),dtype=np.longdouble) for j in range(k)],dtype=np.longdouble)
    coefficient_allowance=COEFFICIENT_ALLOWANCE*float(a.sum())
    # Larger than standard floating-point sum bounds at the declared m limit.
    accumulation_allowance=1e-8*(m+1)
    lam=float(-dots.max())-accumulation_allowance
    bound=max(0.,lam-coefficient_allowance)
    return dict(ambiguous_comparisons=m,models=k,solver_success=bool(result.success),solver_status=int(result.status),
        solver_message=result.message,elapsed_seconds=time.time()-started,lambda_before_allowance=float(-dots.max()),
        coefficient_allowance=coefficient_allowance,accumulation_allowance=accumulation_allowance,
        certified_extra_error_mass=bound,nonzero_dual_weights=int((a>0).sum()),minimum_dual_weight=float(a.min()),maximum_dual_weight=float(a.max())),a

def self_check():
    predictions=np.array([[.1,1.,.01],[.1,.01,1.]],dtype=np.float64);y=np.array([1,0,0])
    R,counts=build_margins(predictions,y);r,a=certify(R,10)
    assert r['solver_success'] and abs(r['certified_extra_error_mass']-.9)<1e-6
    upper=1-(counts['shared_strict_errors']+.5*counts['all_model_ties']+r['certified_extra_error_mass'])/counts['comparisons']
    scores=[roc_auc_score(y,w*predictions[0]+(1-w)*predictions[1]) for w in np.linspace(0,1,1001)]
    assert max(scores)<=upper and .54<upper<.56
    return dict(status='passed',known_extra_error_lower_bound=.9,numerical_lower_bound=r['certified_extra_error_mass'],
        grid_best_auc=max(scores),certified_auc_upper_bound=upper)

def main():
    check=self_check();prior=json.loads((HERE/'shared_error_bound.json').read_text());records=[]
    for fold in range(3):
        data=load_inner(fold);paths=[Path(p) for p in prior['prediction_hashes'] if f'_f{fold}_s42_inner.npz' in p]
        assert len(paths)==prior['records'][fold]['models']
        scores=[]
        for p in paths:
            assert sha256(p)==prior['prediction_hashes'][str(p)]
            with np.load(p) as z:
                for key,dk in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[dk])
                scores.append(z['probability'].astype(np.float64))
        scores=np.stack(scores);R,counts=build_margins(scores,data['val_y'])
        assert counts['shared_strict_errors']==prior['records'][fold]['shared_strict_errors']
        assert counts['all_model_ties']==prior['records'][fold]['all_model_ties']
        proof,a=certify(R)
        upper=1-(counts['shared_strict_errors']+.5*counts['all_model_ties']+proof['certified_extra_error_mass'])/counts['comparisons']
        best=max(roc_auc_score(data['val_y'],p) for p in scores)
        assert best<=upper+1e-12 and upper<=prior['records'][fold]['auc_upper_bound']+1e-12
        path=HERE/f'mixture_dual_f{fold}.npz';np.savez_compressed(path,a=a)
        record=dict(fold=fold,**counts,**proof,certified_auc_upper_bound=float(upper),best_individual_auc=float(best),
            unanimous_only_upper_bound=prior['records'][fold]['auc_upper_bound'],certificate_sha256=sha256(path))
        records.append(record);print(json.dumps(record),flush=True)
    out=dict(scope='Only fixed nonnegative weighted mixtures of the 56 saved original-training internal prediction vectors per fold. Not a new-model, conditional-gating, logit-trained-model, or original-test performance limit.',
        self_check=check,records=records,mean_certified_auc_upper_bound=float(np.mean([r['certified_auc_upper_bound'] for r in records])),
        prior_bound_sha256=sha256(HERE/'shared_error_bound.json'),source_sha256=sha256(HERE/'mixture_certificate.py'),
        no_model_trained=True,used_for_selection=False,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'mixture_certificate.json').write_text(json.dumps(out,indent=2));print('MEAN UPPER BOUND',out['mean_certified_auc_upper_bound'],flush=True)

if __name__=='__main__':main()
