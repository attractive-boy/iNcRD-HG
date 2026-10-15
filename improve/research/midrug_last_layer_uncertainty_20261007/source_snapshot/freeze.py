"""Freeze all source and preregistered readouts only after prechecks pass."""
import json,shutil,datetime
from common import HERE,CLEAN,RELEASE,sha256,controls,SOURCE_PLAN

def main():
    out=HERE/'results';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists()
    check=HERE/'initial_checks.json';assert json.loads(check.read_text())['status']=='passed'
    sources=list(HERE.glob('*.py'))+[HERE/'README.md',HERE.parent/'midrug_subgraph_20261006/common.py',
        HERE.parent/'midrug_masked_reconstruction_20261006/model.py',HERE.parent/'midrug_degree_encoder_20261006/check.py']+list((RELEASE/'incrd_hg_m123').glob('*.py'))
    evidence=[check,SOURCE_PLAN,HERE.parent/'midrug_curvature_literature_20261007/READING_NOTES.md',
        HERE.parent/'midrug_curvature_literature_20261007/review_manifest.json']
    p=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fits=3,folds=[0,1,2],seed=42,
        readouts=['point','laplace','sandwich','constant'],eligible_candidates=['sandwich'],
        comparators=['control','unmasked10','point','laplace','constant'],promotion_minimum_both_gains_over_each_control=.002,
        prior_precision=1.,penalized_intercept=True,dirichlet_alpha=1.,quadrature_points=64,
        convergence_gradient_inf=1e-9,maximum_newton_updates=100,checkpoint_selection='stationarity on training only',
        controls={k:controls()[k] for k in ['control','unmasked10']},
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (out/'plan.json').write_text(json.dumps(p,indent=2));snap=HERE/'source_snapshot';snap.mkdir()
    for s in HERE.glob('*.py'):shutil.copy2(s,snap/s.name)
    print(sha256(out/'plan.json'))

if __name__=='__main__':main()
