"""Freeze the complete finite protocol after training-only correctness checks."""
import datetime,json,shutil
from common import HERE,CLEAN,GROUPS,RELEASE,SOURCE_PLAN,sha256,write,controls

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    out=HERE/'results';out.mkdir();snap=HERE/'source_snapshot';snap.mkdir()
    sources=list(HERE.glob('*.py'))+[HERE/'README.md']
    prior=HERE.parent/'new_mechanisms_20260915'
    evidence=[HERE/'initial_checks.json',HERE/'READING_NOTES.md',SOURCE_PLAN,CLEAN/'manifest.json',CLEAN/'inner_data.npz',CLEAN/'graph.pt',
        GROUPS/'groups.npz',GROUPS/'groups.json',GROUPS/'REPORT.md',prior/'tgsbm_export.pdf',prior/'tgsbm_full.txt',
        prior/'alternate_source_manifest.json',prior/'READING_NOTES.md',HERE.parent/'midrug_entity_mixup_20261007/archive_manifest.json']
    p=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),folds=[0,1,2],source_neural_seed=42,
        iterative_fits=6,closed_form_fits=6,starts=['feature','shuffled'],chosen_by='training ELBO; feature wins exact tie',
        eligible_candidates=['learned'],comparators=['control','unmasked10','fixed_feature','fixed_shuffled'],
        readouts=['learned','fixed_feature','fixed_shuffled'],membership_groups=[16,8],beta_prior=[.5,.5],dirichlet_per_category=1.,
        initial_one_hot_mass=.95,fusion_weight=.1,max_sweeps=2000,membership_tolerance=1e-8,relative_elbo_tolerance=1e-10,
        stable_sweeps=3,promotion_minimum_both_gains_over_each_control=.002,
        controls={k:controls()[k] for k in ['control','unmasked10']},source_hashes={str(f):sha256(f) for f in sources},
        evidence_hashes={str(f):sha256(f) for f in evidence},no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    write(out/'plan.json',p)
    for f in HERE.glob('*.py'):shutil.copy2(f,snap/f.name)
    print('Plan frozen',sha256(out/'plan.json'),flush=True)

if __name__=='__main__':main()

