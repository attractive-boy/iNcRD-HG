"""Freeze all proper-loss arms and implementation after correctness checks."""
import datetime,json,shutil
from common import HERE,CLEAN,STRICT,RECON,RELEASE,sha256
from model import CONFIGS

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    old=HERE.parent/'midrug_training_size_20261007/results/plan.json';controls=json.loads(old.read_text())['controls']
    source=list(HERE.glob('*.py'))+[HERE/'README.md',STRICT,RECON/'model.py',HERE.parent/'midrug_degree_encoder_20261006/check.py']+list((RELEASE/'incrd_hg_m123').glob('*.py'))
    evidence=[old,HERE/'initial_checks.json',HERE.parent/'midrug_loss_shape_literature_20261007/review_manifest.json',
        HERE.parent/'midrug_loss_shape_literature_20261007/READING_NOTES.md',HERE.parent/'midrug_conjugate_blocks_20261007/archive_manifest.json']
    p=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,folds=[0,1,2],seed=42,new_fits=9,
        maximum_epochs=800,minimum_epochs=300,patience=200,checkpoint_selection='earliest maximum mean(inner AUC,AP)',
        controls={n:controls[n] for n in ['control','unmasked10']},eligible_candidates=['negative_tail'],comparators=['control','unmasked10','positive_tail','brier'],
        promotion_minimum_both_gains_over_each_control=.002,source_hashes={str(f):sha256(f) for f in source},evidence_hashes={str(f):sha256(f) for f in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    out=HERE/'results';out.mkdir();(out/'plan.json').write_text(json.dumps(p,indent=2));snap=HERE/'source_snapshot';snap.mkdir()
    for f in HERE.glob('*.py'):shutil.copy2(f,snap/f.name)
    print(sha256(out/'plan.json'))

if __name__=='__main__':main()
