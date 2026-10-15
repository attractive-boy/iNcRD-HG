"""Freeze all choices and dependencies before any inner score."""
import datetime,json,shutil
from common import HERE,CLEAN,STRICT,RECON,RELEASE,sha256
from model import CONFIGS

def main():
 assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
 provenance=json.loads((HERE/'input_manifest.json').read_text());assert provenance['status']=='passed'
 old=HERE.parent/'midrug_training_size_20261007/results/plan.json';controls=json.loads(old.read_text())['controls']
 sources=list(HERE.glob('*.py'))+[HERE/'README.md',STRICT,RECON/'model.py',HERE.parent/'midrug_degree_encoder_20261006/check.py']+list((RELEASE/'incrd_hg_m123').glob('*.py'))
 evidence=[old,HERE/'initial_checks.json',HERE/'READING_NOTES.md',HERE/'input_manifest.json',HERE/'predecessor_review.json']+list((HERE/'literature').glob('*'))
 inputs={**provenance['source_hashes'],str(HERE/'final_tokens.pt'):provenance['final_tokens_sha256'],str(CLEAN/'graph.pt'):sha256(CLEAN/'graph.pt'),str(CLEAN/'inner_data.npz'):sha256(CLEAN/'inner_data.npz')}
 for f,h in inputs.items():assert sha256(f)==h
 plan=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,folds=[0,1,2],seed=42,new_fits=9,
  maximum_epochs=800,minimum_epochs=300,patience=200,checkpoint_selection='earliest maximum mean(inner AUC,AP)',
  controls=controls,eligible_candidates=['mirna'],comparators=['control','unmasked10','population','pooled'],promotion_minimum_both_gains_over_each_control=.002,
  source_hashes={str(f):sha256(f) for f in sources},evidence_hashes={str(f):sha256(f) for f in evidence},input_hashes=inputs,
  token_inclusion='Original final-layer states, original512 truncation; exclude all special IDs and padding from pooling, retain original contextualization',
  optimizer='RMSprop alpha.99 eps1e-8 momentum0 centeredFalse; main lr1e-4 wd0, branch lr3e-4 wd1e-4, reconstruction lr1e-4 wd1e-4',
  reconstruction='unmasked original feature reconstruction10 ramp50, private mask generator20261006+fold',no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 out=HERE/'results';out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));snap=HERE/'source_snapshot';snap.mkdir()
 for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
 print('FROZEN',sha256(out/'plan.json'))
if __name__=='__main__':main()
