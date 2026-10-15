import json,datetime,shutil
from common import HERE,CLEAN,STRICT,RECON,RELEASE,sha256
from model import PARAMS,CONFIGS,REFERENCE

def main():
 assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
 inputs=json.loads((HERE/'input_manifest.json').read_text());old=json.loads(REFERENCE.read_text())
 sources=list(HERE.glob('*.py'))+[HERE/'README.md',STRICT,RECON/'model.py']+list((RELEASE/'incrd_hg_m123').glob('*.py'))
 review=HERE.parent/'midrug_decoupling_review_20261007';prior=HERE.parent/'midrug_raw_residual_20261007'
 evidence=[REFERENCE,HERE/'initial_checks.json',HERE/'input_manifest.json',review/'REPORT.md',review/'archive_manifest.json',prior/'precheck.log',prior/'model.py',prior/'check.py',prior/'input_manifest.json']
 ih={**inputs['artifact_hashes'],**inputs['source_checkpoints'],str(CLEAN/'graph.pt'):inputs['graph_sha256'],str(CLEAN/'inner_data.npz'):inputs['data_sha256']}
 for f,h in ih.items():assert sha256(f)==h
 plan=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,params=PARAMS,folds=[0,1,2],seed=42,new_fits=9,
  maximum_rounds=1600,minimum_rounds=300,patience=150,stage0_included=True,checkpoint_selection='earliest maximum mean(inner AUC,AP)',
  controls=old['controls'],eligible_candidates=['raw'],comparators=['control','unmasked10','pca32','structure'],promotion_minimum_both_gains_over_each_control=.002,
  source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},input_hashes=ih,
  no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 out=HERE/'results';out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));snap=HERE/'source_snapshot';snap.mkdir()
 for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
 print(sha256(out/'plan.json'))
if __name__=='__main__':main()
