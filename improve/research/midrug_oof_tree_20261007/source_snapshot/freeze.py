import json,datetime,shutil
from common import HERE,PREV,BASES,CLEAN,STRICT,RECON,RELEASE,sha256
from model import CONFIGS,PARAMS

def main():
 assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
 inp=json.loads((HERE/'input_manifest.json').read_text());assert inp['status']=='passed' and inp['current_base_checkpoint_replays']==18
 old=json.loads((PREV/'results/plan.json').read_text())
 sources=list(HERE.glob('*.py'))+[HERE/'README.md',STRICT,RECON/'model.py',PREV/'model.py',PREV/'audit_correction_v1/oracle.py']+list((RELEASE/'incrd_hg_m123').glob('*.py'))
 evidence=[HERE/'initial_checks.json',HERE/'input_manifest.json',PREV/'results/plan.json',PREV/'selection.json',PREV/'archive_recheck.json',BASES/'README.md',BASES/'REPORT.md',BASES/'bases/plan.json',BASES/'base_verification.json',BASES/'archive_recheck.json']
 extras={n:{} for n in ['full300','raw_selected']}
 for f in range(3):
  for n,p in [('full300',BASES/'offsets'/f'full300_f{f}.json'),('raw_selected',PREV/'results'/f'raw_f{f}_s42.json')]:extras[n][str(f)]=dict(path=str(p),sha256=sha256(p))
  r=json.loads((PREV/'results'/f'raw_f{f}_s42.json').read_text())
  for file in r['artifact_hashes']:evidence.append(PREV/'results'/file)
 ih={**inp['source_hashes'],**inp['input_hashes']}
 for p,h in ih.items():assert sha256(p)==h
 plan=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,params=PARAMS,folds=[0,1,2],seed=42,new_tree_fits=9,new_base_fits=0,current_base_replays=18,
  maximum_rounds=1600,minimum_rounds=300,patience=150,stage0_included=True,checkpoint_selection='earliest maximum mean(inner AUC,AP)',controls=old['controls'],extra_controls=extras,
  eligible_candidates=['oof_raw'],comparators=['control','unmasked10','full300','raw_selected','insample_raw','oof_pca32'],promotion_minimum_both_gains_over_each_control=.002,
  source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},input_hashes=ih,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 out=HERE/'results';out.mkdir();(out/'plan.json').write_text(json.dumps(plan,indent=2));snap=HERE/'source_snapshot';snap.mkdir()
 for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
 print(sha256(out/'plan.json'))
if __name__=='__main__':main()
