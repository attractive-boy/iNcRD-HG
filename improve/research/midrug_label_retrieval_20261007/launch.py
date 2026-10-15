"""Freeze three label-retrieval arms and all controls before nine fits."""
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RELEASE,STRICT,RECON,sha256
from model import CONFIGS

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    previous=HERE.parent/'midrug_independent_ensemble_20261007';review=HERE.parent/'midrug_task_literature_20261007'
    assert json.loads((previous/'archive_recheck.json').read_text())['status']=='passed'
    sources=[HERE/n for n in ['common.py','model.py','train.py','check.py','launch.py']]
    sources+=[STRICT,RECON/'model.py',HERE.parent/'midrug_degree_encoder_20261006/check.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    evidence=[HERE/'README.md',HERE/'initial_checks.json',previous/'archive_recheck.json',previous/'REPORT.md',
        HERE.parent/'midrug_training_size_20261007/verification.json',HERE.parent/'midrug_training_size_20261007/results/plan.json',
        review/'READING_NOTES.md',review/'review_manifest.json',review/'raw/tabr.py',review/'raw/tabr_v2.html',
        HERE.parent.parent/'local_support_modules.py',HERE.parent.parent/'LOCAL_SUPPORT.md']
    controls=json.loads((HERE.parent/'midrug_training_size_20261007/results/plan.json').read_text())['controls']
    plan=dict(configs=CONFIGS,seed=42,folds=[0,1,2],new_fits=9,controls=controls,
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        width=64,neighbors=32,keys='raw original standardized attributes only; shared negative squared L2',
        values='learned class embedding + MLP(key_query-key_context); current query excluded',
        sampling_seed='20261006+fold',optimizer='RMSprop alpha=.99 lr1e-4 main wd0 auxiliary wd1e-4 set_to_none=True',
        reconstruction_weight=10.,reconstruction_warmup=50,maximum_epochs=800,minimum_epochs=300,patience=200,
        checkpoint_selection='Earliest maximum mean(inner AUC,inner AP)',
        comparators=['control','unmasked10','features','direct'],eligible_candidates=['label'],
        promotion_minimum_both_gains_over_each_control=.002,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    out=HERE/'results';out.mkdir(exist_ok=True);path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    jobs=Queue()
    for fold in range(3):
        for config in CONFIGS:
            if not (out/f'{config}_f{fold}_s42.json').exists():jobs.put((config,fold))
    def worker(device):
        while True:
            try:config,fold=jobs.get_nowait()
            except Empty:return
            with (out/f'{config}_f{fold}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',config,'--fold',str(fold),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',config,fold,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1']]:future.result()

if __name__=='__main__':main()
