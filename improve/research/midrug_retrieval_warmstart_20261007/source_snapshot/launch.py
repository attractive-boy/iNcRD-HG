"""Freeze nine warm-start comparisons and their immutable source controls."""
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RELEASE,STRICT,RECON,PREVIOUS,sha256
from model import CONFIGS

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    assert json.loads((PREVIOUS/'archive_recheck.json').read_text())['status']=='passed'
    review=HERE.parent/'midrug_task_literature_20261007';finetune=HERE.parent/'midrug_reconstruction_finetune_v2_20261007'
    sources=[HERE/n for n in ['common.py','model.py','train.py','check.py','launch.py']]
    sources+=[STRICT,RECON/'model.py',PREVIOUS/'model.py',PREVIOUS/'check.py',HERE.parent/'midrug_degree_encoder_20261006/check.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    evidence=[HERE/'README.md',HERE/'initial_checks.json',PREVIOUS/'archive_recheck.json',PREVIOUS/'REPORT.md',
        HERE.parent/'midrug_training_size_20261007/verification.json',HERE.parent/'midrug_training_size_20261007/results/plan.json',
        review/'RETRIEVAL_FOLLOWUP.md',review/'retrieval_interventions.json',review/'retrieval_followup_manifest.json',
        finetune/'README.md',finetune/'train.py',finetune/'model.py',finetune/'REPORT.md']
    controls=json.loads((HERE.parent/'midrug_training_size_20261007/results/plan.json').read_text())['controls']
    plan=dict(configs=CONFIGS,seed=42,folds=[0,1,2],new_fits=9,controls=controls,
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        source='Original unmasked10 selected checkpoints; restore all original weights/buffers; new zero-output retrieval branch',
        case_lr=1e-4,main_lr=1e-5,main_weight_decay=0.,auxiliary_weight_decay=1e-4,
        optimizer='Empty-state RMSprop alpha=.99 set_to_none=True; frozen parameters excluded',
        reconstruction_weight=10.,sampling_seed='20261006+fold advanced through source best_epoch; identical across arms',
        maximum_epochs=300,minimum_epochs=100,patience=100,checkpoint_selection='Earliest maximum mean(inner AUC,inner AP), including source stage0',
        comparators=['control','unmasked10','joint_label','frozen_features'],eligible_candidates=['frozen_label'],
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
