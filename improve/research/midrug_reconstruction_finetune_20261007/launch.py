"""Freeze nine matched weight-initialized fine-tuning fits."""
import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RECON,RELEASE,STRICT,sha256
from model import CONFIGS,LR,SOURCE_PLAN,source_record
STUDY=HERE.parent/'midrug_training_size_20261007'
PRECHECK=HERE.parent/'midrug_reconstruction_gradient_20261007'


def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    assert json.loads((STUDY/'archive_recheck.json').read_text())['status']=='passed'
    prior=json.loads((PRECHECK/'archive_manifest.json').read_text())
    for path,digest in {**prior['artifact_hashes'],**prior['frozen_dependencies']}.items():assert sha256(path)==digest
    controls=json.loads(SOURCE_PLAN.read_text())['controls']
    for fold in range(3):source_record(fold)
    sources=[HERE/n for n in ['common.py','model.py','train.py','check.py','launch.py']]
    sources += [STRICT,RECON/'model.py',RECON/'phase2/train.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    evidence=[HERE/'README.md',HERE/'initial_checks.json',SOURCE_PLAN,STUDY/'archive_recheck.json',STUDY/'verification.json',PRECHECK/'archive_manifest.json',PRECHECK/'gradient_precheck.json',PRECHECK/'PRECHECK_REVIEW.md']
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,new_fits=9,controls=controls,
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        initialization='Same selected unmasked10 checkpoint within each original-training fold; no optimizer state restored',
        optimizer=dict(name='RMSprop',lr=LR,alpha=.99,main_weight_decay=0.,decoder_weight_decay=1e-4,initial_state='empty'),
        max_new_epochs=300,minimum_new_epochs=100,patience=100,include_source_epoch_zero=True,
        checkpoint_selection='earliest maximum mean(inner AUC,inner AP), including source checkpoint',
        eligible_candidates=['off','decay'],comparators={'off':['control','unmasked10','keep'],'decay':['control','unmasked10','keep']},
        promotion_minimum_both_gains_over_each_control=.002,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    out=HERE/'results';out.mkdir(exist_ok=True);path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    jobs=Queue()
    for fold in range(3):
        for name in CONFIGS:
            if not (out/f'{name}_f{fold}_s42.json').exists():jobs.put((name,fold))
    def worker(device):
        while True:
            try:name,fold=jobs.get_nowait()
            except Empty:return
            with (out/f'{name}_f{fold}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(fold),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,fold,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1']]:future.result()

if __name__=='__main__':main()
