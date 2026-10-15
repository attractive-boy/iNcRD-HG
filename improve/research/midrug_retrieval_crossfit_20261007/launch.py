"""Freeze only the predeclared case-offset comparison after base replay."""
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RETRIEVAL,sha256
from model import CONFIGS

def main():
    assert json.loads((HERE/'case_initial_checks.json').read_text())['status']=='passed'
    base=json.loads((HERE/'base_verification.json').read_text());assert base['status']=='passed' and base['new_checkpoint_replays']==18
    bp=json.loads((HERE/'bases/plan.json').read_text())
    sources={**bp['source_hashes'],**{str(p):sha256(p) for p in [HERE/'train.py',HERE/'launch.py',HERE/'check_case.py',HERE/'audit_base.py',RETRIEVAL/'check.py',HERE.parent/'midrug_degree_encoder_20261006/check.py']}}
    evidence={**bp['evidence_hashes'],**{str(p):sha256(p) for p in [HERE/'base_verification.json',HERE/'case_initial_checks.json',HERE/'bases/plan.json',HERE.parent/'midrug_training_size_20261007/verification.json',HERE.parent/'midrug_training_size_20261007/results/plan.json']}}
    for p in (HERE/'offsets').iterdir():evidence[str(p)]=sha256(p)
    controls=json.loads((HERE.parent/'midrug_training_size_20261007/results/plan.json').read_text())['controls']
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,new_base_fits=18,new_case_fits=9,controls=controls,
        source_hashes=sources,evidence_hashes=evidence,offsets=base['offsets'],
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        maximum_epochs=300,minimum_epochs=100,patience=100,checkpoint_selection='earliest max mean(inner AUC,AP), including full300 stage0',
        optimizer='case-only RMSprop alpha=.99 lr1e-4 wd1e-4 set_to_none=True',
        comparators=['control','unmasked10','full300','insample_label','oof_features'],eligible_candidates=['oof_label'],
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
