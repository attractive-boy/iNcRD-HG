"""Freeze nine expert-routing fits with fixed matched controls."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RELEASE,RECON,STRICT,sha256
from model import CONFIGS


def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    previous=HERE.parent/'midrug_weight_average_20261007';assert json.loads((previous/'archive_recheck.json').read_text())['status']=='passed'
    precheck=json.loads((HERE/'precheck.json').read_text());assert precheck['status']=='passed'
    for p,h in precheck['source_hashes'].items():assert sha256(p)==h
    controls=json.loads((previous/'results/plan.json').read_text())['controls']
    sources=[HERE/n for n in ['common.py','model.py','train.py','check.py','launch.py']]
    sources+=[STRICT,RECON/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    evidence=[HERE/n for n in ['README.md','initial_checks.json','PRECHECK_PLAN.md','precheck.json']]
    evidence+=[previous/'archive_recheck.json',previous/'verification.json',previous/'results/plan.json',HERE.parent/'midrug_training_size_20261007/verification.json']
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,new_fits=9,controls=controls,
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        expert_width=64,experts=4,expert_initial_output='zero',expert_lr=1e-3,expert_weight_decay=1e-4,
        main_lr=1e-4,reconstruction_lr=1e-4,reconstruction_weight_decay=1e-4,optimizer='RMSprop alpha=.99 set_to_none=True',
        auxiliary_weight=10.,auxiliary_warmup=50,maximum_epochs=800,minimum_epochs=300,patience=200,
        checkpoint_selection='Earliest maximum mean(inner AUC,inner AP)',
        eligible_candidates=['degree'],comparators={'degree':['control','unmasked10','random','uniform']},
        promotion_minimum_both_gains_over_each_control=.002,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    out=HERE/'results';out.mkdir(exist_ok=True);path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    jobs=Queue()
    for f in range(3):
        for n in CONFIGS:
            if not (out/f'{n}_f{f}_s42.json').exists():jobs.put((n,f))
    def worker(device):
        while True:
            try:name,fold=jobs.get_nowait()
            except Empty:return
            with (out/f'{name}_f{fold}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(fold),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,fold,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker,device) for device in ['cuda:0','cuda:1']]:future.result()


if __name__=='__main__':main()
