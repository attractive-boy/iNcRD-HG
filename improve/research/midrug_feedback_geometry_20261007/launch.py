"""Freeze the nine-fit protocol only after all initial checks pass."""
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RECON,STRICT,RELEASE,sha256
from model import CONFIGS,SOURCE_PLAN
from check import QUERY_CHECK_SOURCE

def main():
    for file in ['initial_checks.json','math_checks.json']:
        c=json.loads((HERE/file).read_text());assert c['status']=='passed'
        for path,h in c['source_hashes'].items():assert sha256(path)==h
    previous=HERE.parent/'midrug_training_size_20261007'
    controls=json.loads(SOURCE_PLAN.read_text())['controls']
    sources=[HERE/p for p in ['common.py','modules.py','model.py','train.py','launch.py','check.py','check_math.py','README.md']]
    sources += [STRICT,RECON,QUERY_CHECK_SOURCE]
    sources += sorted((RELEASE/'incrd_hg_m123').glob('*.py'))
    evidence=[HERE/'initial_checks.json',HERE/'math_checks.json',previous/'verification.json',SOURCE_PLAN,
        HERE.parent/'midrug_post_crossfit_literature_20261007/READING_NOTES.md',
        HERE.parent/'midrug_literature_20261006/raw/expression_geof.html']
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,new_fits=9,controls=controls,
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        maximum_epochs=300,minimum_epochs=100,patience=100,checkpoint_selection='earliest max mean(inner AUC,AP), including selected source stage0',
        optimizer='feedback-only RMSprop alpha=.99 lr1e-4 wd1e-4 set_to_none=True',
        comparators=['control','unmasked10','diagonal','mean'],eligible_candidates=['full'],
        promotion_minimum_both_gains_over_each_control=.002,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
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
