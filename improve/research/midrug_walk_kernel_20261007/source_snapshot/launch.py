"""Freeze all inputs and five matched arms before fitting."""
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RECON,STRICT,RELEASE,sha256
from model import CONFIGS,SOURCE_PLAN,OLD,EXTRACT_SOURCE
from check import QUERY_CHECK_SOURCE

def main():
    for name in ['initial_checks.json','math_checks.json']:
        c=json.loads((HERE/name).read_text());assert c['status']=='passed'
        for p,h in c['source_hashes'].items():assert sha256(p)==h
    sources=[HERE/p for p in ['common.py','model.py','train.py','launch.py','check.py','check_math.py','README.md']]
    sources += [STRICT,RECON,QUERY_CHECK_SOURCE,OLD/'model.py',EXTRACT_SOURCE]+sorted((RELEASE/'incrd_hg_m123').glob('*.py'))
    evidence=[HERE/'initial_checks.json',HERE/'math_checks.json',SOURCE_PLAN,HERE.parent/'midrug_training_size_20261007/verification.json']
    evidence += [OLD/'cache'/f'f{f}_h1_k16{ext}' for f in range(3) for ext in ['.json','.npz']]
    evidence += [HERE.parent/'midrug_post_head_literature_20261007'/p for p in ['READING_NOTES.md','review_manifest.json','raw/mose.html','applicability_checks.json']]
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,new_fits=15,controls=json.loads(SOURCE_PLAN.read_text())['controls'],
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        maximum_epochs=300,minimum_epochs=100,patience=100,checkpoint_selection='earliest max mean(inner AUC,AP), including selected source stage0',
        optimizer='subgraph-only RMSprop lr1e-4 alpha=.99 wd1e-4 set_to_none=True; frozen M123 and all original buffers',
        comparators=['control','unmasked10','endpoints','fixed','no_edges','mpnn'],eligible_candidates=['pathwise'],
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
