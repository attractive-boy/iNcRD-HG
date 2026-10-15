"""Freeze the limited head adaptation protocol before nine new fits."""
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RECON,STRICT,RELEASE,sha256
from model import CONFIGS,SOURCE_PLAN
from check import PREVIOUS,QUERY_CHECK_SOURCE

def main():
    c=json.loads((HERE/'initial_checks.json').read_text());assert c['status']=='passed'
    for p,h in c['source_hashes'].items():assert sha256(p)==h
    controls=json.loads(SOURCE_PLAN.read_text())['controls']
    for old in ['full','mean']:
        controls['frozen_'+old]={}
        for fold in range(3):
            p=PREVIOUS/'results'/f'{old}_f{fold}_s42.json';r=json.loads(p.read_text())
            controls['frozen_'+old][str(fold)]=dict(path=str(p),metadata_sha256=sha256(p),checkpoint_sha256=r['checkpoint_sha256'],
                probability_sha256=sha256(p.with_name(p.stem+'_inner.npz')))
    sources=[HERE/p for p in ['common.py','modules.py','model.py','train.py','launch.py','check.py','README.md']]
    sources += [STRICT,RECON,QUERY_CHECK_SOURCE,PREVIOUS/'check.py']+sorted((RELEASE/'incrd_hg_m123').glob('*.py'))
    evidence=[HERE/'initial_checks.json',SOURCE_PLAN,HERE.parent/'midrug_training_size_20261007/verification.json']
    evidence += [PREVIOUS/p for p in ['archive_manifest.json','verification.json','math_checks.json','README.md','DERIVATION.md','selection.json']]
    evidence += [HERE.parent/p/'REPORT.md' for p in ['midrug_retrieval_warmstart_20261007','midrug_reconstruction_finetune_v2_20261007']]
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,new_fits=9,controls=controls,
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        maximum_epochs=300,minimum_epochs=100,patience=100,checkpoint_selection='earliest max mean(inner AUC,AP), including source stage0',
        optimizer='RMSprop alpha=.99 head lr1e-5 wd0; branch lr1e-4 wd1e-4; set_to_none=True; encoders and buffers frozen',
        comparators=['control','unmasked10','frozen_full','frozen_mean','mean_head','head_only'],eligible_candidates=['full_head'],
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
