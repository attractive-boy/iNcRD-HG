"""Freeze 18 fixed-epoch unmasked10 base fits and the downstream protocol."""
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RELEASE,STRICT,RECON,RETRIEVAL,sha256

def main():
    assert json.loads((HERE/'base_initial_checks.json').read_text())['status']=='passed'
    old=HERE.parent/'midrug_crossfit_20261006';warm=HERE.parent/'midrug_retrieval_warmstart_20261007'
    sources=[HERE/n for n in ['common.py','model.py','base.py','check_base.py','launch_base.py']]
    sources+=[STRICT,RECON/'model.py',RETRIEVAL/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    evidence=[HERE/'README.md',HERE/'base_initial_checks.json',old/'README.md',old/'REPORT.md',old/'base_verification.json',
        old/'common.py',old/'base.py',warm/'archive_recheck.json',warm/'REPORT.md',
        HERE.parent/'midrug_task_literature_20261007/WARMSTART_FOLLOWUP.md']
    plan=dict(folds=[0,1,2],nested=[-1,0,1,2,3,4],seed=42,epochs=300,new_fits=18,
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        base_selection='fixed300; no prediction-fold metrics or early stop',
        downstream_protocol_sha256=sha256(HERE/'README.md'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    out=HERE/'bases';out.mkdir(exist_ok=True);path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    jobs=Queue()
    for fold in range(3):
        for nested in range(-1,5):
            if not (out/f'base_f{fold}_k{nested}.json').exists():jobs.put((fold,nested))
    def worker(device):
        while True:
            try:fold,nested=jobs.get_nowait()
            except Empty:return
            with (out/f'base_f{fold}_k{nested}.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'base.py'),'--fold',str(fold),'--nested',str(nested),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',fold,nested,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1']]:future.result()

if __name__=='__main__':main()
