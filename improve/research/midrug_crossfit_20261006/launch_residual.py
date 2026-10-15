"""Frozen nested-residual comparison; never read original validation/test labels."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from common import HERE,PREVIOUS,SG,sha256
from residual import CONFIGS,load_context

def main():
    out=HERE/'results';out.mkdir(exist_ok=True)
    assert json.loads((HERE/'base_verification.json').read_text())['status']=='passed'
    plan=dict(configs=CONFIGS,subgraph=SG,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,optimizer='AdamW lr=0.001 weight_decay=0.001',
        checkpoint_selection='mean(inner AUC, inner AP)',config_selection='minimum of three-fold mean AUC/AP gains over fixed-300 base',
        promotion_minimum_both_gains=.002,also_require_both_gains_over_previous_optimized_control=.002,
        source_hashes={str(p):sha256(p) for p in [HERE/'common.py',HERE/'residual.py',PREVIOUS/'common.py',PREVIOUS/'model.py',PREVIOUS/'extract.py']},
        base_verification_sha256=sha256(HERE/'base_verification.json'),base_plan_sha256=sha256(HERE/'bases/plan.json'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for f in range(3):
        load_context(f,'cpu')
        for name in CONFIGS:
            if not (out/f'{name}_f{f}_s42.json').exists():q.put((f,name))
    def worker(device):
        while True:
            try:f,name=q.get_nowait()
            except Empty:return
            with (out/f'{name}_f{f}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'residual.py'),'--config',name,'--fold',str(f),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',f,name,device,flush=True)
    with ThreadPoolExecutor(4) as pool:
        for f in [pool.submit(worker,d) for d in ['cuda:0','cuda:1','cuda:0','cuda:1']]:f.result()

if __name__=='__main__':main()
