"""Extend the weight boundary only after all first-stage replays pass."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from common import HERE,PARENT,CLEAN,PREVIOUS,RELEASE,sha256
from model import CONFIGS

def main():
    check=json.loads((PARENT/'verification.json').read_text());assert check['status']=='passed' and len(check['records'])==15
    out=HERE/'results';out.mkdir(exist_ok=True)
    dependencies=[HERE/'common.py',HERE/'model.py',HERE/'train.py',PARENT/'common.py',PARENT/'model.py',PREVIOUS/'common.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,auxiliary_weight_warmup=50,
        optimizer='RMSprop lr=1e-4 alpha=.99; auxiliary decoder/token weight_decay=1e-4',mask_seed='20261006 + fold',
        motivation='First-stage weight=1 improves both metrics in all three inner folds and is best at the tested upper boundary; extend to 3/10 with matched unmasked controls.',
        previous_verification_sha256=sha256(PARENT/'verification.json'),source_hashes={str(p):sha256(p) for p in dependencies},
        checkpoint_selection='mean(inner AUC, inner AP)',config_selection='minimum of paired mean AUC/AP gains over optimized pure-miRNA M123 controls',
        promotion_minimum_both_gains=.002,graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        controls={str(f):dict(path=str(CLEAN/'results'/f'control_f{f}_s42.json'),checkpoint_sha256=sha256(CLEAN/'results'/f'control_f{f}_s42.pt')) for f in range(3)},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for f in range(3):
        for name in CONFIGS:
            if not (out/f'{name}_f{f}_s42.json').exists():q.put((f,name))
    def worker(device):
        while True:
            try:f,name=q.get_nowait()
            except Empty:return
            with (out/f'{name}_f{f}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(f),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',f,name,device,flush=True)
    with ThreadPoolExecutor(4) as pool:
        for f in [pool.submit(worker,d) for d in ['cuda:0','cuda:1','cuda:0','cuda:1']]:f.result()

if __name__=='__main__':main()
