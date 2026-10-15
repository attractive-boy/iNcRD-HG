"""Frozen 6-by-3 training-internal auxiliary-transfer experiment."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from model import HERE,CONFIGS,sha256


def main():
    if (HERE/'STOPPED_BY_SCOPE.json').exists():
        raise RuntimeError('User prohibits lncRNA. This experiment is ineligible and must not resume.')
    out=HERE/'results';out.mkdir(exist_ok=True)
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,source_sha256=sha256(HERE/'train.py'),model_sha256=sha256(HERE/'model.py'),
        data_sha256=sha256(HERE/'inner_data.npz'),checkpoint_selection='mean(inner AUC, inner AP)',
        config_selection='minimum of paired mean inner AUC/AP gains over control',
        outer_promotion_minimum_both_gains=.002,outer_validation_evaluated=False,test_evaluated=False)
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for fold in range(3):
        for name in CONFIGS:
            path=out/f'{name}_f{fold}_s42.json'
            if path.exists():
                r=json.loads(path.read_text())
                assert r['source_sha256']==plan['source_sha256'] and r['model_sha256']==plan['model_sha256']
                assert r['checkpoint_sha256']==sha256(path.with_suffix('.pt'))
            else:q.put((name,fold))
    def worker(device):
        while True:
            try:name,fold=q.get_nowait()
            except Empty:return
            with (out/f'{name}_f{fold}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(fold),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,fold,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for f in [pool.submit(worker,d) for d in ('cuda:0','cuda:1')]:f.result()

if __name__=='__main__':main()
