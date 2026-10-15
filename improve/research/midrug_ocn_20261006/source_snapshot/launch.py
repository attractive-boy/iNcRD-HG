"""Fixed four-configuration, three-inner-fold grid; four local processes."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from common import HERE,CLEAN,sha256
from features import CONFIGS

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    out=HERE/'results';out.mkdir(exist_ok=True)
    controls={str(f):dict(path=str(CLEAN/'results'/f'control_f{f}_s42.json'),checkpoint_sha256=sha256(CLEAN/'results'/f'control_f{f}_s42.pt')) for f in range(3)}
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,controls=controls,
        hashes={name:sha256(HERE/name) for name in ('common.py','features.py','model.py','train.py')},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        cache_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted((HERE/'cache').glob('*.npz'))},
        checkpoint_selection='mean(inner AUC, inner AP)',config_selection='minimum of mean paired gains in AUC/AP over pure-miRNA M123 controls',
        promotion_minimum_both_gains=.002,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for f in range(3):
        for name in CONFIGS:
            p=out/f'{name}_f{f}_s42.json'
            if p.exists():
                r=json.loads(p.read_text());assert r['hashes']==plan['hashes'] and r['checkpoint_sha256']==sha256(p.with_suffix('.pt'))
            else:q.put((name,f))
    def worker(device):
        while True:
            try:name,f=q.get_nowait()
            except Empty:return
            with (out/f'{name}_f{f}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(f),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,f,device,flush=True)
    with ThreadPoolExecutor(4) as pool:
        for f in [pool.submit(worker,d) for d in ['cuda:0','cuda:1','cuda:0','cuda:1']]:f.result()

if __name__=='__main__':main()
