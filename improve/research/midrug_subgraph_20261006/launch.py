"""Fixed four-configuration by three-fold subgraph grid."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from model import HERE,CLEAN,CONFIGS,sha256


def main():
    out=HERE/'results';out.mkdir(exist_ok=True)
    controls={}
    for f in range(3):
        p=CLEAN/'results'/f'control_f{f}_s42.json'
        r=json.loads(p.read_text());assert r['checkpoint_sha256']==sha256(p.with_suffix('.pt'))
        controls[str(f)]=dict(result_sha256=sha256(p),checkpoint_sha256=sha256(p.with_suffix('.pt')),result_path=str(p))
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,workers=['cuda:0','cuda:1','cuda:0','cuda:1'],
        source_hashes={name:sha256(HERE/name) for name in ('common.py','model.py','extract.py','train.py')},
        cache_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted((HERE/'cache').glob('*.npz'))},controls=controls,
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        checkpoint_selection='mean(inner AUC, inner AP)',config_selection='minimum of paired mean inner AUC/AP gains over strict M123 control',
        outer_promotion_minimum_both_gains=.002,lncrna_used=False,outer_validation_evaluated=False,test_evaluated=False)
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for fold in range(3):
        for name in CONFIGS:
            path=out/f'{name}_f{fold}_s42.json'
            if path.exists():
                r=json.loads(path.read_text());assert r['source_sha256']==plan['source_hashes']['train.py'] and r['model_sha256']==plan['source_hashes']['model.py']
                assert r['checkpoint_sha256']==sha256(path.with_suffix('.pt'))
            else:q.put((name,fold))
    def worker(device):
        while True:
            try:name,fold=q.get_nowait()
            except Empty:return
            with (out/f'{name}_f{fold}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(fold),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,fold,device,flush=True)
    with ThreadPoolExecutor(4) as pool:
        for f in [pool.submit(worker,d) for d in plan['workers']]:f.result()

if __name__=='__main__':main()
