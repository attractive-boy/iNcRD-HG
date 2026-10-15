"""Freeze six encoder configurations and matched controls before training."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from common import HERE,CLEAN,RECON,RELEASE,STRICT,sha256
from model import CONFIGS


def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    out=HERE/'results';out.mkdir(exist_ok=True)
    controls={}
    for name,root in [('control',CLEAN),('unmasked10',RECON/'phase2')]:
        controls[name]={str(f):dict(path=str(root/'results'/f'{name}_f{f}_s42.json'),
            checkpoint_sha256=sha256(root/'results'/f'{name}_f{f}_s42.pt')) for f in range(3)}
    sources=[HERE/'common.py',HERE/'model.py',HERE/'train.py',HERE/'launch.py',STRICT,
             RECON/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,controls=controls,
        source_hashes={str(p):sha256(p) for p in sources},graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        optimizer='RMSprop lr=1e-4 alpha=.99; reconstruction-only weight_decay=1e-4',
        reconstruction='unmasked, 50% targets, original generator seeds, 50-epoch weight ramp',
        checkpoint_selection='mean(inner AUC, inner AP)',
        config_selection='minimum paired mean AUC/AP gain over BOTH pure M123 and unmasked10',
        promotion_minimum_both_gains_over_each_control=.002,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for fold in range(3):
        for name in CONFIGS:
            if not (out/f'{name}_f{fold}_s42.json').exists():q.put((name,fold))
    def worker(device):
        while True:
            try:name,fold=q.get_nowait()
            except Empty:return
            with (out/f'{name}_f{fold}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(fold),'--device',device],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,fold,device,flush=True)
    with ThreadPoolExecutor(4) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1','cuda:0','cuda:1']]:future.result()


if __name__=='__main__':main()
