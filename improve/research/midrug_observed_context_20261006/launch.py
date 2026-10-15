"""Freeze the matched observed-context experiment before training."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from common import HERE,CLEAN,RECON,RELEASE,STRICT,sha256
from model import CONFIGS


def main():
    checks=json.loads((HERE/'initial_checks.json').read_text());assert checks['status']=='passed'
    previous=HERE.parent/'midrug_global_features_20261006/selection.json'
    assert not json.loads(previous.read_text())['promote_to_outer_validation']
    out=HERE/'results';out.mkdir(exist_ok=True);controls={}
    for name,root in [('control',CLEAN),('unmasked10',RECON/'phase2')]:
        controls[name]={str(f):dict(path=str(root/'results'/f'{name}_f{f}_s42.json'),
            checkpoint_sha256=sha256(root/'results'/f'{name}_f{f}_s42.pt')) for f in range(3)}
    comparators={name:['control','unmasked10']+([name.replace('zero_','positive_')] if cfg['support']==0 else []) for name,cfg in CONFIGS.items()}
    sources=[HERE/f for f in ['common.py','model.py','train.py','launch.py']]
    sources += [STRICT,RECON/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,controls=controls,
        comparators=comparators,source_hashes={str(p):sha256(p) for p in sources},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        optimizer='RMSprop lr=1e-4 alpha=.99; extra-context and reconstruction wd=1e-4, original base wd=0',
        extra_messages='two directions/layers, bias-free linear, all-zero initialization, additive receiver degree^(-alpha) normalized context',
        checkpoint_selection='mean(inner AUC, inner AP)',
        config_selection='minimum mean AUC/AP gain over every assigned comparator',
        promotion_minimum_both_gains_over_each_control=.002,
        derivation_sha256=sha256(HERE/'README.md'),initial_checks_sha256=sha256(HERE/'initial_checks.json'),
        predecessor_review_sha256=sha256(HERE/'predecessor_review.json'),previous_selection_sha256=sha256(previous),
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
