"""Freeze the label-free feature routing experiment before any fit."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from common import HERE,CLEAN,RECON,RELEASE,STRICT,sha256
from model import CONFIGS


def main():
    checks=json.loads((HERE/'initial_checks.json').read_text());assert checks['status']=='passed'
    previous=json.loads((HERE.parent/'midrug_exact_attention_20261006/selection.json').read_text())
    assert not previous['promote_to_outer_validation']
    out=HERE/'results';out.mkdir(exist_ok=True);controls={}
    for name,root in [('control',CLEAN),('unmasked10',RECON/'phase2')]:
        controls[name]={str(f):dict(path=str(root/'results'/f'{name}_f{f}_s42.json'),
            checkpoint_sha256=sha256(root/'results'/f'{name}_f{f}_s42.pt')) for f in range(3)}
    comparators={}
    for name,cfg in CONFIGS.items():
        refs=['control','unmasked10'];suffix='_rec10' if cfg['reconstruction_weight'] else ''
        if cfg['mode']!='local':refs.append('local'+suffix)
        if cfg['mode']=='route':refs.append('pool'+suffix)
        comparators[name]=refs
    sources=[HERE/f for f in ['common.py','model.py','train.py','launch.py']]
    sources += [STRICT,RECON/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,controls=controls,
        comparators=comparators,source_hashes={str(p):sha256(p) for p in sources},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        optimizer='RMSprop lr=1e-4 alpha=.99; active refiner/reconstruction wd=1e-4, original base wd=0',
        refiner='width64, 8 learned tokens per type, .1*original_feature_sd residual scale, zero output initialization; deterministic no dropout',
        checkpoint_selection='mean(inner AUC, inner AP)',
        config_selection='minimum mean AUC/AP gain over every assigned comparator',
        promotion_minimum_both_gains_over_each_control=.002,
        literature_sha256=sha256(HERE.parent/'midrug_literature_20261006/FOLLOWUP_NOTES.md'),
        initial_checks_sha256=sha256(HERE/'initial_checks.json'),
        previous_attention_selection_sha256=sha256(HERE.parent/'midrug_exact_attention_20261006/selection.json'),
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
