"""Freeze and run the three-way completion ablation on internal folds."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from common import HERE,CLEAN,RECON,RELEASE,STRICT,sha256
from model import CONFIGS

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    previous=HERE.parent/'midrug_soft_paths_20261006'
    assert json.loads((previous/'verification.json').read_text())['status']=='passed'
    assert not json.loads((previous/'selection.json').read_text())['promote_to_outer_validation']
    out=HERE/'results';out.mkdir(exist_ok=True);controls={}
    for name,root in [('control',CLEAN),('unmasked10',RECON/'phase2')]:
        controls[name]={str(f):dict(path=str(root/'results'/f'{name}_f{f}_s42.json'),
            checkpoint_sha256=sha256(root/'results'/f'{name}_f{f}_s42.pt')) for f in range(3)}
    comparators={}
    for name,cfg in CONFIGS.items():
        suffix='_rec10' if cfg['reconstruction_weight'] else ''
        refs=['control','unmasked10']
        if cfg['mode']=='soft':refs += ['direct'+suffix,'uniform'+suffix]
        comparators[name]=refs
    sources=[HERE/f for f in ['common.py','model.py','train.py','launch.py']]
    sources += [STRICT,RECON/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,controls=controls,
        comparators=comparators,eligible_candidates=['soft','soft_rec10'],source_hashes={str(p):sha256(p) for p in sources},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        optimizer='RMSprop lr=1e-4 alpha=.99; soft-path/reconstruction wd=1e-4, original base wd=0',
        method='two observed positives plus one unknown pair; exactly query-deleted first-layer GNN scorer rank32; stopped gradient through soft completion; query-deleted scorer BCE .1; exact endpoint exclusion; float64 accumulation',
        checkpoint_selection='mean(inner AUC, inner AP)',config_selection='minimum mean AUC/AP gain over all assigned controls',
        promotion_minimum_both_gains_over_each_control=.002,derivation_sha256=sha256(HERE/'README.md'),
        initial_checks_sha256=sha256(HERE/'initial_checks.json'),previous_selection_sha256=sha256(previous/'selection.json'),
        predecessor_review_sha256=sha256(HERE/'predecessor_review.json'),
        literature_manifest_sha256=sha256(HERE.parent/'midrug_literature_20261006/completion_reading_manifest.json'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for f in range(3):
        for name in CONFIGS:
            if not (out/f'{name}_f{f}_s42.json').exists():q.put((name,f))
    def worker(device):
        while True:
            try:name,f=q.get_nowait()
            except Empty:return
            with (out/f'{name}_f{f}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(f),'--device',device],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,f,device,flush=True)
    with ThreadPoolExecutor(4) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1','cuda:0','cuda:1']]:future.result()

if __name__=='__main__':main()
