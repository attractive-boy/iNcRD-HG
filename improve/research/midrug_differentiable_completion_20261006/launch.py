"""Freeze a gradient-only change and six new internal fits."""
import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RECON,RELEASE,STRICT,sha256
from model import CONFIGS,PREVIOUS

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    audit=json.loads((PREVIOUS/'verification.json').read_text());assert audit['status']=='passed'
    assert not json.loads((PREVIOUS/'selection.json').read_text())['promote_to_outer_validation']
    prior_tags={r['tag'] for r in audit['records'] if r['status']=='passed'}
    controls={}
    locations=[('control',CLEAN/'results'),('unmasked10',RECON/'phase2/results')]
    locations += [(n,PREVIOUS/'results') for n in ['soft','soft_rec10','direct','direct_rec10','uniform','uniform_rec10']]
    for name,root in locations:
        controls[name]={}
        for f in range(3):
            stem=f'{name}_f{f}_s42';assert stem in prior_tags
            controls[name][str(f)]=dict(path=str(root/f'{stem}.json'),metadata_sha256=sha256(root/f'{stem}.json'),
                checkpoint_sha256=sha256(root/f'{stem}.pt'),probability_sha256=sha256(root/f'{stem}_inner.npz'))
    sources=[HERE/f for f in ['common.py','model.py','train.py','launch.py']]
    sources += [PREVIOUS/'model.py',STRICT,RECON/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    comparators={n:['control','unmasked10']+[k+('_rec10' if c['reconstruction_weight'] else '') for k in ['soft','direct','uniform']] for n,c in CONFIGS.items()}
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,controls=controls,
        comparators=comparators,eligible_candidates=list(CONFIGS),source_hashes={str(p):sha256(p) for p in sources},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        optimizer='RMSprop lr=1e-4 alpha=.99; completion/reconstruction wd=1e-4, original base wd=0',
        method='Identical graph-conditioned query-repaired forward function, full gradients through completion probability and path fields; all other hyperparameters unchanged',
        checkpoint_selection='mean(inner AUC, inner AP)',config_selection='minimum mean AUC/AP gain over all assigned controls',
        promotion_minimum_both_gains_over_each_control=.002,derivation_sha256=sha256(HERE/'README.md'),
        initial_checks_sha256=sha256(HERE/'initial_checks.json'),previous_selection_sha256=sha256(PREVIOUS/'selection.json'),
        previous_verification_sha256=sha256(PREVIOUS/'verification.json'),predecessor_review_sha256=sha256(HERE/'predecessor_review.json'),
        reused_control_verification='Prior complete checkpoint replay plus unchanged source/checkpoint/probability/input hashes, and current saved-score metric replay',
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    out=HERE/'results';out.mkdir(exist_ok=True);p=out/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:p.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for f in range(3):
        for n in CONFIGS:
            if not (out/f'{n}_f{f}_s42.json').exists():q.put((n,f))
    def worker(device):
        while True:
            try:n,f=q.get_nowait()
            except Empty:return
            with (out/f'{n}_f{f}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',n,'--fold',str(f),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',n,f,device,flush=True)
    with ThreadPoolExecutor(4) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1','cuda:0','cuda:1']]:future.result()

if __name__=='__main__':main()
