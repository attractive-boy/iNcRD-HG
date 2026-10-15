"""Freeze representation perturbation and twelve new original-training fits."""
import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RECON,RELEASE,STRICT,sha256
from model import CONFIGS
PREVIOUS=HERE.parent/'midrug_sam_20261006'

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    audit=json.loads((PREVIOUS/'verification.json').read_text());assert audit['status']=='passed'
    assert not json.loads((PREVIOUS/'selection.json').read_text())['promote_to_outer_validation']
    old=json.loads((PREVIOUS/'results/plan.json').read_text());controls={n:old['controls'][n] for n in ['control','unmasked10']}
    for n in ['sam','sam_rec10']:
        controls['weight_'+n]={}
        for fold in range(3):
            p=PREVIOUS/'results'/f'{n}_f{fold}_s42.json'
            controls['weight_'+n][str(fold)]=dict(path=str(p),metadata_sha256=sha256(p),checkpoint_sha256=sha256(p.with_suffix('.pt')),probability_sha256=sha256(p.with_name(p.stem+'_inner.npz')))
    sources=[HERE/n for n in ['common.py','model.py','train.py','launch.py']]
    sources += [STRICT,RECON/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    comparators={n:['control','unmasked10','random'+s,'weight_sam'+s] for n,s in [('adv',''),('adv_rec10','_rec10')]}
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,controls=controls,
        eligible_candidates=['adv','adv_rec10'],comparators=comparators,source_hashes={str(p):sha256(p) for p in sources},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        optimizer='RMSprop lr=1e-4 alpha=.99; original wd0, reconstruction wd1e-4',
        feature_perturbation='Per-query standardized L2 ball radius .05*sqrt(width), std over original-training features only, floor .001, gradient active threshold1e-12',
        loss='(1-alpha)*clean BCE + alpha*perturbed BCE + reconstruction; alpha .5*ramp(epoch30..60), reconstruction ramp0..50',
        mask_seed='20261006 + fold',noise_seed='61007000 + fold',checkpoint_selection='mean(inner AUC, inner AP)',
        config_selection='minimum paired mean AUC/AP gain over all assigned controls',promotion_minimum_both_gains_over_each_control=.002,
        derivation_sha256=sha256(HERE/'README.md'),initial_checks_sha256=sha256(HERE/'initial_checks.json'),predecessor_review_sha256=sha256(HERE/'predecessor_review.json'),
        previous_verification_sha256=sha256(PREVIOUS/'verification.json'),previous_selection_sha256=sha256(PREVIOUS/'selection.json'),
        previous_sensitivity_sha256=sha256(PREVIOUS/'sensitivity_diagnostics.json'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    out=HERE/'results';out.mkdir(exist_ok=True);p=out/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:p.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for fold in range(3):
        for n in CONFIGS:
            if not (out/f'{n}_f{fold}_s42.json').exists():q.put((n,fold))
    def worker(device):
        while True:
            try:n,fold=q.get_nowait()
            except Empty:return
            with (out/f'{n}_f{fold}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',n,'--fold',str(fold),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',n,fold,device,flush=True)
    with ThreadPoolExecutor(4) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1','cuda:0','cuda:1']]:future.result()

if __name__=='__main__':main()
