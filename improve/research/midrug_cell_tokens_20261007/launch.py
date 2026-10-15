"""Freeze drug-conditioned vs mean-conditioned expression-column token attention."""
import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RECON,RELEASE,STRICT,sha256
from model import CONFIGS
PREVIOUS=HERE.parent/'midrug_boundary_supervision_20261007'

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    assert json.loads((PREVIOUS/'verification.json').read_text())['status']=='passed'
    assert not json.loads((PREVIOUS/'selection.json').read_text())['promote_to_outer_validation']
    prior=json.loads((PREVIOUS/'results/plan.json').read_text());controls={n:prior['controls'][n] for n in ['control','unmasked10']}
    sources=[HERE/n for n in ['common.py','model.py','train.py','launch.py']]
    sources += [STRICT,RECON/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,controls=controls,
        eligible_candidates=['drug','drug_rec10'],comparators={'drug':['control','unmasked10','mean'],'drug_rec10':['control','unmasked10','mean_rec10']},
        source_hashes={str(p):sha256(p) for p in sources},graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        method='Drug query attends over all938 original expression columns, using column profiles across605 original miRNAs plus candidate-miRNA scalar expression; lin1 residual fusion',
        control='Mean of all216 original drug query representations; identical parameters, initialization and per-pair all-column attention',
        optimizer='RMSprop alpha.99, original lr1e-4/wd0, branch+projection lr3e-4/wd1e-4, reconstruction lr1e-4/wd1e-4',
        mask_seed='20261006 + fold',checkpoint_selection='mean(inner AUC, inner AP)',config_selection='minimum paired mean AUC/AP gain over all assigned controls',
        promotion_minimum_both_gains_over_each_control=.002,derivation_sha256=sha256(HERE/'README.md'),initial_checks_sha256=sha256(HERE/'initial_checks.json'),
        predecessor_review_sha256=sha256(HERE/'predecessor_review.json'),previous_verification_sha256=sha256(PREVIOUS/'verification.json'),previous_selection_sha256=sha256(PREVIOUS/'selection.json'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
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
