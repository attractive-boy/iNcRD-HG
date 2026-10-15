"""Freeze six joint-ensemble head fits, then train without original holdouts."""
import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from pathlib import Path
from common import HERE,CLEAN,RECON,RELEASE,STRICT,sha256
from model import CONFIGS
PREVIOUS=HERE.parent/'midrug_cell_qknorm_20261007'


def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    assert json.loads((PREVIOUS/'archive_recheck.json').read_text())['status']=='passed'
    previous=json.loads((PREVIOUS/'results/plan.json').read_text())
    controls={n:previous['controls'][n] for n in ['control','unmasked10']}
    sources=[HERE/n for n in ['common.py','model.py','train.py','launch.py','check.py']]
    sources += [STRICT,RECON/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,
        controls=controls,eligible_candidates=['independent_rec10'],
        comparators={'independent_rec10':['control','unmasked10','mixture_rec10']},
        source_hashes={str(p):sha256(p) for p in sources},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        evidence_hashes={str(p):sha256(p) for p in [HERE/'README.md',PREVIOUS/'post_result_review.json',HERE.parent/'midrug_literature_20261006/tabm_scope_manifest.json',HERE/'initial_checks.json',PREVIOUS/'archive_recheck.json',PREVIOUS/'selection.json',PREVIOUS/'verification.json']},
        mechanism='16-member shared-matrix 512/256 ensemble head after unchanged original M123 fusion; mean individual BCE versus BCE of probability mixture; both use original reconstruction10',
        checkpoint_selection='mean(inner AUC, inner AP)',promotion_minimum_both_gains_over_each_control=.002,
        new_fits=6,immutable_reused_controls=6,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
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
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1']]:future.result()

if __name__=='__main__':main()
