"""Freeze the independent ensemble protocol, then dispatch 21 member fits."""
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RELEASE,STRICT,RECON,sha256,member_seed

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    previous=HERE.parent/'midrug_structural_teacher_20261007'
    assert json.loads((previous/'archive_recheck.json').read_text())['status']=='passed'
    sources=[HERE/n for n in ['common.py','model.py','train.py','check.py','launch.py']]
    sources+=[STRICT,RECON/'model.py',HERE.parent/'midrug_degree_encoder_20261006/check.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    evidence=[HERE/'README.md',HERE/'initial_checks.json',previous/'archive_recheck.json',previous/'REPORT.md',
        HERE.parent/'midrug_training_size_20261007/verification.json',HERE.parent/'midrug_training_size_20261007/results/plan.json',
        HERE.parent/'midrug_ensemble_fusion_20261007/REPORT.md',HERE.parent/'midrug_weight_average_20261007/REPORT.md',
        HERE.parent.parent/'ensemble.py',HERE.parent.parent/'val_ensemble.py',HERE.parent.parent/'RESULTS.md']
    controls=json.loads((HERE.parent/'midrug_training_size_20261007/results/plan.json').read_text())['controls']
    plan=dict(config='ensemble8',members=8,root_seed=42,member_seeds=[member_seed(42,j) for j in range(8)],
        future_root_seeds={str(r):[member_seed(r,j) for j in range(8)] for r in [42,43,44]},folds=[0,1,2],new_fits=21,
        controls=controls,source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        aggregation='float64 equal mean of eight saved float32 probabilities',member0='immutable unmasked10 seed42',
        sampling_seed='20261006+fold shared across members',optimizer='original RMSprop alpha=.99 lr1e-4 main wd0 reconstruction wd1e-4',
        reconstruction_weight=10.,reconstruction_warmup=50,maximum_epochs=800,minimum_epochs=300,patience=200,
        checkpoint_selection='Earliest maximum mean(inner AUC,inner AP) independently per member',
        comparators=['control','unmasked10','mean_individual_members'],eligible_candidates=['ensemble8'],
        promotion_minimum_both_gains_over_each_control=.002,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    out=HERE/'results';out.mkdir(exist_ok=True);path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    jobs=Queue()
    for fold in range(3):
        for member in range(1,8):
            if not (out/f'member{member}_f{fold}_r42.json').exists():jobs.put((member,fold))
    def worker(device):
        while True:
            try:member,fold=jobs.get_nowait()
            except Empty:return
            with (out/f'member{member}_f{fold}_r42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--member',str(member),'--fold',str(fold),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',member,fold,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1']]:future.result()

if __name__=='__main__':main()
