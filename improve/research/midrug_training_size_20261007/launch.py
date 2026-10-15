"""Freeze the data-scale diagnostic and run twelve new fits."""
import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from pathlib import Path
from common import HERE,CLEAN,RECON,RELEASE,STRICT,sha256
from model import CONFIGS,SIZES
PREVIOUS=HERE.parent/'midrug_ensemble_fusion_20261007'


def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    assert json.loads((PREVIOUS/'archive_recheck.json').read_text())['status']=='passed'
    previous=json.loads((PREVIOUS/'results/plan.json').read_text());controls=previous['controls']
    assert set(controls)=={'control','unmasked10'}
    full_sources=[CLEAN/'train.py',CLEAN/'model.py',RECON/'phase2/train.py',RECON/'phase2/common.py',RECON/'phase2/model.py',RECON/'common.py']
    for name,folds in controls.items():
        for fold,reference in folds.items():
            path=Path(reference['path']);r=json.loads(path.read_text());assert sha256(path)==reference['metadata_sha256']
            assert r['seed']==42 and r['fold']==int(fold)
            assert r['data_sha256']==sha256(CLEAN/'inner_data.npz') and r['graph_sha256']==sha256(CLEAN/'graph.pt')
            if name=='control':
                assert r['source_sha256']==sha256(CLEAN/'train.py') and r['model_sha256']==sha256(CLEAN/'model.py')
            else:
                for source,digest in r['hashes'].items():assert sha256(RECON/'phase2'/source)==digest
    sources=[HERE/n for n in ['common.py','model.py','prepare.py','check.py','train.py','launch.py']]
    sources+=[STRICT,RECON/'model.py',*full_sources,*(RELEASE/'incrd_hg_m123').glob('*.py')]
    evidence=[HERE/'README.md',HERE/'subset_manifest.json',HERE/'initial_checks.json',PREVIOUS/'archive_recheck.json',PREVIOUS/'verification.json',PREVIOUS/'post_result_review.json',PREVIOUS/'NEXT_EXPERIMENT.md']
    plan=dict(study='nested_training_size_diagnostic',configs=CONFIGS,sizes=SIZES,full_size_fraction=1.,folds=[0,1,2],seed=42,
        epochs=800,minimum_epochs=300,patience=200,controls=controls,new_fits=12,replayed_existing_full_size_models=6,
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),subsets_sha256=sha256(HERE/'subsets.npz'),
        optimizer='RMSprop alpha.99, original lr1e-4/wd0, reconstruction lr1e-4/wd1e-4; rec weight10 ramp50',
        checkpoint_selection='mean(inner AUC, inner AP)',diagnostics=['same-holdout metrics','successive size deltas','reconstruction minus control','exact-query train/heldout gaps'],
        no_candidate_selection=True,no_performance_extrapolation=True,no_automatic_outer_promotion=True,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    out=HERE/'results';out.mkdir(exist_ok=True);path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    jobs=Queue()
    for fold in range(3):
        for size in SIZES:
            for name in CONFIGS:
                if not (out/f'{name}_{size}_f{fold}_s42.json').exists():jobs.put((name,size,fold))
    def worker(device):
        while True:
            try:name,size,fold=jobs.get_nowait()
            except Empty:return
            with (out/f'{name}_{size}_f{fold}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--size',size,'--fold',str(fold),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,size,fold,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1']]:future.result()

if __name__=='__main__':main()
