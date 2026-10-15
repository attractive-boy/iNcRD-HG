"""Freeze nine masked-context teacher fits."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,CLEAN,RELEASE,RECON,STRICT,sha256
from model import CONFIGS


def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status']=='passed'
    previous=HERE.parent/'midrug_fusion_normalization_20261007';assert json.loads((previous/'archive_recheck.json').read_text())['status']=='passed'
    controls=json.loads((previous/'results/plan.json').read_text())['controls']
    sources=[HERE/n for n in ['common.py','model.py','train.py','check.py','launch.py','prepare.py']]
    sources+=[STRICT,RECON/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    evidence=[HERE/'README.md',HERE/'initial_checks.json',HERE/'mask_manifest.json',HERE/'masks.npz',
        previous/'archive_recheck.json',previous/'verification.json',previous/'results/plan.json',previous/'REPORT.md',
        HERE.parent/'midrug_training_size_20261007/verification.json',
        HERE.parent/'midrug_representation_geometry_20261007/FUSION_FOLLOWUP.md',
        HERE.parent/'midrug_representation_literature_20261007/manifest.json',
        HERE.parent/'midrug_representation_literature_20261007/READING_NOTES.md',
        HERE.parent.parent/'masked_graph_teacher.py',HERE.parent/'new_mechanisms_20260915/READING_NOTES.md']
    evidence+=list((HERE/'precheck_unbalanced_masks').iterdir())
    evidence+=list((HERE/'precheck_teacher_copy_failure').iterdir())
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,new_fits=9,controls=controls,
        source_hashes={str(p):sha256(p) for p in sources},evidence_hashes={str(p):sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        latent_target='Type-centered L2-normalized clean teacher states; masked targets only',
        context_predictor='Same/cross visible-source means with four degree descriptors; per-type 516->128->256 GELU',
        latent_weight=.1,latent_warmup_start=30,latent_warmup_end=60,
        masks_sha256=sha256(HERE/'masks.npz'),mask_fraction=.5,paired_node_exposures_matched=True,
        main_lr=1e-4,reconstruction_lr=1e-4,reconstruction_weight_decay=1e-4,optimizer='RMSprop alpha=.99 set_to_none=True',
        auxiliary_weight=10.,auxiliary_warmup=50,maximum_epochs=800,minimum_epochs=300,patience=200,
        checkpoint_selection='Earliest maximum mean(inner AUC,inner AP)',
        eligible_candidates=['block_ema'],comparators={'block_ema':['control','unmasked10','random_ema','block_current']},
        promotion_minimum_both_gains_over_each_control=.002,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    out=HERE/'results';out.mkdir(exist_ok=True);path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    jobs=Queue()
    for f in range(3):
        for n in CONFIGS:
            if not (out/f'{n}_f{f}_s42.json').exists():jobs.put((n,f))
    def worker(device):
        while True:
            try:name,fold=jobs.get_nowait()
            except Empty:return
            with (out/f'{name}_f{fold}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(fold),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,fold,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker,device) for device in ['cuda:0','cuda:1']]:future.result()


if __name__=='__main__':main()
