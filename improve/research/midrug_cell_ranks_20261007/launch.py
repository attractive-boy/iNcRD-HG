"""Freeze once before fitting; reruns resume only absent complete fits."""
import json, sys, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from common import HERE, CLEAN, RELEASE, RECON, STRICT, sha256
from model import CONFIGS

def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status'] == 'passed'
    out = HERE/'results'; out.mkdir(exist_ok=True)
    prior = json.loads((HERE.parent/'midrug_training_size_20261007/results/plan.json').read_text())
    controls = prior['controls']
    sources = [HERE/f for f in ['common.py','prepare.py','model.py','check.py','train.py','launch.py','audit.py','summarize.py']]
    sources += [STRICT, RECON/'model.py', HERE.parent/'midrug_degree_encoder_20261006/check.py',
                *(RELEASE/'incrd_hg_m123').glob('*.py')]
    inputs = [HERE/'covariates.npz', HERE/'input_manifest.json', HERE/'initial_checks.json', HERE/'README.md',
              CLEAN/'graph.pt', CLEAN/'inner_data.npz',
              HERE.parent/'midrug_affine_covariates_20261007/selection.json',
              HERE.parent/'midrug_affine_covariates_20261007/LITERATURE_NOTES.md',
              HERE.parent/'midrug_affine_covariates_20261007/parameterfree_biorxiv.txt',
              HERE.parent/'midrug_training_size_20261007/results/plan.json',
              HERE.parent/'midrug_internal_20261006/plmf_protection.json']
    for refs in controls.values():
        for ref in refs.values():
            path = Path(ref['path'])
            assert sha256(path) == ref['metadata_sha256']
            assert sha256(path.with_suffix('.pt')) == ref['checkpoint_sha256']
            assert sha256(path.with_name(path.stem+'_inner.npz')) == ref['probability_sha256']
            inputs += [path, path.with_suffix('.pt'), path.with_name(path.stem+'_inner.npz')]
    manifest = json.loads((HERE/'input_manifest.json').read_text())
    plan = dict(configs=CONFIGS, folds=[0,1,2], seed=42, epochs=800, minimum_epochs=300, patience=200,
        controls=controls, eligible_candidate='cell_gaussian',
        comparators=['cell_minmax','row_gaussian','control','unmasked10'], minimum_both_mean_gains=.002,
        checkpoint_selection='earliest maximum mean(inner AUC,AP)', new_fits=9,
        source_hashes={str(p):sha256(p) for p in sources},
        input_hashes={**{str(p):sha256(p) for p in inputs}, **manifest['original_source_hashes']},
        no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False)
    path = out/'plan.json'
    if path.exists(): assert json.loads(path.read_text()) == plan
    else: path.write_text(json.dumps(plan, indent=2))
    queue = Queue()
    for fold in range(3):
        for name in CONFIGS:
            if not (out/f'{name}_f{fold}_s42.json').exists(): queue.put((name,fold))
    def worker(device):
        while True:
            try: name, fold = queue.get_nowait()
            except Empty: return
            with (out/f'{name}_f{fold}_s42.log').open('x') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(fold),'--device',device],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE', name, fold, device, flush=True)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker, d) for d in ['cuda:0','cuda:1']]: future.result()
if __name__ == '__main__': main()
