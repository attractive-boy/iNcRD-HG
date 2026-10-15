"""Freeze configurations and two mixture weights before all internal evaluations."""
import json
import subprocess
import sys
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from common import HERE, CLEAN, RELEASE, RECON, STRICT, sha256
from model import CONFIGS


def main():
    assert json.loads((HERE/'initial_checks.json').read_text())['status'] == 'passed'
    out = HERE/'results'
    out.mkdir(exist_ok=True)
    controls = {}
    for name, root in [('control',CLEAN), ('unmasked10',RECON/'phase2')]:
        controls[name] = {str(f): dict(path=str(root/'results'/f'{name}_f{f}_s42.json'),
            checkpoint_sha256=sha256(root/'results'/f'{name}_f{f}_s42.pt'),
            prediction_sha256=sha256(root/'results'/f'{name}_f{f}_s42_inner.npz')) for f in range(3)}
    sources = [HERE/'common.py', HERE/'model.py', HERE/'train.py', HERE/'launch.py', STRICT]
    plan = dict(configs=CONFIGS, folds=[0,1,2], deterministic=True, max_steps=3000, stationarity_tolerance=1e-5,
        controls=controls, source_hashes={str(p):sha256(p) for p in sources},
        graph_sha256=sha256(CLEAN/'graph.pt'), data_sha256=sha256(CLEAN/'inner_data.npz'),
        fit_selection='training objective convergence only; no heldout checkpoint selection',
        mixture='(1-w)*unmasked10_logit+w*(nominal_training_prior_logit+completion_residual)',
        mixture_weights=[.1,.25],
        config_selection='minimum paired mean AUC/AP gain versus both unmasked10 and pure M123 controls',
        promotion_minimum_both_gains_over_each_control=.002,
        no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False)
    path = out/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        path.write_text(json.dumps(plan, indent=2))
    queue = Queue()
    for fold in range(3):
        for name in CONFIGS:
            if not (out/f'{name}_f{fold}.json').exists():
                queue.put((name,fold))
    def worker():
        while True:
            try:
                name,fold = queue.get_nowait()
            except Empty:
                return
            with (out/f'{name}_f{fold}.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(fold)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print('DONE',name,fold,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker) for _ in range(2)]:
            future.result()


if __name__ == '__main__':
    main()
