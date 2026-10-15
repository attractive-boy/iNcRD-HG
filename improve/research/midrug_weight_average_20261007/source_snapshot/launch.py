"""Freeze and run three trajectories with matched raw/mean evaluations."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from common import HERE, CLEAN, RELEASE, RECON, STRICT, sha256
from model import START_EPOCH


def main():
    assert json.loads((HERE / 'initial_checks.json').read_text())['status'] == 'passed'
    previous = HERE.parent / 'midrug_reconstruction_finetune_v2_20261007'
    assert json.loads((previous / 'archive_recheck.json').read_text())['status'] == 'passed'
    literature = HERE.parent / 'midrug_finetune_literature_20261007'
    manifest = json.loads((literature / 'manifest.json').read_text())
    for path, digest in {**manifest['artifact_hashes'], **manifest['dependencies']}.items(): assert sha256(path) == digest
    controls = json.loads((previous / 'results/plan.json').read_text())['controls']
    sources = [HERE / n for n in ['common.py', 'model.py', 'train.py', 'check.py', 'launch.py']]
    sources += [STRICT, RECON / 'model.py', RECON / 'phase2/train.py', *(RELEASE / 'incrd_hg_m123').glob('*.py')]
    evidence = [HERE / 'README.md', HERE / 'initial_checks.json', literature / 'manifest.json', literature / 'READING_NOTES.md',
                previous / 'archive_recheck.json', previous / 'verification.json', previous / 'results/plan.json',
                HERE.parent / 'midrug_training_size_20261007/verification.json']
    plan = dict(configs=['raw', 'mean'], folds=[0, 1, 2], seed=42, new_training_trajectories=3, selected_models=6,
        controls=controls, source_hashes={str(p): sha256(p) for p in sources}, evidence_hashes={str(p): sha256(p) for p in evidence},
        graph_sha256=sha256(CLEAN / 'graph.pt'), data_sha256=sha256(CLEAN / 'inner_data.npz'),
        average_start_epoch=START_EPOCH, average_method='All parameter snapshots from start through current epoch, equal weight; float64 sum; copy current nonparameter buffers',
        maximum_epochs=800, minimum_epochs=300, patience=200, early_stopping_by='raw only',
        optimizer=dict(name='RMSprop', lr=1e-4, alpha=.99, main_weight_decay=0., auxiliary_weight_decay=1e-4, zero_grad_set_to_none=True),
        auxiliary_weight=10., auxiliary_warmup=50, auxiliary_node_ratio=.5,
        checkpoint_selection='Earliest maximum mean(inner AUC,inner AP) independently for raw and mean',
        eligible_candidates=['mean'], comparators={'mean': ['control', 'unmasked10', 'raw']},
        promotion_minimum_both_gains_over_each_control=.002,
        no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False)
    out = HERE / 'results'; out.mkdir(exist_ok=True); path = out / 'plan.json'
    if path.exists(): assert json.loads(path.read_text()) == plan
    else: path.write_text(json.dumps(plan, indent=2))
    def run(fold):
        if all((out / f'{n}_f{fold}_s42.json').exists() for n in ['raw', 'mean']): return
        with (out / f'trajectory_f{fold}.log').open('w') as log:
            subprocess.run([sys.executable, str(HERE / 'train.py'), '--fold', str(fold), '--device', 'cuda:' + str(fold % 2)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print('DONE', fold, flush=True)
    with ThreadPoolExecutor(2) as pool: list(pool.map(run, range(3)))


if __name__ == '__main__': main()
