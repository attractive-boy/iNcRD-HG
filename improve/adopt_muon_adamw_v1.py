"""Read-only audit and exact artifact adoption of the completed ERM baseline."""
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil

import numpy as np
import torch

from improve import anchor_muon_experiment_v1 as experiment
from improve import tabular_foundation as base


def verify_recipe(current, previous):
    for key in ('core_parameters', 'screen_seed', 'epochs', 'check_interval', 'lrs',
                'weight_decay', 'strong_weight', 'lnc_policy'):
        if current[key] != previous[key]:
            raise ValueError('Baseline recipe mismatch: ' + key)
    if previous['implementation_module'] != 'sharpness_graph_v1':
        raise ValueError('Unexpected source architecture')
    if current['implementation_module'] != 'muon_graph_v1':
        raise ValueError('Unexpected target architecture')
    if current['optimizer_parameters']['betas'] != [.9, .999] or current['optimizer_parameters']['eps'] != 1e-8:
        raise ValueError('AdamW defaults changed')
    if previous['optimizer_parameters']['clip'] != 5.:
        raise ValueError('Source gradient clipping changed')


def verify_arrays(current, previous):
    if set(current) != set(previous):
        raise ValueError('Baseline input members differ')
    for key in current:
        np.testing.assert_array_equal(current[key], previous[key], err_msg='Baseline input mismatch: ' + key)


def adopt(out):
    out = Path(out).resolve()
    p = experiment.common.protocol(out)
    source = Path(p['baseline_adoption']['source'])
    previous = experiment.common.protocol(source)
    verify_recipe(p, previous)
    ph, old_ph = base.digest(out / 'protocol.json'), base.digest(source / 'protocol.json')
    result_path = out / 'ADAMW_ADOPTION_V1.json'
    if result_path.exists():
        raise ValueError('Preserve the completed baseline adoption')
    if not p['baseline_adoption']['cpu_exact_trajectory_test_passed']:
        raise ValueError('The required exact trajectory check is missing')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    records = []
    for task in p['tasks']:
        a = experiment.common.load(out / 'inputs' / task / 'arrays.npz')
        old_a = experiment.common.load(source / 'inputs' / task / 'arrays.npz')
        verify_arrays(a, old_a)
        for lr in p['lrs']:
            old = source / 'runs' / task / 'erm' / ('lr' + str(lr)) / ('s' + str(p['screen_seed']))
            digest_before = base.digest(old / 'result.json')
            if p['baseline_adoption']['result_sha256'].get(str(old / 'result.json')) != digest_before:
                raise ValueError('The declared baseline source changed')
            record = experiment.common.verify(old, old_ph)
            if record['seed'] != p['screen_seed'] or record['kind'] != 'erm':
                raise ValueError('Unexpected baseline record')
            values = experiment.common.load(old / 'scores.npz')
            model, xs, graphs, _ = experiment.setup(a, p, 'adamw', 'cuda:1')
            model.load_state_dict(torch.load(old / 'model.pt', map_location='cuda:1', weights_only=True))
            query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device='cuda:1')
            replay = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
            np.testing.assert_array_equal(replay, values['standalone'])
            np.testing.assert_array_equal(a['y_val'], values['y'])
            np.testing.assert_array_equal(a['pairs_val'], values['pairs'])
            np.testing.assert_array_equal(a['anchor_' + str(p['screen_seed'])] + p['strong_weight'] * replay,
                                          values['anchor_composed'])
            dest = out / 'runs' / task / 'adamw' / ('lr' + str(lr)) / ('s' + str(p['screen_seed']))
            if dest.exists():
                raise ValueError('Use a fresh adoption destination')
            dest.mkdir(parents=True)
            for name, digest in record['files'].items():
                shutil.copy2(old / name, dest / name)
                if base.digest(dest / name) != digest:
                    raise ValueError('Adopted artifact is not byte-identical')
            revised = dict(record, protocol_sha256=ph, kind='adamw',
                adopted_without_new_training=True, new_training_steps=0,
                adoption_source_result=str(old / 'result.json'),
                adoption_source_result_sha256=digest_before,
                adoption_source_protocol_sha256=old_ph, adoption_replay_max_abs=0.,
                original_gradient_budget_preserved=True,
                mathematical_one_pass_erm_equivalence_test_passed=True)
            base.write_json(dest / 'result.json', revised)
            experiment.common.verify(dest, ph)
            if base.digest(old / 'result.json') != digest_before:
                raise ValueError('Source result was unexpectedly modified')
            records.append(dict(task=task, lr=lr, source=str(old / 'result.json'),
                source_sha256=digest_before, destination=str(dest / 'result.json'),
                destination_sha256=base.digest(dest / 'result.json'),
                trained_neural_model=record['trained_neural_model'], new_neural_fit=False,
                exact_artifact_copy=True, replay_max_abs=0.))
            print('ADOPTED_ADAMW', task, lr, record['trained_neural_model'], flush=True)
            del model, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=ph, source_protocol_sha256=old_ph, records=records,
        adopted_neural_models=2, adopted_identity_records=2, new_neural_fits=0,
        original_files_unchanged=True, confirmed_final_innovations=0)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    adopt(args.out)
