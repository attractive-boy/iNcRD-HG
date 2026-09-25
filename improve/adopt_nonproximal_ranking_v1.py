"""Screen-only byte-identical adoption of observed nonproximal ranking weights."""
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil

import numpy as np
import torch

from improve import anchor_ranking_experiment_v2 as experiment
from improve import tabular_foundation as base
from improve.adopt_cooperative_local_v1 import verify_arrays


def verify_recipe(current, previous):
    for key in ('core_parameters', 'screen_seed', 'confirm_seeds', 'epochs', 'check_interval',
                'lrs', 'weight_decay', 'strong_weight', 'lnc_policy'):
        if current[key] != previous[key]:
            raise ValueError('Nonproximal recipe mismatch: ' + key)
    if previous['implementation_module'] != 'staco_ranking_v1':
        raise ValueError('Unexpected source architecture')
    if current['implementation_module'] != 'staco_ranking_v2':
        raise ValueError('Unexpected target architecture')


def cells(p, previous_selection, stage):
    if stage == 'screen':
        return [(lr, p['screen_seed']) for lr in p['lrs']]
    raise ValueError('Only screening adoption is registered')


def adopt(out, stage):
    if stage != 'screen':
        raise ValueError('Only screening adoption is registered')
    out = Path(out).resolve()
    p = experiment.common.protocol(out)
    adoption = base.read_json(Path(p['parent']) / 'protocol.json')['baseline_adoption']
    if (adoption['source_kind'], adoption['target_kind']) != ('no_proximal', 'full'):
        raise ValueError('Unregistered adoption mapping')
    source = Path(adoption['source'])
    previous = experiment.common.protocol(source)
    verify_recipe(p, previous)
    previous_selection = base.read_json(source / 'SELECTION.json')
    ph, old_ph = base.digest(out / 'protocol.json'), base.digest(source / 'protocol.json')
    result_path = out / ('RANK_NONPROX_ADOPTION_' + stage.upper() + '_V1.json')
    if result_path.exists():
        raise ValueError('Preserve completed adoption')
    if not adoption['cpu_forward_gradient_rng_equivalence_test_passed']:
        raise ValueError('Exact predictor test is required')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    records = []
    for task in p['tasks']:
        a = experiment.common.load(out / 'inputs' / task / 'arrays.npz')
        verify_arrays(a, experiment.common.load(source / 'inputs' / task / 'arrays.npz'))
        for lr, seed in cells(p, previous_selection, stage):
            old = source / 'runs' / task / 'no_proximal' / ('lr' + str(lr)) / ('s' + str(seed))
            before = base.digest(old / 'result.json')
            if adoption['result_sha256'].get(str(old / 'result.json')) != before:
                raise ValueError('Declared source changed')
            record = experiment.common.verify(old, old_ph)
            if record['seed'] != seed or record['kind'] != 'no_proximal':
                raise ValueError('Unexpected source record')
            values = experiment.common.load(old / 'scores.npz')
            model, xs, graphs, _ = experiment.setup(a, p, 'full', 'cuda:1')
            model.load_state_dict(torch.load(old / 'model.pt', map_location='cuda:1', weights_only=True))
            query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device='cuda:1')
            replay = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
            np.testing.assert_array_equal(replay, values['standalone'])
            np.testing.assert_array_equal(a['y_val'], values['y'])
            np.testing.assert_array_equal(a['pairs_val'], values['pairs'])
            np.testing.assert_array_equal(a['anchor_' + str(seed)] + p['strong_weight'] * replay,
                                          values['anchor_composed'])
            dest = out / 'runs' / task / 'full' / ('lr' + str(lr)) / ('s' + str(seed))
            if dest.exists():
                raise ValueError('Use a fresh adoption destination')
            dest.mkdir(parents=True)
            for name, digest in record['files'].items():
                shutil.copy2(old / name, dest / name)
                if base.digest(dest / name) != digest:
                    raise ValueError('Adopted artifact is not byte-identical')
            revised = dict(record, protocol_sha256=ph, kind='full',
                adopted_without_new_training=True, new_training_steps=0,
                adoption_source_result=str(old / 'result.json'), adoption_source_result_sha256=before,
                adoption_source_protocol_sha256=old_ph, adoption_replay_max_abs=0.,
                source_forward_gradient_rng_equivalence_test_passed=True)
            if record['trained_neural_model']:
                revised['completed_averaging_stages'] = revised.pop('completed_proximal_stages')
            base.write_json(dest / 'result.json', revised)
            experiment.common.verify(dest, ph)
            if base.digest(old / 'result.json') != before:
                raise ValueError('Source result unexpectedly changed')
            records.append(dict(task=task, lr=lr, seed=seed, source=str(old / 'result.json'),
                source_sha256=before, destination=str(dest / 'result.json'),
                destination_sha256=base.digest(dest / 'result.json'),
                trained_neural_model=record['trained_neural_model'], new_neural_fit=False,
                exact_artifact_copy=True, replay_max_abs=0.))
            print('ADOPTED_RANK_NONPROX', stage, task, lr, seed, record['trained_neural_model'], flush=True)
            del model, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        stage=stage, protocol_sha256=ph, source_protocol_sha256=old_ph, records=records,
        adopted_neural_models=2, adopted_identity_records=2, new_neural_fits=0,
        original_files_unchanged=True, confirmed_final_innovations=0)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--stage', required=True, choices=('screen',))
    args = parser.parse_args()
    adopt(args.out, args.stage)
