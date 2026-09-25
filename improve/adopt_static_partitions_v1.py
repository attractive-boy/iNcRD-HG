"""Exact reuse of previously observed static-policy partition artifacts."""
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil

import numpy as np
import torch

from improve import anchor_plain_validate_v4 as validation
from improve import anchor_plain_validate_v2 as previous_validation
from improve import tabular_foundation as base


def verify_recipe(current, previous):
    if current['family'] != 'cooperative_static' or previous['family'] != 'cooperative_local':
        raise ValueError('Unexpected partition adoption families')
    for key in ('data_protocol_sha256', 'model_seed', 'nested_protocol_sha256', 'strong_weight', 'lnc_policy'):
        if current[key] != previous[key]:
            raise ValueError('Partition recipe mismatch: ' + key)
    for key in ('core_parameters', 'epochs', 'weight_decay', 'strong_weight', 'lnc_policy'):
        if current['training_parameters'][key] != previous['training_parameters'][key]:
            raise ValueError('Partition training recipe mismatch: ' + key)
    for task in current['recipes']:
        if current['recipes'][task]['full'] != previous['recipes'][task]['static_policy']:
            raise ValueError('Selected partition steps or learning rate changed')


def adopt(out):
    out = Path(out).resolve()
    dp, cp = validation.checked(out, 'cooperative_static')
    old_dp, old_cp = previous_validation.checked(out, 'cooperative_local')
    verify_recipe(cp, old_cp)
    if dp['sha256'] != old_dp['sha256']:
        raise ValueError('Data inputs differ')
    declaration = cp['partition_adoption']
    if declaration['source_candidate_sha256'] != old_cp['sha256']:
        raise ValueError('Declared source candidate changed')
    result_path = out / 'reports' / cp['label'] / 'STATIC_PARTITION_ADOPTION_V1.json'
    if result_path.exists():
        raise ValueError('Preserve completed partition adoption')
    experiment = validation.experiment_for('cooperative_static')
    p = cp['training_parameters']
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    rows = []
    for task in dp['tasks']:
        for split in dp['split_seeds']:
            old = out / 'runs' / old_cp['label'] / task / str(split) / 'static_policy'
            source_path = old / 'result.json'
            before = base.digest(source_path)
            if declaration['result_sha256'][str(source_path)] != before:
                raise ValueError('Declared partition result changed')
            r = base.read_json(source_path)
            if r['candidate_sha256'] != old_cp['sha256'] or r['recipe'] != cp['recipes'][task]['full']:
                raise ValueError('Source result recipe mismatch')
            validation.shared.verify_artifacts(old, r)
            a, _ = validation.shared.inputs.arrays(out, dp, task, split)
            training = {k: v for k, v in a.items() if k in experiment.TRAIN_KEYS}
            training['anchor_oof'], manifest = experiment.offset_arrays(cp['nested_reference'], task, split, training)
            if base.digest(manifest) != r['nested_offset_manifest_sha256']:
                raise ValueError('Training offsets differ')
            model, xs, graphs, _ = experiment.setup(training, p, 'full', 'cuda:1')
            model.load_state_dict(torch.load(old / 'model.pt', map_location='cuda:1', weights_only=True))
            query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device='cuda:1')
            replay = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
            scores = experiment.common.load(old / 'scores.npz')
            np.testing.assert_array_equal(a['pairs_val'], scores['pairs'])
            np.testing.assert_array_equal(a['y_val'], scores['y'])
            np.testing.assert_array_equal(replay, scores['standalone'])
            np.testing.assert_array_equal(scores['anchor'] + cp['strong_weight'] * replay, scores['anchor_composed'])
            dest = out / 'runs' / cp['label'] / task / str(split) / 'full'
            if dest.exists():
                raise ValueError('Use a fresh partition adoption destination')
            dest.mkdir(parents=True)
            for name, digest in r['files'].items():
                shutil.copy2(old / name, dest / name)
                if base.digest(dest / name) != digest:
                    raise ValueError('Partition artifact copy changed bytes')
            revised = dict(r, candidate_sha256=cp['sha256'], kind='full',
                adopted_without_new_training=True, new_training_steps=0,
                adoption_source_result=str(source_path), adoption_source_result_sha256=before,
                adoption_source_candidate_sha256=old_cp['sha256'], adoption_replay_max_abs=0.,
                source_forward_gradient_rng_equivalence_test_passed=True)
            base.write_json(dest / 'result.json', revised)
            validation.shared.verify_artifacts(dest, revised)
            if base.digest(source_path) != before:
                raise ValueError('Original source changed')
            rows.append(dict(task=task, split=split, source=str(source_path), source_sha256=before,
                destination=str(dest / 'result.json'), destination_sha256=base.digest(dest / 'result.json'),
                trained_neural_model=r['trained_neural_model'], new_neural_fit=False, replay_max_abs=0.))
            print('ADOPTED_STATIC_PARTITION', task, split, r['trained_neural_model'], flush=True)
            del model, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        candidate_sha256=cp['sha256'], records=rows, adopted_neural_fits=3,
        adopted_identity_records=3, new_neural_fits=0, exact_artifact_copies=True,
        confirmed_final_innovations=0, source_files_preserved=True)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    adopt(parser.parse_args().out)
