"""Read-only joint-persistence audit, including unchanged-model provenance."""
from datetime import datetime, timezone
import gc
from pathlib import Path

import numpy as np
import torch

from improve import anchor_spectral_persistence_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_persistence_pairing_v1_20260916'


def main():
    dest = HERE / 'PERSISTENCE_PAIRING_CONSOLIDATION_V1.json'
    if dest.exists():
        raise ValueError('Preserve completed consolidation')
    queue = base.check_seal(base.read_json(HERE / 'PERSISTENCE_PAIRING_QUEUE_RESULT_V1.json'))
    recovery = None
    if 'recovery_protocol_sha256' in queue:
        recovery = base.check_seal(base.read_json(HERE / 'PERSISTENCE_PAIRING_RECOVERY_PROTOCOL_V1.json'))
        recovered = base.check_seal(base.read_json(HERE / 'PERSISTENCE_PAIRING_RECOVERY_RESULT_V1.json'))
        assert queue['recovery_protocol_sha256'] == recovered['recovery_protocol_sha256'] == recovery['sha256']
        assert queue['recovery_source_sha256'] == recovery['source_sha256'] == base.digest(HERE / 'resume_persistence_pairing_v1.py')
        assert recovered['input_sha256_after'] == recovery['input_sha256']
        for path, digest in recovery['input_sha256'].items():
            assert base.digest(path) == digest, path
    stage = queue['stage']
    p, ph = experiment.common.protocol(OUT), base.digest(OUT / 'protocol.json')
    selection = base.read_json(OUT / 'SELECTION.json')
    filename = 'SCREEN_ASSESSMENT.json' if stage == 'screen' else 'HISTORICAL_ASSESSMENT.json'
    stored = base.check_seal(base.read_json(OUT / filename))
    seeds = [p['screen_seed']] + (p['confirm_seeds'] if stage == 'confirm' else [])
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records, files, replays, runs, rngs = {}, {}, [], [], {}
    for task in p['tasks']:
        a = experiment.common.load(OUT / 'inputs' / task / 'arrays.npz')
        training = {key: value for key, value in a.items() if key in experiment.TRAIN_KEYS}
        query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device='cuda:1')
        for path in sorted((OUT / 'runs' / task).glob('*/*/*/result.json')):
            before = base.digest(path)
            r = experiment.common.verify(path.parent, ph)
            if r.get('adopted_without_new_training', False):
                source = Path(r['adoption_source_result'])
                assert base.digest(source) == r['adoption_source_result_sha256']
                previous = base.read_json(source)
                assert previous['kind'] == 'no_spectrum' and r['kind'] == 'full' and r['seed'] == 666
                assert r['new_training_steps'] == 0 and r['adoption_prediction_max_abs'] == 0.
                for name, digest in r['files'].items():
                    assert base.digest(source.parent / name) == digest == base.digest(path.parent / name)
            z = experiment.common.load(path.parent / 'scores.npz')
            np.testing.assert_array_equal(z['y'], a['y_val'])
            np.testing.assert_array_equal(z['pairs'], a['pairs_val'])
            for prefix in ('anchor', 'gnn', 'tabicl'):
                np.testing.assert_array_equal(z[prefix], a[prefix + '_' + str(r['seed'])])
                np.testing.assert_array_equal(z[prefix + '_composed'], z[prefix] + .1 * z['standalone'])
            np.testing.assert_array_equal(z['offset'], z['anchor'])
            np.testing.assert_array_equal(z['offset_composed'], z['anchor'])
            for score, expected in {**r['metrics'], **r['offset_metrics']}.items():
                actual = base.metrics(z['y'], z[score], z['pairs'])
                for metric in ('auc', 'ap', 'macro_drug_ap'):
                    assert abs(expected[metric] - actual[metric]) < 1e-12
            trace = base.read_json(path.parent / 'trace.json')
            if r['trained_neural_model']:
                assert len(trace) == 20 and r['training_steps'] == r['gradient_evaluations'] == 400
                selected = max(trace, key=lambda row: (row['anchor_composed']['auc'] + row['anchor_composed']['ap']) / 2)
                assert r['epoch'] == selected['epoch']
                for metric in ('auc', 'ap', 'macro_drug_ap'):
                    assert abs(selected['anchor_composed'][metric] - r['metrics']['anchor_composed'][metric]) < 1e-12
                assert r['persistence_diagnostics']['calls'] == 505
                assert r['persistence_diagnostics']['graphs'] == 80 * len(a['y_train']) + 105 * len(a['y_val'])
                final = torch.load(path.parent / 'training_final.pt', map_location='cpu', weights_only=True)
                assert final['steps'] == 400
                assert all(int(state['step']) == 400 for state in final['optimizer']['state'].values())
                assert all(torch.isfinite(value).all() for state in final['optimizer']['state'].values()
                           for value in state.values() if isinstance(value, torch.Tensor))
                key = (r['lr'], r['seed'])
                if key in rngs:
                    for direction in ('cpu_rng', 'cuda_rng'):
                        torch.testing.assert_close(final[direction], rngs[key][direction], atol=0., rtol=0.)
                else:
                    rngs[key] = {direction: final[direction] for direction in ('cpu_rng', 'cuda_rng')}
            else:
                assert task == 'LncDrug' and not trace and r['training_steps'] == 0
                np.testing.assert_array_equal(z['anchor'], z['anchor_composed'])
            model, xs, graphs, _ = experiment.setup(training, p, r['kind'], 'cuda:1')
            if task == 'MiDrug':
                for index, (name, _) in enumerate(model.named_parameters()):
                    if name.startswith('spectrum.'):
                        state = final['optimizer']['state'][index]
                        assert torch.count_nonzero(state['exp_avg']) == 0
                        assert torch.count_nonzero(state['exp_avg_sq']) == 0
            model.load_state_dict(torch.load(path.parent / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            actual = experiment.predict(model, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
            error = float(np.max(np.abs(actual - z['standalone'])))
            assert error <= 3e-5, error
            assert base.digest(path) == before
            files[str(path)] = before
            replays.append(dict(task=task, kind=r['kind'], seed=r['seed'], lr=r['lr'], replay_max_abs=error))
            runs.append(r)
            if r['lr'] == selection['chosen'][r['kind']]['lr'] and r['seed'] in seeds:
                records[task, r['kind'], r['seed']] = r
            del model, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
    recomputed = audit.evaluate(p, records, seeds)
    for key, value in recomputed.items():
        assert stored[key] == value, key
    assert queue['historical_gate'] == stored['declared_gate']
    neural = sum(r['trained_neural_model'] for r in runs)
    adopted_neural = sum(r['trained_neural_model'] and r.get('adopted_without_new_training', False) for r in runs)
    adopted_identity = sum(not r['trained_neural_model'] and r.get('adopted_without_new_training', False) for r in runs)
    assert neural - adopted_neural == queue['new_neural_fits']
    assert len(runs) - neural - adopted_identity == queue['new_identity_records']
    assert adopted_neural == queue['adopted_neural_fits'] == 2
    assert adopted_identity == queue['adopted_identity_records'] == 2
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='persistence_pairing',
        stage=stage, protocol_sha256=ph, source_sha256=base.digest(__file__), result_sha256=files,
        assessment_sha256=base.digest(OUT / filename), queue_result_sha256=base.digest(HERE / 'PERSISTENCE_PAIRING_QUEUE_RESULT_V1.json'),
        all_stored_metrics_recomputed=True, all_selected_checkpoints_recomputed=True,
        every_saved_weight_strictly_reloaded=True, saved_predictions_replayed=True,
        final_optimizer_and_random_streams_verified=True, both_arms_spectrum_optimizer_moments_zero=True,
        frozen_native_source_and_binary_hashes_verified=True, unchanged_parent_adoption_verified=True,
        max_replay_error=max(row['replay_max_abs'] for row in replays), replays=replays, assessment=recomputed,
        new_formal_neural_fits=neural - adopted_neural, adopted_neural_fits=adopted_neural,
        new_identity_records=len(runs) - neural - adopted_identity, adopted_identity_records=adopted_identity,
        added_by_this_audit_neural_fits=0, added_partition_fits=0, added_outer_fits=0,
        cumulative_formal_partition_neural_fits=333, confirmed_final_innovations=0,
        prior_parent_outcomes_seen=True, independent_screen=False,
        confirmation_io_recovery_verified=recovery is not None,
        recovery_protocol_sha256=None if recovery is None else recovery['sha256'],
        existing_default_replaced=False, original_test_source_access=False))
    base.write_json(dest, result)
    print('PERSISTENCE_PAIRING_CONSOLIDATED', neural - adopted_neural, result['max_replay_error'], recomputed['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
