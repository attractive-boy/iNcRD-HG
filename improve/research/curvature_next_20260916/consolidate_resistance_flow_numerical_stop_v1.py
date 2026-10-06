"""Audit completed resistance-flow fits after a frozen numerical-gate stop."""
from datetime import datetime, timezone
import gc
from pathlib import Path

import numpy as np
import torch

from improve import anchor_resistance_flow_experiment_v1 as experiment
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_resistance_flow_v1_20260916'


def main():
    destination = HERE / 'RESISTANCE_FLOW_NUMERICAL_STOP_V1.json'
    decision_path = OUT / 'NUMERICAL_STOP_DECISION.json'
    if destination.exists() or decision_path.exists():
        raise ValueError('Preserve completed numerical-stop audit')
    p, ph = experiment.common.protocol(OUT), base.digest(OUT / 'protocol.json')
    frozen = base.check_seal(base.read_json(HERE / 'RESISTANCE_FLOW_QUEUE_PROTOCOL_V1.json'))
    assert frozen['source_sha256'] == base.digest(HERE / 'run_resistance_flow_queue_v1.py')
    for path, digest in frozen['input_sha256'].items():
        assert base.digest(path) == digest, path
    worker_log = OUT / 'screen_MiDrug.log'
    queue_log = HERE / 'run_resistance_flow_queue_v1.log'
    log = worker_log.read_text()
    assert log.rstrip().endswith('FloatingPointError: Registered curvature-flow step lost positivity')
    assert 'TRAIN_RESISTANCE_FLOW local_resistance 0.001 666 300 ' in log
    assert 'TRAIN_RESISTANCE_FLOW local_resistance 0.001 666 400 ' not in log
    assert 'VERIFIED_RESISTANCE_FLOW MiDrug local_resistance' not in log
    assert "Resistance-flow worker failed: ('screen', 'MiDrug', 1)" in queue_log.read_text()
    for name in ('SELECTION.json', 'SCREEN_ASSESSMENT.json', 'HISTORICAL_ASSESSMENT.json'):
        assert not (OUT / name).exists()
    expected = {('MiDrug', 'full', lr, 666) for lr in (.001, .003)}
    found, records, hashes, rngs = set(), [], {}, []
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a = experiment.common.load(OUT / 'inputs/MiDrug/arrays.npz')
    training = {key: value for key, value in a.items() if key in experiment.TRAIN_KEYS}
    query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device='cuda:1')
    for path in sorted((OUT / 'runs').glob('*/*/*/*/result.json')):
        before = base.digest(path)
        r = experiment.common.verify(path.parent, ph)
        found.add((r['task'], r['kind'], r['lr'], r['seed']))
        assert r['trained_neural_model'] and r['training_steps'] == r['optimizer_steps'] == 400
        assert r['gradient_evaluations'] == 400
        z = experiment.common.load(path.parent / 'scores.npz')
        np.testing.assert_array_equal(z['y'], a['y_val'])
        np.testing.assert_array_equal(z['pairs'], a['pairs_val'])
        for prefix in ('anchor', 'gnn', 'tabicl'):
            np.testing.assert_array_equal(z[prefix], a[prefix + '_' + str(r['seed'])])
            np.testing.assert_array_equal(z[prefix + '_composed'], z[prefix] + .1 * z['standalone'])
        np.testing.assert_array_equal(z['offset'], z['anchor'])
        np.testing.assert_array_equal(z['offset_composed'], z['anchor'])
        for score, expected_metrics in {**r['metrics'], **r['offset_metrics']}.items():
            actual = base.metrics(z['y'], z[score], z['pairs'])
            for metric in ('auc', 'ap', 'macro_drug_ap'):
                assert abs(expected_metrics[metric] - actual[metric]) < 1e-12
        trace = base.read_json(path.parent / 'trace.json')
        assert [row['epoch'] for row in trace] == list(range(20, 401, 20))
        selected = max(trace, key=lambda row: (row['anchor_composed']['auc'] + row['anchor_composed']['ap']) / 2)
        assert r['epoch'] == selected['epoch']
        for metric in ('auc', 'ap', 'macro_drug_ap'):
            assert abs(selected['anchor_composed'][metric] - r['metrics']['anchor_composed'][metric]) < 1e-12
        d = r['flow_diagnostics']
        assert d['calls'] == 505 and d['entity_flows'] == 1010
        assert d['inverse_calls'] == d['flow_steps'] == 3030
        assert d['minimum_multiplier'] > 0. and d['minimum_true_resistance'] > 0.
        assert d['maximum_inverse_relative_residual'] <= 1e-10
        for row in trace:
            calls = row['epoch'] + 5 * (row['epoch'] // 20)
            detail = row['flow_diagnostics']
            assert detail['calls'] == calls and detail['entity_flows'] == 2 * calls
            assert detail['inverse_calls'] == detail['flow_steps'] == 6 * calls
            assert np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
        final = torch.load(path.parent / 'training_final.pt', map_location='cpu', weights_only=True)
        assert final['steps'] == 400
        assert all(int(state['step']) == 400 for state in final['optimizer']['state'].values())
        assert all(torch.isfinite(value).all() for state in final['optimizer']['state'].values()
                   for value in state.values() if isinstance(value, torch.Tensor))
        assert all(torch.isfinite(value).all() for value in final['model'].values())
        rngs.append({name: final[name] for name in ('cpu_rng', 'cuda_rng')})
        model, xs, graphs, _ = experiment.setup(training, p, r['kind'], 'cuda:1')
        model.load_state_dict(torch.load(path.parent / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
        reverse = experiment.predict(model, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
        chunks = np.concatenate([experiment.predict(model, xs, graphs, query[:, i:i + 800], p).cpu().numpy()
                                 for i in range(0, query.shape[1], 800)])
        error = max(float(np.max(abs(reverse - z['standalone']))), float(np.max(abs(chunks - z['standalone']))))
        assert error <= 3e-5
        assert base.digest(path) == before
        hashes[str(path)] = before
        records.append(dict(lr=r['lr'], seed=r['seed'], kind=r['kind'], epoch=r['epoch'],
            metrics=r['metrics']['anchor_composed'], reference=r['metrics']['anchor'],
            delta={key: r['metrics']['anchor_composed'][key] - r['metrics']['anchor'][key] for key in ('auc', 'ap')},
            replay_max_abs=error, flow_diagnostics=d, recipe_selected=False))
        del model, xs, graphs
        gc.collect()
        torch.cuda.empty_cache()
    assert found == expected
    for name in ('cpu_rng', 'cuda_rng'):
        torch.testing.assert_close(rngs[0][name], rngs[1][name], atol=0., rtol=0.)
    incomplete = OUT / 'runs/MiDrug/local_resistance/lr0.001/s666'
    assert incomplete.is_dir() and not list(incomplete.iterdir())
    run_directories = set(path for path in (OUT / 'runs').glob('*/*/*/*') if path.is_dir())
    completed_directories = {Path(path).parent for path in hashes}
    assert run_directories == completed_directories | {incomplete}
    decision = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        decision='Stop the frozen version at its first numerical-gate failure; no step clipping, retuning or continuation',
        protocol_sha256=ph, failure='Nonpositive or nonfinite explicit distance-flow multiplier',
        stage='screen', task='MiDrug', kind='local_resistance', lr=.001, seed=666,
        partial_fit_completed_optimizer_steps_bounds=[300, 399], exact_failure_step_saved=False,
        full_performance_assessment_possible=False, historical_joint_gate=None,
        remaining_screen_arms_run=False, confirmation_run=False, partition_run=False, outer_run=False,
        confirmed_final_innovations=0, default_replaced=False,
        log_sha256={str(path): base.digest(path) for path in (worker_log, queue_log)}))
    base.write_json(decision_path, decision)
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='resistance_flow',
        protocol_sha256=ph, source_sha256=base.digest(__file__), queue_protocol_sha256=frozen['sha256'],
        numerical_stop_decision_sha256=base.digest(decision_path), result_sha256=hashes,
        new_formal_neural_fit_attempts=3, completed_new_formal_neural_fits=2,
        incomplete_formal_neural_fits=1, incomplete_directory=str(incomplete),
        new_formal_identity_records=0, prior_short_neural_fits=3, prior_short_identity_replays=3,
        records=records, all_completed_metrics_recomputed=True, all_completed_checkpoints_recomputed=True,
        completed_models_strictly_reloaded=True, completed_predictions_replayed=True,
        completed_final_optimizer_and_random_states_verified=True, source_and_input_hashes_verified=True,
        added_neural_fits_by_this_audit=0, added_partition_fits=0, added_outer_fits=0,
        cumulative_formal_partition_neural_fits=333, original_test_source_access=False,
        historical_joint_gate=None, full_mechanism_evidence_incomplete=True,
        confirmed_final_innovations=0, default_replaced=False))
    base.write_json(destination, result)
    print('RESISTANCE_FLOW_NUMERICAL_STOP_AUDITED', len(records),
          max(record['replay_max_abs'] for record in records), flush=True)


if __name__ == '__main__':
    main()
