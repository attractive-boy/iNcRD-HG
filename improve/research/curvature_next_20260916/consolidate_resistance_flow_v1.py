"""Read-only resistance-flow metric, checkpoint and saved-weight audit."""
from datetime import datetime, timezone
import gc
from pathlib import Path

import numpy as np
import torch

from improve import anchor_resistance_flow_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_resistance_flow_v1_20260916'


def main():
    dest = HERE / 'RESISTANCE_FLOW_CONSOLIDATION_V1.json'
    if dest.exists():
        raise ValueError('Preserve completed consolidation')
    queue = base.check_seal(base.read_json(HERE / 'RESISTANCE_FLOW_QUEUE_RESULT_V1.json'))
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
    records, files, replays, runs, rngs, counts = {}, {}, [], [], {}, {}
    for task in p['tasks']:
        a = experiment.common.load(OUT / 'inputs' / task / 'arrays.npz')
        training = {k: v for k, v in a.items() if k in experiment.TRAIN_KEYS}
        query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device='cuda:1')
        for path in sorted((OUT / 'runs' / task).glob('*/*/*/result.json')):
            before = base.digest(path)
            r = experiment.common.verify(path.parent, ph)
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
                d = r['flow_diagnostics']
                assert d['calls'] == 505 and d['entity_flows'] == 1010
                assert d['inverse_calls'] == d['flow_steps'] == 3030
                assert d['maximum_inverse_relative_residual'] <= 1e-10
                assert d['minimum_multiplier'] > 0. and d['minimum_true_resistance'] > 0.
                assert all(np.isfinite(value) for value in d.values())
                fixed_counts = {key: d[key] for key in ('calls', 'entity_flows', 'inverse_calls', 'flow_steps')}
                if task in counts:
                    assert counts[task] == fixed_counts
                else:
                    counts[task] = fixed_counts
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
    assert neural == queue['new_neural_fits'] and len(runs) - neural == queue['identity_scoring_records']
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='resistance_flow',
        stage=stage, protocol_sha256=ph, source_sha256=base.digest(__file__), result_sha256=files,
        assessment_sha256=base.digest(OUT / filename), queue_result_sha256=base.digest(HERE / 'RESISTANCE_FLOW_QUEUE_RESULT_V1.json'),
        all_stored_metrics_recomputed=True, all_selected_checkpoints_recomputed=True,
        every_saved_weight_strictly_reloaded=True, saved_predictions_replayed=True,
        final_optimizer_and_random_streams_verified=True,
        same_inverse_and_flow_step_counts_across_arms=True, flow_counts=counts,
        frozen_source_hashes_verified=True, max_replay_error=max(row['replay_max_abs'] for row in replays),
        replays=replays, assessment=recomputed, new_formal_neural_fits=neural, identity_records=len(runs)-neural,
        added_by_this_audit_neural_fits=0, added_partition_fits=0, added_outer_fits=0,
        cumulative_formal_partition_neural_fits=333, confirmed_final_innovations=0,
        existing_default_replaced=False, original_test_source_access=False))
    base.write_json(dest, result)
    print('RESISTANCE_FLOW_CONSOLIDATED', neural, result['max_replay_error'], recomputed['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
