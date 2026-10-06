"""Read-only Stuart-Landau score, checkpoint, state and saved-weight audit."""
from datetime import datetime, timezone
import gc
from pathlib import Path

import numpy as np
import torch

from improve import anchor_stuart_landau_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base
from improve.research.oscillatory_next_20260916 import run_stuart_landau_queue_v1 as queue_checks

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_stuart_landau_v1_20260916'


def main():
    destination = HERE / 'STUART_LANDAU_CONSOLIDATION_V1.json'
    if destination.exists():
        raise ValueError('Preserve completed consolidation')
    queue = base.check_seal(base.read_json(HERE / 'STUART_LANDAU_QUEUE_RESULT_V1.json'))
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
    records, files, replays, runs, rngs, counts, capacities = {}, {}, [], [], {}, {}, {}
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
                np.testing.assert_array_equal(z[prefix], a[prefix+'_'+str(r['seed'])])
                np.testing.assert_array_equal(z[prefix+'_composed'], z[prefix]+.1*z['standalone'])
            np.testing.assert_array_equal(z['offset'], z['anchor'])
            np.testing.assert_array_equal(z['offset_composed'], z['anchor'])
            for score, expected in {**r['metrics'], **r['offset_metrics']}.items():
                actual = base.metrics(z['y'], z[score], z['pairs'])
                for metric in ('auc', 'ap', 'macro_drug_ap'):
                    assert abs(expected[metric]-actual[metric]) < 1e-12
            rng = queue_checks.audit_record(path.parent, r, a, p)
            if rng is not None:
                trace = base.read_json(path.parent / 'trace.json')
                chosen = max(trace, key=lambda row: (row['anchor_composed']['auc']+row['anchor_composed']['ap'])/2)
                assert r['epoch'] == chosen['epoch']
                for metric in ('auc', 'ap', 'macro_drug_ap'):
                    assert abs(chosen['anchor_composed'][metric]-r['metrics']['anchor_composed'][metric]) < 1e-12
                d = r['oscillator_diagnostics']
                fixed_counts = {key: d[key] for key in queue_checks.COUNT_KEYS}
                if task in counts:
                    assert counts[task] == fixed_counts
                else:
                    counts[task] = fixed_counts
                capacities[r['kind']] = {key: r[key] for key in ('parameters', 'trainable_parameters', 'fixed_parameters')}
                key = (r['lr'], r['seed'])
                if key in rngs:
                    for direction in ('cpu_rng', 'cuda_rng'):
                        torch.testing.assert_close(rng[direction], rngs[key][direction], atol=0., rtol=0.)
                else:
                    rngs[key] = rng
            model, xs, graphs, _ = experiment.setup(training, p, r['kind'], 'cuda:1')
            model.load_state_dict(torch.load(path.parent / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            actual = experiment.predict(model, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(model, xs, graphs, query[:, i:i+503], p).cpu().numpy()
                                     for i in range(0, query.shape[1], 503)])
            error = float(max(np.max(np.abs(actual-z['standalone'])), np.max(np.abs(chunks-z['standalone']))))
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
    assert neural == queue['new_neural_fits'] and len(runs)-neural == queue['identity_scoring_records']
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='stuart_landau',
        stage=stage, protocol_sha256=ph, source_sha256=base.digest(__file__), result_sha256=files,
        assessment_sha256=base.digest(OUT / filename),
        queue_result_sha256=base.digest(HERE / 'STUART_LANDAU_QUEUE_RESULT_V1.json'),
        all_stored_metrics_recomputed=True, all_selected_checkpoints_recomputed=True,
        every_saved_weight_strictly_reloaded=True, saved_predictions_replayed_in_reversed_order_and_new_batches=True,
        final_optimizer_and_random_streams_verified=True, all_model_parameters_trainable=True,
        radial_residual_tolerance_verified=True,
        same_radial_and_coupling_counts_across_arms=True, computation_counts=counts,
        capacities=capacities, equal_trainable_parameter_counts_verified=True,
        equal_function_spaces_claimed=False,
        frozen_source_hashes_verified=True, max_replay_error=max(row['replay_max_abs'] for row in replays),
        replays=replays, assessment=recomputed, new_formal_neural_fits=neural, identity_records=len(runs)-neural,
        added_by_this_audit_neural_fits=0, added_partition_fits=0, added_outer_fits=0,
        cumulative_formal_partition_neural_fits=333, confirmed_final_innovations=0,
        existing_default_replaced=False, original_test_source_access=False))
    base.write_json(destination, result)
    print('STUART_LANDAU_CONSOLIDATED', neural, result['max_replay_error'], recomputed['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
