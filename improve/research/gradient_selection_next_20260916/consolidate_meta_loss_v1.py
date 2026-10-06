"""Read-only replay of saved meta-loss weights, optimizer evidence and metrics."""
from datetime import datetime, timezone
import gc
from pathlib import Path

import numpy as np
import torch

from improve import anchor_meta_loss_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_meta_loss_v1_20260916'


def main():
    dest = HERE / 'META_LOSS_CONSOLIDATION_V1.json'
    if dest.exists():
        raise ValueError('Preserve completed consolidation')
    queue = base.check_seal(base.read_json(HERE / 'META_LOSS_QUEUE_RESULT_V1.json'))
    stage = queue['stage']
    p, ph = experiment.common.protocol(OUT), base.digest(OUT / 'protocol.json')
    selection = base.read_json(OUT / 'SELECTION.json')
    name = 'SCREEN_ASSESSMENT.json' if stage == 'screen' else 'HISTORICAL_ASSESSMENT.json'
    stored = base.check_seal(base.read_json(OUT / name))
    seeds = [p['screen_seed']] + (p['confirm_seeds'] if stage == 'confirm' else [])
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records, files, replays, runs = {}, {}, [], []
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
                assert len(trace) == 400 and r['training_steps'] == 400 and r['gradient_evaluations'] == 1200
                checkpoints = [row for row in trace if 'anchor_composed' in row]
                assert len(checkpoints) == 20
                selected = max(checkpoints, key=lambda row: (row['anchor_composed']['auc'] + row['anchor_composed']['ap']) / 2)
                assert r['epoch'] == selected['epoch']
                for metric in ('auc', 'ap', 'macro_drug_ap'):
                    assert abs(selected['anchor_composed'][metric] - r['metrics']['anchor_composed'][metric]) < 1e-12
                assert [row['task_moment_step'] for row in trace] == list(range(1, 401))
                final = torch.load(path.parent / 'training_final.pt', map_location='cpu', weights_only=True)
                assert final['steps'] == final['task_optimizer']['step'] == 400
                assert all(int(state['step']) == 400 for state in final['rescaler_optimizer']['state'].values())
                assert all(torch.isfinite(value).all() for part in ('first', 'second') for value in final['task_optimizer'][part].values())
                for key, memory in final['memories'].items():
                    fold = int(key.split('_')[0])
                    for values in memory['rows'].values():
                        ids = values[:, 5].numpy().astype(np.int64)
                        assert np.all(a['folds'][ids] == fold)
                if r['kind'] in ('no_meta_gradient', 'uniform'):
                    assert all(row['hypergradient_norm'] == 0. for row in trace)
                if r['kind'] in ('no_outer_rescaling', 'uniform'):
                    assert all(row['direct_gradient_norm'] == 0. for row in trace)
            else:
                assert task == 'LncDrug' and not trace and r['training_steps'] == 0
                np.testing.assert_array_equal(z['anchor'], z['anchor_composed'])
            torch.random.default_generator.manual_seed(r['seed'])
            torch.cuda.manual_seed(r['seed'])
            model, xs, graphs, _ = experiment.setup(training, p, r['kind'], 'cuda:1')
            initial_rescaler = (experiment.cpu_copy(model.rescaler.state_dict())
                                if task == 'MiDrug' and r['kind'] == 'uniform' else None)
            model.load_state_dict(torch.load(path.parent / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            if initial_rescaler is not None:
                for key, value in model.rescaler.state_dict().items():
                    torch.testing.assert_close(value.cpu(), initial_rescaler[key], atol=0, rtol=0)
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
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='meta_loss', stage=stage,
        protocol_sha256=ph, source_sha256=base.digest(__file__), result_sha256=files,
        assessment_sha256=base.digest(OUT / name), queue_result_sha256=base.digest(HERE / 'META_LOSS_QUEUE_RESULT_V1.json'),
        all_stored_metrics_recomputed=True, all_selected_checkpoints_recomputed=True,
        every_saved_weight_strictly_reloaded=True, saved_predictions_replayed=True,
        final_task_and_rescaler_optimizer_states_verified=True, training_only_memories_verified=True,
        max_replay_error=max(r['replay_max_abs'] for r in replays), replays=replays, assessment=recomputed,
        new_formal_neural_fits=neural, identity_records=len(runs)-neural,
        added_by_this_audit_neural_fits=0, added_partition_fits=0, added_outer_fits=0,
        cumulative_formal_partition_neural_fits=333, confirmed_final_innovations=0,
        existing_default_replaced=False, original_test_source_access=False))
    base.write_json(dest, result)
    print('META_LOSS_CONSOLIDATED', neural, result['max_replay_error'], recomputed['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
