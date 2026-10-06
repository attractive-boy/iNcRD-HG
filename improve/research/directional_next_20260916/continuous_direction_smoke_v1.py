"""Training-only CoED adaptation smoke, saved-state replay and fold audits."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_direction_experiment_v1 as experiment
from improve import continuous_direction_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT


def main():
    out = ROOT / 'results/continuous_direction_smoke_v1_20260916'
    destination = HERE / 'CONTINUOUS_DIRECTION_SMOKE_V1.json'
    if out.exists() or destination.exists():
        raise ValueError('Preserve existing smoke evidence')
    out.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='continuous_direction', implementation_module='continuous_direction_graph_v1',
        source_sha256={'continuous_direction_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1)
    records, audits = [], []
    for task in ('MiDrug', 'LncDrug'):
        source = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        query = torch.as_tensor(a['pairs_train'][:, :97] + a['pair_offset'][:, None], device='cuda:1')
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
            dest = out / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            if detail['trained_neural_model']:
                torch.save(model._training_final_state, dest / 'training_final.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            normal = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
            reverse = experiment.predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, query[:, i:i + 31], p)
                .cpu().numpy() for i in range(0, query.shape[1], 31)])
            for actual in (reverse, chunks):
                np.testing.assert_allclose(normal, actual, atol=3e-5, rtol=3e-5)
            if task == 'LncDrug':
                np.testing.assert_array_equal(normal, np.zeros_like(normal))
                assert not detail['trained_neural_model'] and not trace
            elif kind != 'fixed_direction':
                assert detail['direction_summary']['mean_abs_angle_change'] > 0
            names = ['model.pt', 'trace.json'] + (['training_final.pt'] if detail['trained_neural_model'] else [])
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(normal - reverse).max(), abs(normal - chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                files={name: base.digest(dest / name) for name in names}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('DIRECTION_SMOKE', task, kind, detail['seconds'], record['replay_max_abs'],
                  record['peak_allocated_gib'], detail['direction_summary'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
        paths = [('historical', source)] + [(str(split), ROOT /
            'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            for split in (61453, 7919, 104729)]
        for split, path in paths:
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            model, xs, graphs, folds = core.setup(training, core.parameters(), 'full', 'cpu')
            for fold in range(5):
                if task == 'LncDrug':
                    assert graphs[fold] is None and not list(model.parameters())
                    audits.append(dict(task=task, split=split, fold=fold, identity=True,
                                       new_graph_constructed=False, new_neural_fit=False))
                    continue
                graph = graphs[fold]
                omitted = folds[fold].numpy()
                q = training['pairs_train'][:, omitted] + training['pair_offset'][:, None]
                n = int(training['node_counts'].sum())
                query_codes = np.r_[q[0] * n + q[1], q[1] * n + q[0]]
                edges = graph.edges.numpy()
                live_codes = np.r_[edges[0] * n + edges[1], edges[1] * n + edges[0]]
                if np.intersect1d(live_codes, query_codes).size:
                    raise ValueError('Query or reverse query remains in directional support')
                np.testing.assert_array_equal(edges, training['edges'][:, graph.edge_ids.numpy()])
                changed = {key: value.copy() for key, value in training.items()}
                changed['y_train'][omitted] = 1. - changed['y_train'][omitted]
                other = core.context(changed, omitted, 'cpu')
                for field in ('edges', 'edge_ids', 'relation', 'counts'):
                    torch.testing.assert_close(getattr(graph, field), getattr(other, field), atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, identity=False,
                    forbidden_overlap=0, query_labels_invariant=True, angle_indices_aligned=True))
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__),
               ROOT / 'test_continuous_direction_graph_v1.py', ROOT / 'anchor_priority_graph_v1.py']
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family='continuous_direction', core_parameters=core.parameters(), records=records, fold_audits=audits,
        device='cuda:1', new_short_neural_fits=3, identity_replays=3, formal_fits=0,
        performance_metrics_computed=False, evaluation_labels_read=False,
        source_sha256={str(path): base.digest(path) for path in sources}))
    base.write_json(destination, result)
    print('DIRECTION_SMOKE_COMPLETE', len(records), len(audits), flush=True)


if __name__ == '__main__':
    main()
