"""Training-only determinant selection replay, resource probe and fold audits."""
from datetime import datetime, timezone
import gc
from pathlib import Path
import traceback

import numpy as np
import torch

from improve import anchor_determinantal_selection_experiment_v1 as experiment
from improve import determinantal_selection_graph_v1 as core
from improve import graph_prediction_precision as precision
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT
OUT = ROOT / 'results/determinantal_selection_smoke_v1_20260916'


@torch.no_grad()
def predict_omitted_fold(model, xs, graphs, query, fold=0):
    # Training pairs can remain in other support folds. Replay only on the graph
    # with their entire fold removed, never on a mean of exposed graphs.
    with precision.deterministic_evaluation():
        model.eval()
        return model(xs, graphs[fold], query).double().cpu().numpy()


def main():
    destination = HERE / 'DETERMINANTAL_SMOKE_V1.json'
    if OUT.exists() or destination.exists():
        raise ValueError('Preserve previous short-training evidence')
    unit = base.check_seal(base.read_json(HERE / 'DETERMINANTAL_UNIT_V2.json'))
    assert unit['success'] and unit['tests'] == 7
    for path, digest in unit['source_sha256'].items():
        assert base.digest(path) == digest, path
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='determinantal_selection',
        implementation_module='determinantal_selection_graph_v1',
        source_sha256={'determinantal_selection_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, strong_weight=.1)
    records, audits, final_rngs, inputs = [], [], [], {}
    for task in ('MiDrug', 'LncDrug'):
        source = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        inputs[str(source)] = base.digest(source)
        reference, manifest = experiment.shared.offset_arrays(
            ROOT / 'results/nested_anchor_v2_20260915/historical', task, 'historical', a)
        np.testing.assert_array_equal(a['anchor_oof'], reference)
        inputs[str(manifest)] = base.digest(manifest)
        ids = np.flatnonzero(a['folds'] == 0)[:97]
        query = torch.as_tensor(a['pairs_train'][:, ids] + a['pair_offset'][:, None], device='cuda:1')
        initial, initial_rng = None, None
        for kind in core.KINDS:
            # Compare real feature widths and relation counts before any fitting.
            torch.random.default_generator.manual_seed(986)
            torch.cuda.manual_seed(986)
            check_model, check_xs, check_graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            if initial is None:
                initial = experiment.cpu_copy(check_model.state_dict())
                initial_rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state('cuda:1'))
            else:
                for name, value in check_model.state_dict().items():
                    torch.testing.assert_close(value.cpu(), initial[name], atol=0., rtol=0.)
                torch.testing.assert_close(torch.random.get_rng_state(), initial_rng[0], atol=0., rtol=0.)
                torch.testing.assert_close(torch.cuda.get_rng_state('cuda:1'), initial_rng[1], atol=0., rtol=0.)
            del check_model, check_xs, check_graphs
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, value, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
            assert value is None and all('anchor_composed' not in row for row in trace)
            dest = OUT / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            if detail['trained_neural_model']:
                assert detail['training_steps'] == detail['gradient_evaluations'] == detail['optimizer_steps'] == 5
                assert detail['selection_diagnostics']['calls'] == 5
                assert detail['selection_diagnostics']['queries'] == len(a['y_train'])
                assert detail['selection_diagnostics']['pruned_queries'] > 0
                assert detail['selection_diagnostics']['minimum_selected_pivot'] > 0.
                final = model.training_final_state
                assert all(int(state['step']) == 5 for state in final['optimizer']['state'].values())
                assert all(torch.isfinite(v).all() for state in final['optimizer']['state'].values()
                           for v in state.values() if isinstance(v, torch.Tensor))
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor.cpu(), final['model'][name], atol=0., rtol=0.)
                final_rngs.append({name: final[name] for name in ('cpu_rng', 'cuda_rng')})
                torch.save(final, dest / 'training_final.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            normal = predict_omitted_fold(model, xs, graphs, query)
            reverse = predict_omitted_fold(loaded, xs, graphs, query.flip(1))[::-1]
            chunks = np.concatenate([predict_omitted_fold(loaded, xs, graphs, query[:, i:i + 31])
                                     for i in range(0, query.shape[1], 31)])
            for actual in (reverse, chunks):
                np.testing.assert_allclose(normal, actual, atol=3e-5, rtol=3e-5)
            if task == 'MiDrug':
                assert np.isfinite(normal).all() and np.max(np.abs(normal)) > 1e-8
            else:
                np.testing.assert_array_equal(normal, np.zeros_like(normal))
                assert not detail['trained_neural_model'] and not trace
            np.savez_compressed(dest / 'replay.npz', query=query.cpu().numpy(),
                                normal=normal, reverse=reverse, chunks=chunks)
            names = ['model.pt', 'trace.json', 'replay.npz']
            if detail['trained_neural_model']:
                names.append('training_final.pt')
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(normal - reverse).max(), abs(normal - chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                files={name: base.digest(dest / name) for name in names}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('DETERMINANTAL_SMOKE', task, kind, detail['seconds'], record['replay_max_abs'],
                  record['peak_allocated_gib'], detail['selection_diagnostics'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
        paths = [('historical', source)] + [(str(split), ROOT /
            'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            for split in (61453, 7919, 104729)]
        for split, path in paths:
            inputs[str(path)] = base.digest(path)
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            model, xs, graphs, folds = core.setup(training, core.parameters(), 'full', 'cpu')
            for fold, graph in enumerate(graphs):
                if task == 'LncDrug':
                    assert graph is None and not list(model.parameters())
                    audits.append(dict(task=task, split=split, fold=fold, identity=True,
                                       new_graph_constructed=False, new_neural_fit=False))
                    continue
                omitted = folds[fold].numpy()
                query_pairs = training['pairs_train'][:, omitted] + training['pair_offset'][:, None]
                n = int(training['node_counts'].sum())
                query_codes = np.r_[query_pairs[0] * n + query_pairs[1], query_pairs[1] * n + query_pairs[0]]
                edges = graph.neural_graph[0].numpy()
                live_codes = edges[0] * n + edges[1]
                assert not np.intersect1d(live_codes, query_codes).size
                sizes, membership_counts = [], []
                for u, v in query_pairs.T:
                    vertices, membership = graph.extract(u, v)
                    left, right = set(graph.adjacency[u].indices), set(graph.adjacency[v].indices)
                    assert set(vertices) == (left | right) - {int(u), int(v)}
                    np.testing.assert_array_equal(membership[:, 0], np.array([x in left for x in vertices]))
                    np.testing.assert_array_equal(membership[:, 1], np.array([x in right for x in vertices]))
                    sizes.append(len(vertices))
                    membership_counts.append(int((membership.sum(1) == 2).sum()))
                changed = {key: value.copy() for key, value in training.items()}
                changed['y_train'][omitted] = 1. - changed['y_train'][omitted]
                physical, relation = core.graph_lib.masked_physical(changed, omitted)
                alternate = core.DiversityContext(physical, n, core.graph_lib.graph_inputs(
                    physical, relation, np.empty((2, 0), dtype=np.int64), n,
                    int(changed['relation'].max()) + 1, 'cpu'))
                assert (graph.adjacency != alternate.adjacency).nnz == 0
                for value, expected in zip(graph.neural_graph, alternate.neural_graph):
                    torch.testing.assert_close(value, expected, atol=0., rtol=0.)
                audits.append(dict(task=task, split=split, fold=fold, identity=False,
                    forbidden_overlap=0, query_labels_invariant=True, candidate_node_cap=None,
                    all_union_neighbors_retained=True, queries=len(omitted),
                    maximum_candidates=max(sizes), median_candidates=float(np.median(sizes)),
                    queries_over_retention_budget=int(np.count_nonzero(np.asarray(sizes) > 16)),
                    total_candidates=sum(sizes), total_common_neighbors=sum(membership_counts)))
            del model, xs, graphs
            gc.collect()
    for rng in final_rngs[1:]:
        for name in ('cpu_rng', 'cuda_rng'):
            torch.testing.assert_close(rng[name], final_rngs[0][name], atol=0., rtol=0.)
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__),
        ROOT / 'test_determinantal_selection_graph_v1.py', ROOT / 'anchor_priority_graph_v1.py',
        ROOT / 'graph_prediction_precision.py']
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family='determinantal_selection', core_parameters=core.parameters(), records=records, fold_audits=audits,
        device='cuda:1', new_short_neural_fits=3, identity_replays=3, formal_fits=0,
        initialized_parameters_and_random_streams_equal=True, final_random_streams_equal_across_arms=True,
        replay_uses_only_omitted_fold=True, performance_metrics_computed=False, evaluation_labels_read=False,
        input_sha256=inputs, source_sha256={str(path): base.digest(path) for path in sources}))
    base.write_json(destination, result)
    print('DETERMINANTAL_SMOKE_COMPLETE', len(records), len(audits), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists() and not (OUT / 'FAILURE.json').exists():
            base.write_json(OUT / 'FAILURE.json', base.seal(dict(
                created_at_utc=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc(),
                source_sha256=base.digest(__file__), performance_metrics_computed=False,
                evaluation_labels_read=False, formal_fits=0)))
        raise
