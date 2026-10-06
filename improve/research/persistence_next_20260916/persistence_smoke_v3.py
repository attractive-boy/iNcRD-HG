"""Training-only spectral persistence resource probe, replay and support audits."""
from datetime import datetime, timezone
import gc
from pathlib import Path

import numpy as np
import torch

from improve import anchor_spectral_persistence_experiment_v1 as experiment
from improve import spectral_persistence_graph_v1 as core
from improve import tabular_foundation as base
from improve import graph_prediction_precision as precision

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT


@torch.no_grad()
def predict_omitted_fold(model, xs, graphs, query, fold=0):
    # A training pair can appear in other support folds. Replay only on the graph
    # where the whole pair's fold was removed; never average over exposed graphs.
    with precision.deterministic_evaluation():
        model.eval()
        return model(xs, graphs[fold], query).double().cpu().numpy()


def main():
    out = ROOT / 'results/spectral_persistence_smoke_v3_20260916'
    destination = HERE / 'PERSISTENCE_SMOKE_V3.json'
    if out.exists() or destination.exists():
        raise ValueError('Preserve earlier smoke evidence')
    unit = base.check_seal(base.read_json(HERE / 'PERSISTENCE_UNIT_V2.json'))
    assert unit['success'] and unit['tests'] == 8
    for path, digest in unit['source_sha256'].items():
        assert base.digest(path) == digest
    out.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='spectral_persistence', implementation_module='spectral_persistence_graph_v1',
        source_sha256={'spectral_persistence_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, strong_weight=.1)
    records, audits = [], []
    final_rngs = []
    repeatability = None
    for task in ('MiDrug', 'LncDrug'):
        source = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        reference, _ = experiment.shared.offset_arrays(
            ROOT / 'results/nested_anchor_v2_20260915/historical', task, 'historical', a)
        np.testing.assert_array_equal(a['anchor_oof'], reference)
        ids = np.flatnonzero(a['folds'] == 0)[:97]
        query = torch.as_tensor(a['pairs_train'][:, ids] + a['pair_offset'][:, None], device='cuda:1')
        if task == 'MiDrug':
            implementation = core.native()
            class Uncached:
                def event_spectra(self, *args):
                    return implementation.event_spectra(*args, 0)
            core._NATIVE = Uncached()
            try:
                repeat, _, repeat_trace, repeat_detail = experiment.fit(a, p, 'full', .001, 986, 'cuda:1', fixed_epochs=5)
            finally:
                core._NATIVE = implementation
            old_state = torch.load(ROOT / 'results/spectral_persistence_smoke_v1_20260916/MiDrug/full/model.pt',
                                   map_location='cpu', weights_only=True)
            repeat_error = max(float((value.cpu() - old_state[key]).abs().max())
                               for key, value in repeat.state_dict().items())
            for key, value in repeat.state_dict().items():
                torch.testing.assert_close(value.cpu(), old_state[key], atol=1e-6, rtol=0.)
            check_model, check_xs, check_graphs, _ = experiment.setup(a, p, 'full', 'cuda:1')
            check_model.load_state_dict(repeat.state_dict())
            check_model.eval()
            with torch.no_grad(), precision.deterministic_evaluation():
                packed = check_graphs[0].batch(query)
                h = check_model.encode(check_xs, check_graphs[0])[packed['nodes']]
                eps = core.parameters()['filtration_epsilon']
                v = eps + (1.-2.*eps)*check_model.vertex_filtration(h).squeeze(-1)
                edge = packed['edges']
                e = eps + (1.-2.*eps)*check_model.edge_filtration(h[edge[:,0]]+h[edge[:,1]]).squeeze(-1)
                args = (v.cpu().double().contiguous(), e.cpu().double().contiguous(),
                        packed['edges_cpu'].contiguous(), packed['vertex_slices'], packed['edge_slices'])
                cached_raw, uncached_raw = implementation.event_spectra(*args, 4096), implementation.event_spectra(*args, 0)
                for value, expected in zip(cached_raw[:6], uncached_raw[:6]):
                    torch.testing.assert_close(value, expected, atol=0, rtol=0)
            dest = out / 'uncached_repeat'
            dest.mkdir()
            torch.save(repeat.state_dict(), dest / 'model.pt')
            base.write_json(dest / 'trace.json', repeat_trace)
            repeatability = dict(**repeat_detail, same_seed_uncached_repeat_weight_max_abs=repeat_error,
                real_training_diagrams_cache_bitwise_equal=True,
                repeatability_weight_absolute_tolerance=1e-6,
                files={name:base.digest(dest/name) for name in ('model.pt','trace.json')})
            base.write_json(dest / 'RESULT.json', base.seal(repeatability))
            print('PERSISTENCE_UNCACHED_REPEAT', repeat_error, repeat_detail['seconds'], flush=True)
            del repeat, check_model, check_xs, check_graphs, h, v, e
            gc.collect()
            torch.cuda.empty_cache()
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
            baseline = torch.load(ROOT / 'results/spectral_persistence_smoke_v1_20260916' / task / kind / 'model.pt',
                                  map_location='cpu', weights_only=True)
            weight_error = max((float((value.cpu() - baseline[key]).abs().max())
                                for key, value in model.state_dict().items()), default=0.)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), baseline[key], atol=1e-6, rtol=0.)
            detail['uncached_smoke_weight_max_abs'] = weight_error
            detail['uncached_smoke_weight_tolerance'] = 1e-6
            dest = out / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            if detail['trained_neural_model']:
                assert detail['training_steps'] == detail['gradient_evaluations'] == detail['optimizer_steps'] == 5
                assert detail['persistence_diagnostics']['calls'] == 5
                assert detail['persistence_diagnostics']['graphs'] == len(a['y_train'])
                final_rngs.append(model.training_final_state['cuda_rng'])
                torch.save(model.training_final_state, dest / 'training_final.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            normal = predict_omitted_fold(model, xs, graphs, query)
            reverse = predict_omitted_fold(loaded, xs, graphs, query.flip(1))[::-1]
            chunks = np.concatenate([predict_omitted_fold(loaded, xs, graphs, query[:, i:i + 31])
                                     for i in range(0, query.shape[1], 31)])
            for actual in (reverse, chunks):
                np.testing.assert_allclose(normal, actual, atol=3e-5, rtol=3e-5)
            if task == 'LncDrug':
                np.testing.assert_array_equal(normal, np.zeros_like(normal))
                assert not detail['trained_neural_model'] and not trace
            names = ['model.pt', 'trace.json'] + (['training_final.pt'] if detail['trained_neural_model'] else [])
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(normal - reverse).max(), abs(normal - chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                files={name: base.digest(dest / name) for name in names}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('PERSISTENCE_SMOKE', task, kind, detail['seconds'], record['replay_max_abs'],
                  record['peak_allocated_gib'], detail['persistence_diagnostics'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
        paths = [('historical', source)] + [(str(split), ROOT /
            'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            for split in (61453, 7919, 104729)]
        for split, path in paths:
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            model, xs, graphs, folds = core.setup(training, core.parameters(), 'full', 'cpu')
            for fold, graph in enumerate(graphs):
                if task == 'LncDrug':
                    assert graph is None and not list(model.parameters())
                    audits.append(dict(task=task, split=split, fold=fold, identity=True,
                                       new_graph_constructed=False, new_neural_fit=False))
                    continue
                omitted = folds[fold].numpy()
                query = training['pairs_train'][:, omitted] + training['pair_offset'][:, None]
                n = int(training['node_counts'].sum())
                query_codes = np.r_[query[0] * n + query[1], query[1] * n + query[0]]
                edges = graph.neural_graph[0].numpy()
                live_codes = edges[0] * n + edges[1]
                assert not np.intersect1d(live_codes, query_codes).size
                local_nodes, local_edges = [], []
                for u, v in query.T:
                    vertices, edge = graph.extract(u, v)
                    expected = set(graph.adjacency[u].indices).intersection(graph.adjacency[v].indices)
                    assert set(vertices) == expected | {int(u), int(v)}
                    local_nodes.append(len(vertices))
                    local_edges.append(len(edge))
                changed = {key: value.copy() for key, value in training.items()}
                changed['y_train'][omitted] = 1. - changed['y_train'][omitted]
                physical, relation = core.graph_lib.masked_physical(changed, omitted)
                alternate = core.PersistenceContext(physical, n, core.graph_lib.graph_inputs(
                    physical, relation, np.empty((2, 0), dtype=np.int64), n,
                    int(changed['relation'].max()) + 1, 'cpu'))
                assert (graph.adjacency != alternate.adjacency).nnz == 0
                for value, expected in zip(graph.neural_graph, alternate.neural_graph):
                    torch.testing.assert_close(value, expected, atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, identity=False,
                    forbidden_overlap=0, query_labels_invariant=True, local_node_cap=None,
                    all_common_neighbors_retained=True, queries=len(omitted),
                    maximum_local_nodes=max(local_nodes), median_local_nodes=float(np.median(local_nodes)),
                    maximum_local_edges=max(local_edges), median_local_edges=float(np.median(local_edges)),
                    queries_with_any_local_edge=int(np.count_nonzero(local_edges))))
            del model, xs, graphs
            gc.collect()
    for value in final_rngs[1:]:
        torch.testing.assert_close(value, final_rngs[0], atol=0, rtol=0)
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__),
        ROOT / 'spectral_persistence_native_v1.cpp', ROOT / 'spectral_persistence_build_v1.py',
        ROOT / 'test_spectral_persistence_graph_v1.py', ROOT / 'anchor_priority_graph_v1.py',
        ROOT / 'results/spectral_persistence_native_v1_20260916/BUILD.json']
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family='spectral_persistence', core_parameters=core.parameters(), records=records, fold_audits=audits,
        device='cuda:1', new_short_neural_fits=4, identity_replays=3, formal_fits=0,
        uncached_repeatability=repeatability,
        cuda_random_stream_equal_across_arms=True, replay_uses_only_omitted_fold=True,
        performance_metrics_computed=False, evaluation_labels_read=False,
        source_sha256={str(path): base.digest(path) for path in sources}))
    base.write_json(destination, result)
    print('PERSISTENCE_SMOKE_COMPLETE', len(records), len(audits), flush=True)


if __name__ == '__main__':
    main()
