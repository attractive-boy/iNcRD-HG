"""Three training-only short fits and forty support/completion isolation audits."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_bottleneck_experiment_v1 as experiment
from improve import information_bottleneck_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/information_bottleneck_smoke_v1_20260916'


def assert_sparse_equal(left, right):
    if left.shape != right.shape or (left != right).nnz:
        raise ValueError('Support or completion changed after omitted-label flip')


def main():
    destination = HERE / 'INFORMATION_BOTTLENECK_SMOKE_V1.json'
    if destination.exists() or OUT.exists():
        raise ValueError('Preserve previous smoke artifacts')
    resource_path = HERE / 'CORE_RESOURCE_V1.json'
    resource = base.check_seal(base.read_json(resource_path))
    for path, digest in resource['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Resource-checked core changed')
    OUT.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='information_bottleneck',
        implementation_module='information_bottleneck_graph_v1',
        source_sha256={'information_bottleneck_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1)
    records, audits, input_hashes = [], [], {str(resource_path): base.digest(resource_path)}
    for task in ('MiDrug', 'LncDrug'):
        source = experiment.ROOT / 'results/anchor_residual_energy_v2_20260916/inputs' / task / 'arrays.npz'
        input_hashes[str(source)] = base.digest(source)
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        # These fitting negatives are absent from every support graph; no evaluation arrays are read.
        ids = np.flatnonzero(a['y_train'] == 0)[:97]
        pairs = torch.as_tensor(a['pairs_train'][:, ids] + a['pair_offset'][:, None], device='cuda:1')
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
            dest = OUT / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True))
            value = experiment.predict(model, xs, graphs, pairs, p).cpu().numpy()
            replay = experiment.predict(loaded, xs, graphs, pairs.flip(1), p).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, pairs[:, i:i + 41], p)
                                    .cpu().numpy() for i in range(0, pairs.shape[1], 41)])
            for actual in (replay, chunks):
                np.testing.assert_allclose(actual, value, atol=3e-5, rtol=3e-5)
            if task == 'LncDrug':
                np.testing.assert_array_equal(value, np.zeros_like(value))
                if trace or detail['trained_neural_model'] or detail['training_steps'] or detail['gradient_evaluations']:
                    raise ValueError('Identity deployment was incorrectly treated as training')
            else:
                if detail['gradient_evaluations'] != 5 or not detail['trained_neural_model'] or len(trace) != 5:
                    raise ValueError('Wrong short-fit gradient budget')
                if {row['omitted_fold'] for row in trace} != set(range(5)):
                    raise ValueError('Short fit did not visit all five supports')
                if any((row['weighted_kl'] != 0) != (kind != 'no_kl') for row in trace):
                    raise ValueError('KL branch changed')
                if kind == 'no_completion' and any(row['virtual_edges'] for row in trace):
                    raise ValueError('Completion-free control received virtual edges')
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(value - replay).max(), abs(value - chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                query_source='97 fitting negatives, no evaluation rows',
                source_sha256={str(path): base.digest(path) for path in
                    (Path(core.__file__), Path(experiment.__file__),
                     Path(__file__), dest / 'model.pt', dest / 'trace.json')}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('BOTTLENECK_SMOKE_RECORD', task, kind, detail['trained_neural_model'],
                  detail['training_steps'], record['replay_max_abs'], flush=True)
            del model, loaded, xs, graphs
            core._CONTEXTS.clear()
            gc.collect()
            torch.cuda.empty_cache()
        paths = [('historical', source)] + [(split, experiment.ROOT /
            'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            for split in (61453, 7919, 104729)]
        for split, path in paths:
            input_hashes[str(path)] = base.digest(path)
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            if core.identity_task(training, core.parameters()):
                model, xs, graphs, folds = core.setup(training, core.parameters(), 'full', 'cpu')
                if any(graph is not None for graph in graphs) or list(model.parameters()):
                    raise ValueError('Identity audit unexpectedly built a model or graph')
                audits.extend(dict(task=task, split=split, fold=fold, identity=True,
                    new_graph_constructed=False, new_neural_fit=False) for fold in range(5))
                continue
            for fold in range(5):
                omitted = np.flatnonzero(training['folds'] == fold)
                query = training['pairs_train'][:, omitted] + training['pair_offset'][:, None]
                edges, _ = core.physical.masked_physical(training, omitted)
                raw = core.boolean_adjacency(edges, int(training['node_counts'].sum()))
                graph = core.context(training, omitted, core.parameters(), 'full')
                changed = {k: v.copy() for k, v in training.items()}
                changed['y_train'][omitted] = 1 - changed['y_train'][omitted]
                changed_edges, _ = core.physical.masked_physical(changed, omitted)
                # Independent construction avoids relying on a cache hit as an invariance test.
                other = core.BottleneckContext(changed_edges, training['node_counts'],
                                               core.parameters()['completion_ratio'])
                np.testing.assert_array_equal(edges, changed_edges)
                assert_sparse_equal(graph.adjacency, other.adjacency)
                assert_sparse_equal(graph.origin, other.origin)
                counts = []
                for matrix in (raw, graph.adjacency, graph.origin):
                    overlap = int(np.count_nonzero(np.asarray(matrix[query[0], query[1]])))
                    overlap += int(np.count_nonzero(np.asarray(matrix[query[1], query[0]])))
                    if overlap:
                        raise ValueError('A current target query entered support or completion')
                    counts.append(overlap)
                selected = np.linspace(0, query.shape[1] - 1, 17, dtype=int)
                for u, v in query[:, selected].T:
                    for left, right in zip(graph.extract_extra(u, v), other.extract_extra(u, v)):
                        np.testing.assert_array_equal(left, right)
                if graph.audit['cross_type_virtual_edges'] or graph.audit['virtual_edges_are_supervision']:
                    raise ValueError('Virtual-edge semantics changed')
                audits.append(dict(task=task, split=split, fold=fold, identity=False,
                    original_forbidden_overlap=counts[0], augmented_forbidden_overlap=counts[1],
                    virtual_forbidden_overlap=counts[2], query_count=query.shape[1],
                    query_label_flip_invariant=True, independent_reconstruction=True,
                    exact_sampled_subgraphs_checked=17, completion=graph.audit))
                print('BOTTLENECK_FOLD_AUDIT', task, split, fold, query.shape[1],
                      graph.audit['virtual_undirected_edges'], flush=True)
                core._CONTEXTS.clear()
                del graph, other, raw
                gc.collect()
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), records=records,
        fold_audits=audits, core_parameters=core.parameters(), device='cuda:1',
        new_short_neural_fits=3, identity_replays=3, formal_fits=0,
        evaluation_labels_read=False, performance_metrics_computed=False, confirmed_innovations=0,
        input_sha256=input_hashes,
        source_sha256={str(path): base.digest(path) for path in
            (Path(__file__), Path(core.__file__), Path(experiment.__file__),
             experiment.ROOT / 'test_anchor_bottleneck_experiment_v1.py')}))
    base.write_json(destination, result)
    print('BOTTLENECK_SMOKE_COMPLETE', 3, 3, len(audits), result['sha256'], flush=True)


if __name__ == '__main__':
    main()
