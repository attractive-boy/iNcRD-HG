"""Training-only stage-crossing checks for the PULL-inspired auxiliary mechanism."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_expected_graph_experiment_v1 as experiment
from improve import expected_graph_selftraining_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/expected_graph_selftraining_smoke_v1_20260916'


def main():
    target = HERE / 'EXPECTED_GRAPH_SELFTRAINING_SMOKE_V1.json'
    if target.exists() or OUT.exists():
        raise ValueError('Preserve smoke evidence')
    OUT.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='expected_graph_selftraining',
        implementation_module='expected_graph_selftraining_v1',
        source_sha256={'expected_graph_selftraining_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=55, check_interval=20,
        weight_decay=.001, strong_weight=.1)
    records, audits, inputs = [], [], {}
    for task in ('MiDrug', 'LncDrug'):
        source = experiment.ROOT / 'results/anchor_residual_energy_v2_20260916/inputs' / task / 'arrays.npz'
        inputs[str(source)] = base.digest(source)
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        ids = np.flatnonzero(a['y_train'] == 0)[:97]
        query = torch.as_tensor(a['pairs_train'][:, ids] + a['pair_offset'][:, None], device='cuda:1')
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=55)
            dest = OUT / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            torch.save(model.augmentation_evidence, dest / 'augmentation.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True))
            value = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
            replay = experiment.predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, query[:, i:i + 41], p)
                .cpu().numpy() for i in range(0, query.shape[1], 41)])
            for actual in (replay, chunks):
                np.testing.assert_allclose(actual, value, atol=3e-5, rtol=3e-5)
            if task == 'LncDrug':
                np.testing.assert_array_equal(value, np.zeros_like(value))
                if trace or model.augmentation_evidence or detail['trained_neural_model']:
                    raise ValueError('Identity branch unexpectedly trained')
            else:
                if (detail['gradient_evaluations'] != 55 or detail['teacher_refreshes'] != 2
                        or detail['graph_forward_evaluations'] != 110 or len(trace) != 55):
                    raise ValueError('Short fit did not exercise the real stage boundary')
                if [x['before_step'] for x in model.augmentation_evidence] != [1, 51]:
                    raise ValueError('Stage interval changed')
                active = any(row['virtual_edges'] > 0 for row in trace[50:])
                if active != (kind != 'observed_only'):
                    raise ValueError('Expected-graph branch inactive or leaked to control')
                for snapshot in model.augmentation_evidence:
                    teacher, tx, tg, _ = experiment.setup(a, p, kind, 'cuda:1')
                    teacher.load_state_dict(snapshot['teacher_state'])
                    for graph, saved in zip(tg, snapshot['contexts']):
                        graph.refresh(teacher, tx, snapshot['stage'], p['core_parameters'], kind)
                        now = graph.evidence()
                        for key in ('query_pairs', 'added_pairs'):
                            np.testing.assert_array_equal(now[key], saved[key])
                        for key in ('added_weights', 'query_teacher'):
                            np.testing.assert_allclose(now[key], saved[key], atol=3e-6, rtol=3e-6)
                    del teacher, tx, tg
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(value - replay).max(), abs(value - chunks).max())),
                teacher_graph_reconstruction_checked=task == 'MiDrug',
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                files={str(path): base.digest(path) for path in
                    (dest / 'model.pt', dest / 'augmentation.pt', dest / 'trace.json')}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('EXPECTED_GRAPH_SMOKE_RECORD', task, kind, detail['training_steps'],
                  record['replay_max_abs'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
        paths = [('historical', source)] + [(split, experiment.ROOT /
            'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            for split in (61453, 7919, 104729)]
        for split, path in paths:
            inputs[str(path)] = base.digest(path)
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            model, xs, graphs, folds = core.setup(training, core.parameters(), 'full', 'cpu')
            for fold, graph in enumerate(graphs):
                if task == 'LncDrug':
                    if graph is not None or list(model.parameters()):
                        raise ValueError('Identity audit constructed a graph')
                    audits.append(dict(task=task, split=split, fold=fold, identity=True,
                                       graph_constructed=False))
                    continue
                omitted = folds[fold].numpy()
                changed = {k: v.copy() for k, v in training.items()}
                changed['y_train'][omitted] = 1 - changed['y_train'][omitted]
                other = core.ExpectedContext(changed, omitted, core.parameters(), 'cpu')
                graph.refresh(model, xs, 1, core.parameters(), 'full')
                other.refresh(model, xs, 1, core.parameters(), 'full')
                if (graph.adjacency != other.adjacency).nnz:
                    raise ValueError('Original support changed with query labels')
                for key in ('candidates', 'query_pairs', 'added', 'added_weights', 'query_teacher'):
                    np.testing.assert_array_equal(getattr(graph, key), getattr(other, key))
                torch.testing.assert_close(graph.expected.edges, other.expected.edges, atol=0, rtol=0)
                torch.testing.assert_close(graph.expected.weights, other.expected.weights, atol=0, rtol=0)
                physical = set(map(tuple, graph.original.edges.T.tolist()))
                virtual = set(map(tuple, graph.added.T))
                queries = set(map(tuple, graph.query_pairs.T))
                overlap = sum(q in physical or q[::-1] in physical or q in virtual or q[::-1] in virtual
                              for q in queries)
                if overlap:
                    raise ValueError('Current query entered a physical or expected graph')
                audits.append(dict(task=task, split=split, fold=fold, identity=False,
                    graph_constructed=True, forbidden_overlap=overlap,
                    label_flip_invariant=True, independent_reconstruction=True,
                    candidate_count=graph.candidates.shape[1], virtual_edges=graph.added.shape[1],
                    query_count=len(queries), reference_offsets_passed_to_proposals=False))
                print('EXPECTED_GRAPH_FOLD_AUDIT', task, split, fold, graph.added.shape[1], flush=True)
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), records=records,
        fold_audits=audits, core_parameters=core.parameters(), device='cuda:1',
        new_short_neural_fits=3, short_training_steps_per_fit=55, identity_replays=3,
        formal_fits=0, evaluation_labels_read=False, performance_metrics_computed=False,
        confirmed_innovations=0, input_sha256=inputs,
        source_sha256={str(path): base.digest(path) for path in
            (Path(__file__), Path(core.__file__), Path(experiment.__file__),
             experiment.ROOT / 'test_expected_graph_selftraining_v1.py',
             experiment.ROOT / 'test_anchor_expected_graph_experiment_v1.py')}))
    base.write_json(target, result)
    print('EXPECTED_GRAPH_SMOKE_COMPLETE', 3, 3, len(audits), result['sha256'], flush=True)


if __name__ == '__main__':
    main()
