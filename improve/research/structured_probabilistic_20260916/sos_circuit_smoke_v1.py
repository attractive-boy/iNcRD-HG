"""Training-only GPU probes, replay and all-fold isolation for sum-of-squares circuits."""
from datetime import datetime, timezone
import gc
from pathlib import Path

import numpy as np
import torch

from improve import anchor_sos_circuit_experiment_v1 as experiment
from improve import sos_circuit_graph_v1 as core
from improve import graph_prediction_precision as precision
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT
OUT = ROOT/'results/sos_circuit_smoke_v1_20260916'


@torch.no_grad()
def prediction(model, xs, graph, pairs):
    with precision.deterministic_evaluation():
        model.eval()
        return model(xs, graph, pairs).double().cpu().numpy()


def main():
    dest = HERE/'SOS_CIRCUIT_SMOKE_V1.json'
    if dest.exists() or OUT.exists(): raise ValueError('Preserve earlier probe evidence')
    unit = base.check_seal(base.read_json(HERE/'SOS_CIRCUIT_UNIT_V1.json'))
    assert unit['success'] and unit['tests'] == 6
    for path, digest in unit['source_sha256'].items(): assert base.digest(path) == digest, path
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='sos_circuit', implementation_module='sos_circuit_graph_v1',
        source_sha256={'sos_circuit_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5,
        check_interval=5, weight_decay=.001, strong_weight=.1)
    records, audits, inputs = [], [], {}
    for task in ('MiDrug', 'LncDrug'):
        source = ROOT/'results/anchor_priority_morphological_alignment_v1_20260916/inputs'/task/'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        inputs[str(source)] = base.digest(source)
        reference, manifest = experiment.shared.offset_arrays(
            ROOT/'results/nested_anchor_v2_20260915/historical', task, 'historical', a)
        np.testing.assert_array_equal(reference, a['anchor_oof'])
        inputs[str(manifest)] = base.digest(manifest)
        ids = np.flatnonzero(a['folds'] == 0)[:97]
        query = torch.as_tensor(a['pairs_train'][:, ids]+a['pair_offset'][:, None], device='cuda:1')
        initial_state, initial_rng, final_rngs = None, None, []
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(986)
            torch.cuda.manual_seed(986)
            model, xs, graphs, _ = core.setup(a, core.parameters(), kind, 'cuda:1')
            state = experiment.cpu_copy(model.state_dict())
            rng = [torch.random.get_rng_state(), torch.cuda.get_rng_state('cuda:1')]
            if initial_state is None: initial_state, initial_rng = state, rng
            else:
                for name, value in state.items(): torch.testing.assert_close(value, initial_state[name], atol=0, rtol=0)
                for r, s in zip(rng, initial_rng): torch.testing.assert_close(r, s, atol=0, rtol=0)
            zero = prediction(model, xs, graphs[0], query)
            np.testing.assert_array_equal(zero, np.zeros_like(zero))
            del model, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, value, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
            assert value is None and all('anchor_composed' not in r for r in trace)
            path = OUT/task/kind
            path.mkdir(parents=True)
            torch.save(model.state_dict(), path/'model.pt')
            names = ['model.pt', 'trace.json', 'replay.npz']
            if task == 'MiDrug':
                assert detail['training_steps'] == detail['optimizer_steps'] == detail['gradient_evaluations'] == 5
                assert detail['circuit_diagnostics']['forwards'] == 5
                for name, param in model.named_parameters():
                    assert param.grad is not None and torch.isfinite(param.grad).all(), name
                final = model.training_final_state
                assert final['steps'] == 5
                for state in final['optimizer']['state'].values():
                    assert int(state['step']) == 5
                    assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
                final_rngs.append([final['cpu_rng'], final['cuda_rng']])
                torch.save(final, path/'training_final.pt')
                names.append('training_final.pt')
            else:
                assert not trace and not detail['trained_neural_model'] and detail['training_steps'] == 0
                assert not model.state_dict()
            loaded, xs, graphs, _ = core.setup(a, core.parameters(), kind, 'cuda:1')
            loaded.load_state_dict(torch.load(path/'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            normalization_errors = []
            if task == 'MiDrug':
                all_pairs = np.indices((int(a['node_counts'][1]), int(a['node_counts'][2]))).reshape(2, -1)
                all_pairs = torch.as_tensor(all_pairs+a['pair_offset'][:, None], device='cuda:1')
                with torch.no_grad():
                    for relation in (model.target_relation, model.zero_relation):
                        q = model.log_density(all_pairs, relation).exp()
                        normalization_errors.append(abs(float(q.sum())-1.))
                        assert normalization_errors[-1] < 1e-10
            normal = prediction(model, xs, graphs[0], query)
            reverse = prediction(loaded, xs, graphs[0], query.flip(1))[::-1]
            chunks = np.concatenate([prediction(loaded, xs, graphs[0], query[:, i:i+31])
                                     for i in range(0, query.shape[1], 31)])
            for actual in (reverse, chunks): np.testing.assert_allclose(normal, actual, atol=3e-5, rtol=3e-5)
            if task == 'LncDrug': np.testing.assert_array_equal(normal, np.zeros_like(normal))
            else: assert np.isfinite(normal).all() and np.max(np.abs(normal)) > 1e-8
            base.write_json(path/'trace.json', trace)
            np.savez_compressed(path/'replay.npz', query=query.cpu().numpy(), normal=normal, reverse=reverse, chunks=chunks)
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(normal-reverse).max(), abs(normal-chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1')/2**30,
                real_universe_normalization_errors=normalization_errors,
                files={name:base.digest(path/name) for name in names}))
            base.write_json(path/'RESULT.json', record)
            records.append(record)
            print('SOS_CIRCUIT_SMOKE', task, kind, detail['seconds'],
                  record['replay_max_abs'], record['peak_allocated_gib'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
        for rng in final_rngs[1:]:
            for r, s in zip(rng, final_rngs[0]): torch.testing.assert_close(r, s, atol=0, rtol=0)
        paths = [('historical', source)]+[(str(s), ROOT/'results/mechanism_validation_20260915/inputs'/task/str(s)/'arrays.npz')
                                        for s in (61453, 7919, 104729)]
        for split, path in paths:
            inputs[str(path)] = base.digest(path)
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            model, xs, graphs, folds = core.setup(training, core.parameters(), 'full', 'cpu')
            for fold, graph in enumerate(graphs):
                if task == 'LncDrug':
                    assert graph is None and not list(model.parameters())
                    audits.append(dict(task=task, split=split, fold=fold, identity=True))
                    continue
                n = int(training['node_counts'].sum())
                query = training['pairs_train'][:, folds[fold].numpy()]+training['pair_offset'][:, None]
                forbidden = np.r_[query[0]*n+query[1], query[1]*n+query[0]]
                for relation in (model.target_relation, model.zero_relation):
                    edges = graph[relation].numpy()
                    assert not np.intersect1d(forbidden, edges[0]*n+edges[1]).size
                altered = {k:v.copy() for k,v in training.items()}
                altered['y_train'][folds[fold].numpy()] = 1-altered['y_train'][folds[fold].numpy()]
                e, labels = core.graph_lib.masked_physical(altered, folds[fold].numpy())
                rebuilt = [torch.as_tensor(e[:, labels == r]) for r in range(model.zero_relation)]
                zero = (altered['folds'] != fold) & (altered['y_train'] == 0)
                rebuilt.append(torch.as_tensor(altered['pairs_train'][:, zero]+altered['pair_offset'][:, None]))
                for x, y in zip(graph, rebuilt): torch.testing.assert_close(x, y, atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, identity=False,
                    positive_forbidden_overlap=0, zero_forbidden_overlap=0, query_labels_invariant=True))
            del model, xs, graphs
            gc.collect()
    base.write_json(dest, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        core_parameters=core.parameters(), records=records, fold_audits=audits,
        formal_fits=0, short_neural_probes=4, identity_probes=4,
        evaluation_labels_read=False, performance_metrics_computed=False,
        common_initial_parameters_equal=True, common_random_streams_equal=True,
        initial_residual_exact_zero=True, replay_uses_only_omitted_fold=True,
        source_sha256={str(Path(x).resolve()):base.digest(x) for x in [__file__, core.__file__, experiment.__file__]},
        input_sha256=inputs)))
    print('SOS_CIRCUIT_SMOKE_COMPLETE', len(records), len(audits), flush=True)


if __name__ == '__main__':
    main()
