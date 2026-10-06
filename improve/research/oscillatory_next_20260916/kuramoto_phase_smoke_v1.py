"""Training-only Kuramoto-phase resource, replay, optimizer and fold-isolation checks."""
from datetime import datetime, timezone
import gc
from pathlib import Path
import traceback

import numpy as np
import torch

from improve import anchor_kuramoto_phase_experiment_v1 as experiment
from improve import kuramoto_phase_graph_v1 as core
from improve import graph_prediction_precision as precision
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT
OUT = ROOT / 'results/kuramoto_phase_smoke_v1_20260916'
COUNT_KEYS = ('calls', 'attention_layers', 'phase_updates', 'phase_entries',
    'feature_score_edge_heads', 'coupling_score_edge_heads', 'rotary_pairs')


@torch.no_grad()
def predict_omitted_fold(model, xs, graphs, query):
    with precision.deterministic_evaluation():
        model.eval()
        return model(xs, graphs[0], query).double().cpu().numpy()


def main():
    destination = HERE / 'KURAMOTO_PHASE_SMOKE_V1.json'
    if OUT.exists() or destination.exists():
        raise ValueError('Preserve previous short-training evidence')
    unit = base.check_seal(base.read_json(HERE / 'KURAMOTO_PHASE_UNIT_V1.json'))
    assert unit['success'] and unit['tests'] == 7
    for path, digest in unit['source_sha256'].items():
        assert base.digest(path) == digest, path
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='kuramoto_phase', implementation_module='kuramoto_phase_graph_v1',
        source_sha256={'kuramoto_phase_graph_v1.py': base.digest(core.__file__)},
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
        try:
            experiment.fit(dict(a, y_val=np.array([0.])), p, 'full', .001, 986, 'cuda:1', fixed_epochs=5)
        except ValueError as error:
            assert 'training arrays only' in str(error)
        else:
            raise AssertionError('Fixed fitting accepted an evaluation key')
        reference_state, initial_rng = None, None
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(986)
            torch.cuda.manual_seed(986)
            initial, initial_xs, initial_graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            current_state = experiment.cpu_copy(initial.state_dict())
            if reference_state is None:
                reference_state = current_state
                initial_rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state('cuda:1'))
            else:
                for name, value in current_state.items():
                    torch.testing.assert_close(value, reference_state[name], atol=0., rtol=0.)
                torch.testing.assert_close(torch.random.get_rng_state(), initial_rng[0], atol=0., rtol=0.)
                torch.testing.assert_close(torch.cuda.get_rng_state('cuda:1'), initial_rng[1], atol=0., rtol=0.)
            if task == 'MiDrug':
                fold_edge_counts = [g[0].shape[1] for g in initial_graphs]
            zero = predict_omitted_fold(initial, initial_xs, initial_graphs, query)
            np.testing.assert_array_equal(zero, np.zeros_like(zero))
            del initial, initial_xs, initial_graphs
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, value, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
            assert value is None and all('anchor_composed' not in row for row in trace)
            dest = OUT / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            names = ['model.pt', 'trace.json', 'replay.npz']
            if detail['trained_neural_model']:
                d = detail['phase_diagnostics']
                assert detail['training_steps'] == detail['gradient_evaluations'] == detail['optimizer_steps'] == 5
                cp=p['core_parameters']
                nodes,depth,width,heads=int(a['node_counts'].sum()),cp['depth'],cp['width'],cp['heads']
                expected=dict(calls=5,attention_layers=5*depth,phase_updates=5*depth,
                    phase_entries=5*depth*nodes*(width//2),
                    feature_score_edge_heads=depth*heads*sum(fold_edge_counts),
                    coupling_score_edge_heads=depth*heads*sum(fold_edge_counts),
                    rotary_pairs=depth*(4*5*nodes+sum(fold_edge_counts))*(width//2))
                assert {key:d[key] for key in COUNT_KEYS}==expected
                assert all(np.isfinite(value) for value in d.values())
                assert d['maximum_phase_norm_error'] <= cp['phase_norm_tolerance']
                final = model.training_final_state
                assert final['steps'] == 5
                for name, value in model.named_parameters():
                    if value.requires_grad:
                        assert value.grad is not None and torch.isfinite(value.grad).all()
                    else:
                        assert value.grad is None
                        torch.testing.assert_close(value.cpu(), current_state[name], atol=0., rtol=0.)
                indexes = final['optimizer']['param_groups'][0]['params']
                assert len(indexes) == len(final['optimizer_parameter_names']) == len(final['optimizer']['state'])
                for name, index in zip(final['optimizer_parameter_names'], indexes):
                    state = final['optimizer']['state'][index]
                    assert int(state['step']) == 5
                    assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor))
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor.cpu(), final['model'][name], atol=0., rtol=0.)
                final_rngs.append({name: final[name] for name in ('cpu_rng', 'cuda_rng')})
                torch.save(final, dest / 'training_final.pt')
                names.append('training_final.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            normal = predict_omitted_fold(model, xs, graphs, query)
            reverse = predict_omitted_fold(loaded, xs, graphs, query.flip(1))[::-1]
            chunks = np.concatenate([predict_omitted_fold(loaded, xs, graphs, query[:, i:i+31])
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
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(normal-reverse).max(), abs(normal-chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                files={name: base.digest(dest / name) for name in names}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('KURAMOTO_PHASE_SMOKE', task, kind, detail['seconds'], record['replay_max_abs'],
                  record['peak_allocated_gib'], detail['trainable_parameters'], detail['phase_diagnostics'], flush=True)
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
                query_codes = np.r_[query_pairs[0]*n+query_pairs[1], query_pairs[1]*n+query_pairs[0]]
                edges = graph[0].numpy()
                live_codes = edges[0]*n+edges[1]
                assert not np.intersect1d(live_codes, query_codes).size
                changed = {key: value.copy() for key, value in training.items()}
                changed['y_train'][omitted] = 1. - changed['y_train'][omitted]
                physical, relation = core.graph_lib.masked_physical(changed, omitted)
                alternate = core.graph_lib.graph_inputs(physical, relation,
                    np.empty((2, 0), dtype=np.int64), n, int(changed['relation'].max())+1, 'cpu')
                for value, expected in zip(graph, alternate):
                    torch.testing.assert_close(value, expected, atol=0., rtol=0.)
                audits.append(dict(task=task, split=split, fold=fold, identity=False,
                    forbidden_overlap=0, query_labels_invariant=True, queries=len(omitted)))
            del model, xs, graphs
            gc.collect()
    for rng in final_rngs[1:]:
        for name in ('cpu_rng', 'cuda_rng'):
            torch.testing.assert_close(rng[name], final_rngs[0][name], atol=0., rtol=0.)
    trained = [r for r in records if r['trained_neural_model']]
    for record in trained[1:]:
        for key in COUNT_KEYS:
            assert record['phase_diagnostics'][key] == trained[0]['phase_diagnostics'][key]
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__),
        ROOT / 'test_kuramoto_phase_graph_v1.py', ROOT / 'anchor_priority_graph_v1.py',
        ROOT / 'graph_prediction_precision.py', ROOT / 'overlap_block_models.py', ROOT / 'sharpness_graph_v1.py']
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family='kuramoto_phase', core_parameters=core.parameters(), records=records, fold_audits=audits,
        device='cuda:1', new_short_neural_fits=3, identity_replays=3, formal_fits=0,
        common_initial_parameters_equal=True, initial_control_function_equivalence_checked_in_unit=True, initial_random_streams_equal=True,
        final_random_streams_equal_across_arms=True, initial_residual_exact_zero=True,
        every_active_parameter_gradient_finite=True, fixed_parameters_unchanged=True,
        phase_and_message_counts_equal_across_arms=True, fixed_fit_rejects_evaluation_keys=True,
        replay_uses_only_omitted_fold=True, evaluation_labels_read=False, performance_metrics_computed=False,
        source_sha256={str(path): base.digest(path) for path in sources}, input_sha256=inputs,
        confirmed_innovations=0, existing_default_replaced=False))
    base.write_json(destination, result)
    print('KURAMOTO_PHASE_SMOKE_COMPLETE', len(records), len(audits), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        failure = HERE / 'KURAMOTO_PHASE_SMOKE_FAILURE_V1.json'
        if not failure.exists():
            base.write_json(failure, dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                traceback=traceback.format_exc(), formal_fits=0))
        raise
