"""One new marginal-control short fit and exact parent-primary weight reuse."""
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from improve import anchor_spectral_persistence_experiment_v1 as experiment
from improve import persistence_pairing_graph_v1 as core
from improve import spectral_persistence_graph_v1 as parent
from improve import graph_prediction_precision as precision
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT
OUT = ROOT / 'results/persistence_pairing_smoke_v1_20260916'
PARENT_OUT = ROOT / 'results/spectral_persistence_smoke_v3_20260916'


@torch.no_grad()
def replay(model, xs, graphs, query):
    with precision.deterministic_evaluation():
        model.eval()
        return model(xs, graphs[0], query).double().cpu().numpy()


def main():
    destination = HERE / 'PERSISTENCE_PAIRING_SMOKE_V1.json'
    if destination.exists() or OUT.exists():
        raise ValueError('Preserve earlier pairing short-check evidence')
    unit_path = HERE / 'PERSISTENCE_PAIRING_UNIT_V1.json'
    unit = base.check_seal(base.read_json(unit_path))
    assert unit['success'] and unit['tests'] == 4
    for path, digest in unit['source_sha256'].items():
        assert base.digest(path) == digest, path
    inherited_path = HERE / 'PERSISTENCE_SMOKE_V3.json'
    inherited = base.check_seal(base.read_json(inherited_path))
    assert len(inherited['fold_audits']) == 40 and not inherited['evaluation_labels_read']
    for path, digest in inherited['source_sha256'].items():
        assert base.digest(path) == digest, path
    # The native input and build digests are included in the frozen parent.
    experiment.common.protocol(ROOT / 'results/anchor_priority_spectral_persistence_v1_20260916')
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='persistence_pairing', implementation_module='persistence_pairing_graph_v1',
        source_sha256={'persistence_pairing_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, strong_weight=.1)
    records, inputs = [], {str(inherited_path): base.digest(inherited_path), str(unit_path): base.digest(unit_path)}
    for task in ('MiDrug', 'LncDrug'):
        path = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(path, experiment.TRAIN_KEYS)
        inputs[str(path)] = base.digest(path)
        reference, manifest = experiment.shared.offset_arrays(
            ROOT / 'results/nested_anchor_v2_20260915/historical', task, 'historical', a)
        np.testing.assert_array_equal(a['anchor_oof'], reference)
        inputs[str(manifest)] = base.digest(manifest)
        ids = np.flatnonzero(a['folds'] == 0)[:97]
        query = torch.as_tensor(a['pairs_train'][:, ids] + a['pair_offset'][:, None], device='cuda:1')
        old_dest = PARENT_OUT / task / 'no_spectrum'
        old_record = base.check_seal(base.read_json(old_dest / 'RESULT.json'))
        for name, digest in old_record['files'].items():
            assert base.digest(old_dest / name) == digest
            inputs[str(old_dest / name)] = digest
        old_final = (torch.load(old_dest / 'training_final.pt', map_location='cpu', weights_only=True)
                     if task == 'MiDrug' else None)
        initial = None
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(986)
            torch.cuda.manual_seed(986)
            model, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            if initial is None:
                initial = experiment.cpu_copy(model.state_dict())
            else:
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value.cpu(), initial[name], atol=0., rtol=0.)
            del model, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats('cuda:1')
            dest = OUT / task / kind
            dest.mkdir(parents=True)
            if kind == 'full':
                model, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
                model.load_state_dict(torch.load(old_dest / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
                shutil.copy2(old_dest / 'model.pt', dest / 'model.pt')
                trace = base.read_json(old_dest / 'trace.json')
                detail = dict(new_short_neural_fit=False, adopted_parent_short_model=task == 'MiDrug',
                    trained_neural_model=task == 'MiDrug', training_steps=5 if task == 'MiDrug' else 0,
                    new_training_steps=0, parent_result=str(old_dest / 'RESULT.json'),
                    parent_result_sha256=base.digest(old_dest / 'RESULT.json'))
                baseline, old_xs, old_graphs, _ = parent.setup(a, parent.parameters(), 'no_spectrum', 'cuda:1')
                baseline.load_state_dict(model.state_dict(), strict=True)
                actual, expected = replay(model, xs, graphs, query), replay(baseline, old_xs, old_graphs, query)
                np.testing.assert_array_equal(actual, expected)
                detail['parent_prediction_max_abs'] = 0.
                del baseline, old_xs, old_graphs
            else:
                model, value, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
                assert value is None and all('anchor_composed' not in row for row in trace)
                detail.update(new_short_neural_fit=task == 'MiDrug', adopted_parent_short_model=False)
                torch.save(model.state_dict(), dest / 'model.pt')
                if task == 'MiDrug':
                    assert detail['training_steps'] == detail['gradient_evaluations'] == detail['optimizer_steps'] == 5
                    assert detail['persistence_diagnostics']['calls'] == 5
                    assert detail['persistence_diagnostics']['graphs'] == len(a['y_train'])
                    for name in ('cpu_rng', 'cuda_rng'):
                        torch.testing.assert_close(model.training_final_state[name], old_final[name], atol=0., rtol=0.)
                    assert all(int(state['step']) == 5 for state in model.training_final_state['optimizer']['state'].values())
                    torch.save(model.training_final_state, dest / 'training_final.pt')
                _, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
                actual = replay(model, xs, graphs, query)
            base.write_json(dest / 'trace.json', trace)
            loaded, lx, lg, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            reverse = replay(loaded, lx, lg, query.flip(1))[::-1]
            chunks = np.concatenate([replay(loaded, lx, lg, query[:, i:i + 31]) for i in range(0, query.shape[1], 31)])
            for value in (reverse, chunks):
                np.testing.assert_allclose(actual, value, atol=3e-5, rtol=3e-5)
            if task == 'LncDrug':
                np.testing.assert_array_equal(actual, np.zeros_like(actual))
                assert not detail['trained_neural_model'] and not trace
            np.savez_compressed(dest / 'replay.npz', query=query.cpu().numpy(), normal=actual, reverse=reverse, chunks=chunks)
            names = ['model.pt', 'trace.json', 'replay.npz']
            if (dest / 'training_final.pt').exists():
                names.append('training_final.pt')
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(actual - reverse).max(), abs(actual - chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                files={name: base.digest(dest / name) for name in names}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('PERSISTENCE_PAIRING_SMOKE', task, kind, record['replay_max_abs'], record['peak_allocated_gib'], flush=True)
            del model, loaded, xs, graphs, lx, lg
            gc.collect()
            torch.cuda.empty_cache()
    sources = [Path(__file__), Path(core.__file__), Path(parent.__file__), Path(experiment.__file__),
               ROOT / 'test_persistence_pairing_graph_v1.py']
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='persistence_pairing',
        core_parameters=core.parameters(), records=records, device='cuda:1', new_short_neural_fits=1,
        reused_parent_short_neural_models=1, identity_replays=2, new_formal_fits=0,
        inherited_fold_audit_count=40, new_fold_audits=0, inherited_fold_audit_source=str(inherited_path),
        inherited_fold_audit_sha256=base.digest(inherited_path), initialized_parameters_equal=True,
        final_control_rng_matches_parent_primary=True, primary_prediction_bitwise_identical=True,
        replay_uses_only_omitted_fold=True, new_performance_metrics_computed=False,
        new_evaluation_labels_read=False, prior_parent_outcomes_seen=True,
        input_sha256=inputs, source_sha256={str(path): base.digest(path) for path in sources}))
    base.write_json(destination, result)
    print('PERSISTENCE_PAIRING_SMOKE_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists() and not (OUT / 'FAILURE.json').exists():
            base.write_json(OUT / 'FAILURE.json', base.seal(dict(
                created_at_utc=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc(),
                source_sha256=base.digest(__file__), new_performance_metrics_computed=False,
                new_evaluation_labels_read=False, new_formal_fits=0)))
        raise
