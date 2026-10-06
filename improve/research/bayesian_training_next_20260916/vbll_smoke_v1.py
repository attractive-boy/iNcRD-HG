"""Training-only posterior smoke, prediction integration and support isolation."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_variational_lastlayer_experiment_v1 as experiment
from improve import variational_lastlayer_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/variational_lastlayer_smoke_v1_20260916'


def main():
    target = HERE / 'VBLL_SMOKE_V1.json'
    if target.exists() or OUT.exists():
        raise ValueError('Preserve completed or partial smoke artifacts')
    unit_path = HERE / 'VBLL_EXECUTOR_UNIT_V1.json'
    unit = base.check_seal(base.read_json(unit_path))
    if not unit['passed'] or unit['tests_run'] != 3:
        raise ValueError('Executor checks failed')
    for path, digest in unit['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Unit-checked source changed')
    if not (HERE/'vbll_core_unit_v1.log').read_text().rstrip().endswith('OK'):
        raise ValueError('Core checks failed')
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='variational_lastlayer', implementation_module='variational_lastlayer_graph_v1',
        source_sha256={'variational_lastlayer_graph_v1.py':base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=25, check_interval=20,
        weight_decay=.001, strong_weight=.1)
    records, audits, inputs = [], [], {}
    for task in ('MiDrug', 'LncDrug'):
        source = experiment.ROOT/'results/anchor_residual_energy_v2_20260916/inputs'/task/'arrays.npz'
        inputs[str(source)] = base.digest(source)
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        query = torch.as_tensor(a['pairs_train'][:, :97]+a['pair_offset'][:, None], device='cuda:1')
        anchor = torch.as_tensor(a['anchor_oof'][:97], device='cuda:1', dtype=torch.float64)
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=25)
            dest = OUT/task/kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest/'model.pt')
            base.write_json(dest/'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest/'model.pt', map_location='cuda:1', weights_only=True))
            components = experiment.predict_components(model, xs, graphs, query, p, anchor)
            value = components['residual'].cpu().numpy()
            reverse = experiment.predict(loaded, xs, graphs, query.flip(1), p, anchor.flip(0)).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, query[:, i:i+50], p, anchor[i:i+50])
                                     .cpu().numpy() for i in range(0, query.shape[1], 50)])
            reference = experiment.predict(loaded, xs, graphs, query, p, anchor, 128).cpu().numpy()
            for actual in (reverse, chunks):
                np.testing.assert_allclose(actual, value, atol=3e-5, rtol=3e-5)
            integration_error = float(abs(reference-value).max())
            if integration_error > p['core_parameters']['quadrature_tolerance']:
                raise ValueError('Predictive quadrature exceeded the frozen error tolerance')
            if task == 'MiDrug':
                if detail['training_steps'] != 25 or detail['gradient_evaluations'] != 25:
                    raise ValueError('Training budget mismatch')
                if kind == 'full' and model.last.lower.norm() == 0:
                    raise ValueError('Dense covariance did not train')
            elif trace or detail['trained_neural_model'] or detail['gradient_evaluations']:
                raise ValueError('Identity is not a neural fit')
            if task == 'LncDrug':
                np.testing.assert_array_equal(value, np.zeros_like(value))
            if any('anchor_composed' in row for row in trace):
                raise ValueError('Training-only smoke consulted evaluation outcomes')
            np.savez_compressed(dest/'prediction_audit.npz', residual=value, reverse=reverse,
                chunks=chunks, reference=reference, mean_residual=components['mean_residual'].cpu().numpy(),
                variance=components['variance'].cpu().numpy())
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(value-reverse).max(), abs(value-chunks).max())),
                quadrature_max_abs=integration_error,
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1')/2**30,
                files={n:base.digest(dest/n) for n in ('model.pt','trace.json','prediction_audit.npz')}))
            base.write_json(dest/'RESULT.json', record)
            records.append(record)
            print('VBLL_SMOKE_RECORD',task,kind,detail['training_steps'],integration_error,flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
        paths = [('historical',source)] + [(split,experiment.ROOT/'results/mechanism_validation_20260915/inputs'/task/str(split)/'arrays.npz')
                                         for split in (61453,7919,104729)]
        for split,path in paths:
            inputs[str(path)] = base.digest(path)
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            model, xs, graphs, folds = core.setup(training, core.parameters(), 'full', 'cpu')
            for fold in range(5):
                omitted = folds[fold].numpy()
                if task == 'LncDrug':
                    if graphs[fold] is not None or list(model.parameters()):
                        raise ValueError('Identity unexpectedly constructed a graph or posterior')
                    audits.append(dict(task=task,split=split,fold=fold,identity=True,new_graph_constructed=False))
                    continue
                pairs = training['pairs_train'][:,omitted]+training['pair_offset'][:,None]
                physical = set(map(tuple,graphs[fold][0].numpy().T))
                overlap = sum(tuple(q) in physical or tuple(q[::-1]) in physical for q in pairs.T)
                if overlap:
                    raise ValueError('Query pair visible in support graph')
                changed = {k:v.copy() for k,v in training.items()}
                changed['y_train'][omitted] = 1-changed['y_train'][omitted]
                _, _, other, _ = core.setup(changed, core.parameters(), 'full', 'cpu')
                for x,y in zip(graphs[fold],other[fold]):
                    torch.testing.assert_close(x,y,atol=0,rtol=0)
                audits.append(dict(task=task,split=split,fold=fold,identity=False,
                                   forbidden_overlap=int(overlap),query_labels_invariant=True))
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__), unit_path,
               HERE/'vbll_core_unit_v1.log',experiment.ROOT/'sharpness_graph_v1.py',
               experiment.ROOT/'overlap_block_models.py',experiment.ROOT/'overlap_block_experiment.py']
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),device='cuda:1',
        core_parameters=core.parameters(),new_short_neural_fits=3,identity_replays=3,
        formal_fits=0,performance_metrics_computed=False,records=records,fold_audits=audits,
        input_sha256=inputs,source_sha256={str(path):base.digest(path) for path in sources},
        original_test_source_access=False,confirmed_final_innovations=0))
    base.write_json(target,result)
    print('VBLL_SMOKE_COMPLETE',result['sha256'],flush=True)


if __name__ == '__main__':
    main()
