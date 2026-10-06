"""Training-only resource/replay checks and forty support-boundary audits."""
from datetime import datetime, timezone
from pathlib import Path
import gc
import hashlib

import numpy as np
import torch

from improve import anchor_stochastic_augmentation_v1 as experiment
from improve import stochastic_augmentation_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/stochastic_augmentation_smoke_v1_20260916'


def main():
    result_path = HERE / 'STOCHASTIC_AUGMENTATION_SMOKE_V1.json'
    if result_path.exists() or OUT.exists():
        raise ValueError('Preserve completed or partial smoke artifacts')
    unit = base.check_seal(base.read_json(HERE / 'STOCHASTIC_AUGMENTATION_UNIT_V1.json'))
    if not unit['passed'] or unit['failures'] or unit['errors']:
        raise ValueError('Unit checks failed')
    for path, digest in unit['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Unit-checked source changed')
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='stochastic_augmentation', implementation_module='stochastic_augmentation_graph_v1',
        source_sha256={'stochastic_augmentation_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), private_rng_offset=29015,
        kinds=list(core.KINDS), epochs=5, check_interval=20, weight_decay=.001, strong_weight=.1)
    records, audits, inputs = [], [], {}
    for task in ('MiDrug', 'LncDrug'):
        source = experiment.ROOT / 'results/anchor_residual_energy_v2_20260916/inputs' / task / 'arrays.npz'
        inputs[str(source)] = base.digest(source)
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        pairs = torch.as_tensor(a['pairs_train'][:, :97] + a['pair_offset'][:, None], device='cuda:1')
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
            rng_hash = hashlib.sha256(torch.cuda.get_rng_state('cuda:1').numpy().tobytes()).hexdigest()
            dest = OUT / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            torch.save(model.training_final_state, dest / 'training_final.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True))
            value = experiment.predict(model, xs, graphs, pairs, p).cpu().numpy()
            replay = experiment.predict(loaded, xs, graphs, pairs.flip(1), p).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, pairs[:, i:i+50], p)
                                    .cpu().numpy() for i in range(0, pairs.shape[1], 50)])
            for actual in (replay, chunks):
                np.testing.assert_allclose(actual, value, atol=3e-5, rtol=3e-5)
            if task == 'LncDrug':
                np.testing.assert_array_equal(value, np.zeros_like(value))
                if trace or detail['trained_neural_model'] or detail['gradient_evaluations']:
                    raise ValueError('Identity was counted as a neural fit')
            else:
                if detail['gradient_evaluations'] != 10 or detail['training_steps'] != 5:
                    raise ValueError('Unmatched score/encoder gradient budget')
                state = torch.load(dest/'training_final.pt',map_location='cpu',weights_only=True)
                for group in ('encoder_optimizer','score_optimizer'):
                    if not state[group]['state'] or any(int(item['step']) != 5 for item in state[group]['state'].values()):
                        raise ValueError('Saved optimizer has inconsistent update budget')
                private_hash = hashlib.sha256(state['private_rng'].numpy().tobytes()).hexdigest()
                if any('anchor_composed' in row for row in trace):
                    raise ValueError('Training-only smoke consulted evaluation metrics')
            record = base.seal(dict(task=task, kind=kind, **detail,
                dropout_rng_sha256=rng_hash,private_rng_sha256=private_hash if task == 'MiDrug' else None,
                replay_max_abs=float(max(abs(value-replay).max(), abs(value-chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                files={n: base.digest(dest / n) for n in ('model.pt', 'training_final.pt', 'trace.json')}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('STOCHASTIC_AUGMENTATION_SMOKE_RECORD', task, kind, detail['training_steps'],
                  record['replay_max_abs'], record['peak_allocated_gib'], flush=True)
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
            for fold in range(5):
                omitted = folds[fold].numpy()
                if task == 'LncDrug':
                    if graphs[fold] is not None or list(model.parameters()):
                        raise ValueError('Identity unexpectedly constructed a graph')
                    audits.append(dict(task=task, split=split, fold=fold, identity=True,
                                       new_graph_constructed=False, new_neural_fit=False))
                    continue
                query = training['pairs_train'][:, omitted] + training['pair_offset'][:, None]
                physical = set(map(tuple, graphs[fold][0].numpy().T))
                overlap = sum(tuple(q) in physical or tuple(q[::-1]) in physical for q in query.T)
                if overlap:
                    raise ValueError('Held-out pair is visible in support')
                changed = {k: v.copy() for k, v in training.items()}
                changed['y_train'][omitted] = 1-changed['y_train'][omitted]
                _, _, other, _ = core.setup(changed, core.parameters(), 'full', 'cpu')
                for x, y in zip(graphs[fold], other[fold]):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, forbidden_overlap=int(overlap),
                                   query_labels_invariant=True, identity=False))
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__),
               experiment.ROOT / 'sharpness_graph_v1.py', experiment.ROOT / 'overlap_block_models.py',
               HERE / 'STOCHASTIC_AUGMENTATION_UNIT_V1.json']
    neural = [r for r in records if r['task'] == 'MiDrug']
    if len({r['dropout_rng_sha256'] for r in neural}) != 1 or len({r['private_rng_sha256'] for r in neural}) != 1:
        raise ValueError('Matched arms consumed different random-number streams')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), device='cuda:1',
        core_parameters=core.parameters(), private_rng_offset=29015,
        new_short_neural_fits=3, identity_replays=3, formal_fits=0, performance_metrics_computed=False,
        records=records, fold_audits=audits, input_sha256=inputs,
        source_sha256={str(path): base.digest(path) for path in sources},
        matched_update_and_gradient_budgets=True, original_test_source_access=False,
        confirmed_final_innovations=0))
    base.write_json(result_path, result)
    print('STOCHASTIC_AUGMENTATION_SMOKE_COMPLETE', result['sha256'], flush=True)


if __name__ == '__main__':
    main()
