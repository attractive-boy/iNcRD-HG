"""Three real MiDrug short fits, exact LncDrug identity, and support boundaries."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_muon_experiment_v1 as experiment
from improve import muon_graph_v1 as core
from improve import spectral_muon_v1 as optimizer_core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/spectral_muon_smoke_v1_20260916'


def main():
    destination = HERE / 'MUON_SMOKE_V1.json'
    if destination.exists() or OUT.exists():
        raise ValueError('Preserve previous smoke artifacts')
    OUT.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='spectral_muon', implementation_module='muon_graph_v1',
        source_sha256={'muon_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5, check_interval=5,
        optimizer_parameters=optimizer_core.parameters(),
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1)
    records, audits = [], []
    for task in ('MiDrug', 'LncDrug'):
        source = experiment.ROOT / 'results/anchor_residual_energy_v2_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        pairs = torch.as_tensor(a['pairs_train'][:, :97] + a['pair_offset'][:, None], device='cuda:1')
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
            dest = OUT / task / kind; dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
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
                if trace or detail['trained_neural_model'] or detail['training_steps'] or detail['gradient_evaluations']:
                    raise ValueError('Identity deployment was incorrectly treated as training')
            elif detail['gradient_evaluations'] != 5 or not detail['trained_neural_model']:
                raise ValueError('Wrong MiDrug gradient budget')
            equivalence = None
            if task == 'MiDrug' and kind == 'adamw':
                old_path = experiment.ROOT / 'results/friendly_sharpness_smoke_v1_20260916/MiDrug/erm/model.pt'
                old_state = torch.load(old_path, map_location='cuda:1', weights_only=True)
                differences = {name: float((value - old_state[name]).abs().max())
                               for name, value in model.state_dict().items()}
                equivalence = dict(source=str(old_path), source_sha256=base.digest(old_path),
                    maximum_weight_difference=max(differences.values()),
                    bitwise_equal=all(torch.equal(value, old_state[name])
                                      for name, value in model.state_dict().items()),
                    note='CPU exact trajectory test and unchanged architecture/optimizer establish the recipe match. CUDA training atomic roundoff may differ; formal adoption additionally requires exact saved-score replay.')
                if equivalence['maximum_weight_difference'] > 3e-5:
                    raise ValueError('Unexpected difference from prior ERM smoke weights')
            record = base.seal(dict(task=task, kind=kind, **detail, baseline_equivalence=equivalence,
                replay_max_abs=float(max(abs(value-replay).max(), abs(value-chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                source_sha256={str(path): base.digest(path) for path in
                    (Path(core.__file__), Path(optimizer_core.__file__), Path(experiment.__file__),
                     Path(__file__), dest / 'model.pt', dest / 'trace.json')}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record); print('MUON_SMOKE_RECORD', task, kind,
                detail['trained_neural_model'], detail['training_steps'], record['replay_max_abs'], flush=True)
            del model, loaded, xs, graphs
            gc.collect(); torch.cuda.empty_cache()
        paths = [('historical', source)] + [(split, experiment.ROOT /
            'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            for split in (61453, 7919, 104729)]
        for split, path in paths:
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            model, xs, graphs, folds = core.setup(training, core.parameters(), 'full', 'cpu')
            for fold in range(5):
                omitted = folds[fold].numpy()
                if task == 'LncDrug':
                    assert graphs[fold] is None and not list(model.parameters())
                    audits.append(dict(task=task, split=split, fold=fold,
                        identity=True, new_graph_constructed=False, new_neural_fit=False))
                    continue
                query = training['pairs_train'][:, omitted] + training['pair_offset'][:, None]
                physical = set(map(tuple, graphs[fold][0].numpy().T))
                overlap = sum(tuple(q) in physical or tuple(q[::-1]) in physical for q in query.T)
                if overlap:
                    raise ValueError('Held-out pair entered support')
                changed = {k: v.copy() for k, v in training.items()}
                changed['y_train'][omitted] = 1-changed['y_train'][omitted]
                _, _, other, _ = core.setup(changed, core.parameters(), 'full', 'cpu')
                for x, y in zip(graphs[fold], other[fold]):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, forbidden_overlap=int(overlap),
                    query_labels_invariant=True, identity=False))
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), records=records,
        fold_audits=audits, core_parameters=core.parameters(), optimizer_parameters=p['optimizer_parameters'],
        device='cuda:1', new_short_neural_fits=3, identity_replays=3, formal_fits=0,
        evaluation_labels_read=False, performance_metrics_computed=False, confirmed_innovations=0,
        source_sha256={str(path): base.digest(path) for path in
            (Path(__file__), Path(core.__file__), Path(optimizer_core.__file__), Path(experiment.__file__))}))
    base.write_json(destination, result)
    print('MUON_SMOKE_COMPLETE', 3, 3, len(audits), flush=True)


if __name__ == '__main__':
    main()
