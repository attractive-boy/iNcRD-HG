"""Training-only short executions and support audits for differential attention."""
import argparse
from datetime import datetime, timezone
import gc
import importlib
from pathlib import Path

import numpy as np
import torch

from improve import anchor_priority_graph_v1 as experiment
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
MODULES = {'differential_attention': 'differential_graph_v1'}


def main(family):
    core = importlib.import_module('improve.' + MODULES[family])
    out = experiment.ROOT / ('results/' + family + '_smoke_v1_20260916')
    destination = HERE / (family.upper() + '_SMOKE_V1.json')
    if destination.exists() or out.exists():
        raise ValueError('Preserve previous smoke artifacts')
    out.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family=family, implementation_module=MODULES[family],
        source_sha256={MODULES[family] + '.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1)
    records, audits = [], []
    for task in ('MiDrug', 'LncDrug'):
        source = experiment.ROOT / 'results/anchor_residual_energy_v2_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        pairs = torch.as_tensor(a['pairs_train'][:, :97] + a['pair_offset'][:, None], device='cuda:1')
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
            dest = out / task / kind
            dest.mkdir(parents=True)
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
                assert not trace and not detail['trained_neural_model'] and detail['training_steps'] == 0
                assert detail['gradient_evaluations'] == 0
            else:
                assert detail['gradient_evaluations'] == 5 and detail['trained_neural_model']
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(value-replay).max(), abs(value-chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                source_sha256={str(path): base.digest(path) for path in
                    (Path(core.__file__), Path(experiment.__file__), Path(__file__),
                     dest / 'model.pt', dest / 'trace.json')}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('FEATURE_SMOKE_RECORD', family, task, kind, detail['training_steps'],
                  record['replay_max_abs'], record['peak_allocated_gib'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
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
                for original, altered in zip(graphs[fold], other[fold]):
                    torch.testing.assert_close(original, altered, atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, forbidden_overlap=int(overlap),
                    query_labels_invariant=True, identity=False))
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), records=records,
        fold_audits=audits, family=family, core_parameters=core.parameters(), device='cuda:1',
        new_short_neural_fits=len(core.KINDS), identity_replays=len(core.KINDS), formal_fits=0,
        evaluation_labels_read=False, performance_metrics_computed=False, confirmed_innovations=0,
        source_sha256={str(path): base.digest(path) for path in
            (Path(__file__), Path(core.__file__), Path(experiment.__file__))}))
    base.write_json(destination, result)
    print('FEATURE_SMOKE_COMPLETE', family, len(records), len(audits), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', required=True, choices=MODULES)
    main(parser.parse_args().family)
