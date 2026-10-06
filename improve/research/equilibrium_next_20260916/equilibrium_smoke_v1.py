"""Performance-blind training, adjoint and support-boundary checks."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_equilibrium_experiment_v1 as experiment
from improve import equilibrium_graph_v1 as core
from improve import tabular_foundation as base
from improve import overlap_block_models as graph_lib


HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT


def sparse_equal(first, second):
    for name in ('crow_indices', 'col_indices', 'values'):
        torch.testing.assert_close(getattr(first, name)(), getattr(second, name)(), atol=0, rtol=0)


def main():
    out = ROOT / 'results/equilibrium_smoke_v1_20260916'
    destination = HERE / 'EQUILIBRIUM_SMOKE_V1.json'
    if out.exists() or destination.exists():
        raise ValueError('Preserve existing smoke evidence')
    out.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='equilibrium', implementation_module='equilibrium_graph_v1',
        source_sha256={'equilibrium_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1)
    records, audits = [], []
    for task in ('MiDrug', 'LncDrug'):
        source = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        query = torch.as_tensor(a['pairs_train'][:, :97] + a['pair_offset'][:, None], device='cuda:1')
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=5)
            dest = out / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True))
            normal = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
            reverse = experiment.predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, query[:, i:i + 31], p)
                .cpu().numpy() for i in range(0, query.shape[1], 31)])
            for actual in (reverse, chunks):
                np.testing.assert_allclose(normal, actual, atol=3e-5, rtol=3e-5)
            if task == 'LncDrug':
                np.testing.assert_array_equal(normal, np.zeros_like(normal))
                assert not detail['trained_neural_model'] and not trace
            record = base.seal(dict(task=task, kind=kind, **detail,
                replay_max_abs=float(max(abs(normal - reverse).max(), abs(normal - chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                files={name: base.digest(dest / name) for name in ('model.pt', 'trace.json')}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('EQUILIBRIUM_SMOKE', task, kind, detail['seconds'], record['replay_max_abs'],
                  record['peak_allocated_gib'], detail['solver_diagnostics'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
        paths = [('historical', source)] + [(str(split), ROOT /
            'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            for split in (61453, 7919, 104729)]
        for split, path in paths:
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            model, xs, graphs, folds = core.setup(training, core.parameters(), 'full', 'cpu')
            for fold in range(5):
                if task == 'LncDrug':
                    assert graphs[fold] is None and not list(model.parameters())
                    audits.append(dict(task=task, split=split, fold=fold, identity=True,
                                       new_graph_constructed=False, new_neural_fit=False))
                    continue
                omitted = folds[fold].numpy()
                q = training['pairs_train'][:, omitted] + training['pair_offset'][:, None]
                n, relations = int(training['node_counts'].sum()), int(training['relation'].max()) + 1
                query_codes = np.r_[q[0] * n + q[1], q[1] * n + q[0]]
                for transition, _ in graphs[fold][0]:
                    indices = transition.to_sparse_coo().indices().numpy()
                    if np.intersect1d(indices[0] * n + indices[1], query_codes).size:
                        raise ValueError('Query pair is in a relation channel')
                changed = {key: value.copy() for key, value in training.items()}
                changed['y_train'][omitted] = 1. - changed['y_train'][omitted]
                edge, kind = graph_lib.masked_physical(changed, omitted)
                other = core.relation_channels(edge, kind, n, relations, 'cpu')
                for (first, first_t), (second, second_t) in zip(graphs[fold][0], other[0]):
                    sparse_equal(first, second)
                    sparse_equal(first_t, second_t)
                torch.testing.assert_close(graphs[fold][1], other[1], atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, identity=False,
                                   forbidden_overlap=0, query_labels_invariant=True))
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__),
               ROOT / 'test_equilibrium_graph_v1.py', ROOT / 'anchor_priority_graph_v1.py']
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family='equilibrium', core_parameters=core.parameters(), records=records, fold_audits=audits,
        device='cuda:1', new_short_neural_fits=2, identity_replays=2, formal_fits=0,
        performance_metrics_computed=False, evaluation_labels_read=False,
        source_sha256={str(path): base.digest(path) for path in sources}))
    base.write_json(destination, result)
    print('EQUILIBRIUM_SMOKE_COMPLETE', len(records), len(audits), flush=True)


if __name__ == '__main__':
    main()
