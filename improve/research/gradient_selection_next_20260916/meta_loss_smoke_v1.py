"""Training-only, real-graph double-backward probe and support-boundary audit."""
from datetime import datetime, timezone
import gc
from pathlib import Path

import numpy as np
import torch

from improve import anchor_meta_loss_experiment_v1 as experiment
from improve import meta_loss_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT


def main():
    destination = HERE / 'META_LOSS_SMOKE_V2.json'
    out = ROOT / 'results/meta_loss_smoke_v2_20260916'
    if out.exists() or destination.exists():
        raise ValueError('Preserve previous smoke evidence')
    out.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='meta_loss', implementation_module='meta_loss_graph_v1',
        source_sha256={'meta_loss_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=10, check_interval=10,
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1)
    records, audits, reference_files = [], [], {}
    for task in ('MiDrug', 'LncDrug'):
        source = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        offset, manifest = experiment.shared.offset_arrays(ROOT / 'results/nested_anchor_v2_20260915/historical',
            task, 'historical', a)
        np.testing.assert_array_equal(offset, a['anchor_oof'])
        reference_files[str(manifest)] = base.digest(manifest)
        query = torch.as_tensor(a['pairs_train'][:, :97] + a['pair_offset'][:, None], device='cuda:1')
        rngs, episodes = [], []
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=10)
            dest = out / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            names = ['model.pt', 'trace.json']
            if detail['trained_neural_model']:
                torch.save(model.training_final_state, dest / 'training_final.pt')
                names.append('training_final.pt')
                rngs.append(model.training_final_state['cuda_rng'])
                episodes.append([r['episode_sha256'] for r in trace])
                final = torch.load(dest / 'training_final.pt', map_location='cpu', weights_only=True)
                assert final['task_optimizer']['step'] == 10
                for key, memory in final['memories'].items():
                    f = int(key.split('_')[0])
                    for rows in memory['rows'].values():
                        ids = rows[:, 5].numpy().astype(np.int64)
                        if not np.all(a['folds'][ids] == f):
                            raise ValueError('Feature memory crossed an omitted-fold boundary')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True), strict=True)
            normal = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
            reverse = experiment.predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, query[:, i:i + 31], p)
                .cpu().numpy() for i in range(0, query.shape[1], 31)])
            for actual in (reverse, chunks):
                np.testing.assert_allclose(normal, actual, atol=3e-5, rtol=3e-5)
            if task == 'LncDrug':
                np.testing.assert_array_equal(normal, np.zeros_like(normal))
                assert not detail['trained_neural_model'] and not trace
            summary = dict(max_direct_gradient_norm=max((r['direct_gradient_norm'] for r in trace), default=0.),
                max_hypergradient_norm=max((r['hypergradient_norm'] for r in trace), default=0.))
            record = base.seal(dict(task=task, kind=kind, **detail, **summary,
                replay_max_abs=float(max(abs(normal - reverse).max(), abs(normal - chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                files={name: base.digest(dest / name) for name in names}))
            base.write_json(dest / 'RESULT.json', record)
            records.append(record)
            print('META_LOSS_SMOKE', task, kind, detail['seconds'], record['replay_max_abs'],
                record['peak_allocated_gib'], summary, flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
        for rng in rngs[1:]:
            torch.testing.assert_close(rng, rngs[0], atol=0, rtol=0)
        if episodes and any(value != episodes[0] for value in episodes):
            raise ValueError('Mechanism arms sampled different episodes')
        paths = [('historical', source)] + [(str(split), ROOT /
            'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            for split in (61453, 7919, 104729)]
        for split, path in paths:
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            _, _, graphs, folds = core.setup(training, core.parameters(), 'full', 'cpu')
            cache = (ROOT / 'results/nested_anchor_v2_20260915/historical' if split == 'historical'
                     else ROOT / 'results/nested_anchor_v3_20260915/partitions')
            _, manifest = experiment.shared.offset_arrays(cache, task, split, training)
            reference_files[str(manifest)] = base.digest(manifest)
            for fold in range(5):
                if task == 'LncDrug':
                    assert graphs[fold] is None
                    audits.append(dict(task=task, split=split, fold=fold, identity=True,
                        new_graph_constructed=False, new_neural_fit=False))
                    continue
                omitted = folds[fold].numpy()
                inner, meta = core.episode_indices(omitted, np.random.default_rng(982 + fold))
                np.testing.assert_array_equal(np.sort(np.r_[inner, meta]), omitted)
                assert not np.intersect1d(inner, meta).size
                q = training['pairs_train'][:, omitted] + training['pair_offset'][:, None]
                n = int(training['node_counts'].sum())
                forbidden = np.r_[q[0] * n + q[1], q[1] * n + q[0]]
                graph = graphs[fold]
                edges = graph[0].numpy()
                if np.intersect1d(edges[0] * n + edges[1], forbidden).size:
                    raise ValueError('An inner or meta query fact remains visible')
                changed = {key: value.copy() for key, value in training.items()}
                changed['y_train'][omitted] = 1 - changed['y_train'][omitted]
                _, _, alternative, _ = core.setup(changed, core.parameters(), 'full', 'cpu')
                for left, right in zip(graph, alternative[fold]):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, identity=False,
                    forbidden_overlap=0, query_labels_invariant=True, disjoint_inner_meta=True,
                    same_omitted_fold_anchor_verified=True))
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__),
        ROOT / 'test_meta_loss_graph_v1.py', ROOT / 'anchor_priority_graph_v1.py',
        ROOT / 'sharpness_graph_v1.py', ROOT / 'overlap_block_models.py']
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family='meta_loss', core_parameters=core.parameters(), records=records, fold_audits=audits,
        device='cuda:1', new_short_neural_fits=4, identity_replays=4, formal_fits=0,
        performance_metrics_computed=False, evaluation_labels_read=False,
        episode_rng_matched=True, dropout_rng_matched=True,
        reference_manifest_sha256=reference_files,
        source_sha256={str(path): base.digest(path) for path in sources}))
    base.write_json(destination, result)
    print('META_LOSS_SMOKE_COMPLETE', len(records), len(audits), flush=True)


if __name__ == '__main__':
    main()
