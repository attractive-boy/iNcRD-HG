"""All-fold training-only smoke, reload replay, and positive-path boundary audit."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_residual_graph_v2 as experiment
from improve import loopy_path_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/loopy_path_smoke_v1_20260916'


def main():
    destination = HERE / 'LOOPY_SMOKE_V1.json'
    if destination.exists() or OUT.exists():
        raise ValueError('Preserve previous smoke')
    OUT.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='loopy_path', implementation_module='loopy_path_graph_v1',
        source_sha256={'loopy_path_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1)
    fits, audits = [], []
    for task in ('MiDrug', 'LncDrug'):
        source = experiment.ROOT / 'results/anchor_residual_energy_v2_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        # Use training pair identities for replay; no evaluation array is loaded.
        pairs = torch.as_tensor(a['pairs_train'][:, :97] + a['pair_offset'][:, None], device='cuda:1')
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 967, 'cuda:1', fixed_epochs=5)
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
            record = base.seal(dict(task=task, kind=kind, steps=5,
                replay_max_abs=float(max(abs(value-replay).max(), abs(value-chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                training_seconds=detail['seconds'], timings_may_include_concurrent_pair_study=True,
                paths_per_fold=[len(g.paths) for g in graphs],
                source_sha256={str(path): base.digest(path) for path in
                    (Path(core.__file__), Path(__file__), dest / 'model.pt', dest / 'trace.json')}))
            base.write_json(dest / 'RESULT.json', record)
            fits.append(record); print('LOOPY_SMOKE_FIT', record, flush=True)
            del model, loaded, xs, graphs
            gc.collect(); torch.cuda.empty_cache()
        paths = [('historical', source)] + [(split, experiment.ROOT /
            'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            for split in (61453, 7919, 104729)]
        for split, path in paths:
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            n = int(training['node_counts'].sum())
            for fold in range(5):
                omitted = np.flatnonzero(training['folds'] == fold)
                graph = core.context(training, omitted, 'cpu')
                q = training['pairs_train'][:, omitted] + training['pair_offset'][:, None]
                forbidden = np.r_[q[0]*n+q[1], q[1]*n+q[0]]
                physical = graph.edges.numpy()
                cycles = np.column_stack((graph.centers.numpy(), graph.paths.numpy(), graph.centers.numpy()))
                cycle_edges = (cycles[:, :-1]*n+cycles[:, 1:]).ravel()
                overlap = int(np.isin(cycle_edges, forbidden).sum() +
                              np.isin(physical[0]*n+physical[1], forbidden).sum())
                if overlap:
                    raise ValueError('Query edge occurs in support paths or physical graph')
                changed = {k: v.copy() for k, v in training.items()}
                changed['y_train'][omitted] = 1-changed['y_train'][omitted]
                other = core.context(changed, omitted, 'cpu')
                for name in ('edges', 'relation', 'counts', 'centers', 'paths', 'path_degree'):
                    torch.testing.assert_close(getattr(graph, name), getattr(other, name), atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, forbidden_overlap=overlap,
                    query_labels_invariant=True, ordered_rooted_paths=len(graph.paths),
                    source_sha256=base.digest(path)))
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), fits=fits,
        fold_audits=audits, core_parameters=core.parameters(), evaluation_labels_read=False,
        performance_metrics_computed=False, device='cuda:1', formal_fits=0, confirmed_innovations=0,
        source_sha256={str(path): base.digest(path) for path in (Path(__file__), Path(core.__file__))}))
    base.write_json(destination, result)
    print('LOOPY_SMOKE_COMPLETE', len(fits), len(audits), flush=True)


if __name__ == '__main__':
    main()
