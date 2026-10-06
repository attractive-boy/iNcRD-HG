"""Audit all exposed training boundaries and six CUDA-1 short fits."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_residual_graph_v2 as experiment
from improve import energy_graph_v1 as energy


HERE = Path(__file__).resolve().parent
base = experiment.common.base


def training_inputs(path):
    with np.load(path) as z:
        # Evaluation pair identities are used only for boundary audits.
        a = {k: z[k] for k in z.files if k in experiment.TRAIN_KEYS or k == 'pairs_val'}
    return a


def main():
    output = HERE / 'ENERGY_SMOKE_V1.json'
    if output.exists():
        raise ValueError('Preserve existing smoke result')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    audits, fits, source_hashes = [], [], {}
    source_hashes['energy_graph_v1.py'] = base.digest(energy.__file__)
    cells = []
    for task in ('MiDrug', 'LncDrug'):
        cells.append((task, 'historical', experiment.prior.INPUTS / task / 'arrays.npz'))
        for split in (61453, 7919, 104729):
            cells.append((task, split, experiment.ROOT / 'results/mechanism_validation_20260915/inputs' /
                          task / str(split) / 'arrays.npz'))
    for task, split, path in cells:
        a = training_inputs(path)
        n = int(a['node_counts'].sum())
        for fold in range(5):
            omitted = np.flatnonzero(a['folds'] == fold)
            graph = energy.context(a, omitted, 'cpu')
            queries = np.concatenate([a['pairs_train'][:, omitted], a['pairs_val']], axis=1)
            queries = queries + a['pair_offset'][:, None]
            forbidden = np.r_[queries[0] * n + queries[1], queries[1] * n + queries[0]]
            counts = {}
            for name in ('positive', 'negative'):
                arcs = getattr(graph, name).arcs.numpy()
                overlap = np.isin(arcs[0] * n + arcs[1], forbidden).sum()
                if overlap:
                    raise ValueError('Query facts entered an energy graph')
                counts[name + '_arcs'] = int(arcs.shape[1])
            audits.append(dict(task=task, split=split, fold=fold,
                query_overlap=0, negative_training_rows=len(graph.negative_training_indices), **counts))
        if split != 'historical':
            continue
        a.pop('pairs_val')
        a['anchor_oof'], _ = experiment.offset_arrays(experiment.nested.DEFAULT, task, split, a)
        p = dict(smoke=True, family='energy', implementation_module='energy_graph_v1',
            core_parameters=energy.parameters(), kinds=list(energy.KINDS), source_sha256=source_hashes,
            epochs=5, check_interval=5, weight_decay=.001, private_rng_offset=29015, strong_weight=.1)
        for kind in energy.KINDS:
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 777, 'cuda:1', fixed_epochs=5)
            dest = HERE / 'energy_smoke_models' / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True))
            pairs = torch.as_tensor(a['pairs_train'][:, :101] + a['pair_offset'][:, None], device='cuda:1')
            direct = experiment.predict(model, xs, graphs, pairs, p).cpu().numpy()
            reverse = experiment.predict(loaded, xs, graphs, pairs.flip(1), p).cpu().numpy()[::-1]
            chunks = torch.cat([experiment.predict(loaded, xs, graphs, pairs[:, i:i + 37], p)
                                for i in range(0, pairs.shape[1], 37)]).cpu().numpy()
            np.testing.assert_allclose(direct, reverse, atol=3e-5, rtol=3e-5)
            np.testing.assert_allclose(direct, chunks, atol=3e-5, rtol=3e-5)
            row = dict(task=task, kind=kind, detail=detail, final_loss=trace[-1]['loss'],
                replay_max_abs=float(max(abs(direct - reverse).max(), abs(direct - chunks).max())),
                model_sha256=base.digest(dest / 'model.pt'))
            fits.append(row)
            print('ENERGY_SMOKE', task, kind, row['replay_max_abs'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={**source_hashes, 'script': base.digest(__file__),
                       'executor': base.digest(experiment.__file__)},
        device='cuda:1', boundary_cells=len(cells), boundary_fold_audits=audits,
        smoke_fits=fits, formal_fits=0, evaluation_labels_loaded=False,
        performance_metrics_computed=False, confirmed_final_innovations=0))
    base.write_json(output, result)
    print('ENERGY_SMOKE_COMPLETE', len(fits), 'short fits;', len(audits), 'boundary audits', flush=True)


if __name__ == '__main__':
    main()
