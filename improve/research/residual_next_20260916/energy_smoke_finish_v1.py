"""Finish the smoke audit from its six saved models, without new fitting.

The original smoke script completed all training and replay checks, then used
the wrong module for seal() while packaging the report. It remains preserved.
"""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_residual_graph_v2 as experiment
from improve import energy_graph_v1 as energy
from improve import tabular_foundation as base
from improve.research.residual_next_20260916.energy_smoke_v1 import training_inputs


HERE = Path(__file__).resolve().parent


def main():
    dest = HERE / 'ENERGY_SMOKE_COMPLETE_V1.json'
    if dest.exists():
        raise ValueError('Preserve existing completed audit')
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = {'energy_graph_v1.py': base.digest(energy.__file__)}
    p = dict(smoke=True, family='energy', implementation_module='energy_graph_v1',
        core_parameters=energy.parameters(), kinds=list(energy.KINDS), source_sha256=source)
    rows, boundaries = [], []
    for task in ('MiDrug', 'LncDrug'):
        for split in ('historical', 61453, 7919, 104729):
            path = (experiment.prior.INPUTS / task / 'arrays.npz' if split == 'historical' else
                experiment.ROOT / 'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            a = training_inputs(path)
            n = int(a['node_counts'].sum())
            for fold in range(5):
                omitted = np.flatnonzero(a['folds'] == fold)
                graph = energy.context(a, omitted, 'cpu')
                query = np.concatenate([a['pairs_train'][:, omitted], a['pairs_val']], axis=1)
                query = query + a['pair_offset'][:, None]
                forbidden = np.r_[query[0] * n + query[1], query[1] * n + query[0]]
                sizes = {}
                for name in ('positive', 'negative'):
                    arcs = getattr(graph, name).arcs.numpy()
                    if np.isin(arcs[0] * n + arcs[1], forbidden).any():
                        raise ValueError('Query fact entered an energy graph')
                    sizes[name + '_arcs'] = int(arcs.shape[1])
                boundaries.append(dict(task=task, split=split, fold=fold, query_overlap=0,
                    negative_training_rows=len(graph.negative_training_indices), **sizes))
            if split != 'historical':
                continue
            a.pop('pairs_val')
            for kind in energy.KINDS:
                path = HERE / 'energy_smoke_models' / task / kind / 'model.pt'
                model, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
                model.load_state_dict(torch.load(path, map_location='cuda:1', weights_only=True))
                pairs = torch.as_tensor(a['pairs_train'][:, :101] + a['pair_offset'][:, None], device='cuda:1')
                direct = experiment.predict(model, xs, graphs, pairs, p).cpu().numpy()
                reverse = experiment.predict(model, xs, graphs, pairs.flip(1), p).cpu().numpy()[::-1]
                chunks = torch.cat([experiment.predict(model, xs, graphs, pairs[:, i:i + 37], p)
                                    for i in range(0, pairs.shape[1], 37)]).cpu().numpy()
                np.testing.assert_allclose(direct, reverse, atol=3e-5, rtol=3e-5)
                np.testing.assert_allclose(direct, chunks, atol=3e-5, rtol=3e-5)
                rows.append(dict(task=task, kind=kind, model_sha256=base.digest(path),
                    replay_max_abs=float(max(abs(direct - reverse).max(), abs(direct - chunks).max()))))
                print('REPLAYED_SAVED_ENERGY_SMOKE', task, kind, rows[-1]['replay_max_abs'], flush=True)
                del model, xs, graphs
                gc.collect()
                torch.cuda.empty_cache()
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        correction='Original smoke script called nonexistent non_tabular_experiment.seal after all six short fits. Saved models are retained and replayed; no model fit repeated.',
        source_sha256={**source, 'executor': base.digest(experiment.__file__),
            'failed_smoke': base.digest(HERE / 'energy_smoke_v1.py'), 'completion': base.digest(__file__)},
        original_smoke_fit_count=6, original_steps_per_fit=5, new_fit_count=0,
        saved_models_replayed=rows, boundary_fold_audits=boundaries,
        device='cuda:1', evaluation_labels_loaded=False, performance_metrics_computed=False,
        claimed_final_loss_or_runtime=False, confirmed_final_innovations=0))
    base.write_json(dest, result)
    print('ENERGY_SMOKE_AUDIT_COMPLETE', len(rows), 'saved models;', len(boundaries), 'graph boundaries', flush=True)


if __name__ == '__main__':
    main()
