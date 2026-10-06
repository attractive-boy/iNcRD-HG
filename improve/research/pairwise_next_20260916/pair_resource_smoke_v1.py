"""Two training-only steps on each real graph; exact resource and replay audit."""
from datetime import datetime, timezone
from pathlib import Path
import gc
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import triangular_pair_graph_v1 as core
from improve import anchor_residual_graph_v2 as experiment
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/triangular_pair_resource_smoke_v1_20260916'


def main():
    if OUT.exists() or (HERE / 'PAIR_RESOURCE_SMOKE_V1.json').exists():
        raise ValueError('Preserve previous resource smoke')
    OUT.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = core.parameters()
    fits, audits = [], []
    for task in ('MiDrug', 'LncDrug'):
        source = experiment.ROOT / 'results/anchor_residual_energy_v2_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device='cuda:1')
        offset = torch.as_tensor(a['anchor_oof'], dtype=torch.float64, device='cuda:1')
        labels = torch.as_tensor(a['y_train'], device='cuda:1')
        for kind in core.KINDS:
            torch.manual_seed(935); torch.cuda.manual_seed(935)
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, xs, graphs, folds = core.setup(a, p, kind, 'cuda:1')
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
            times, losses = [], []
            for step in range(2):
                torch.cuda.synchronize('cuda:1'); start = time.monotonic()
                model.train(); optimizer.zero_grad(set_to_none=True)
                ids = folds[step]
                value = model(xs, graphs[step], pairs[:, ids])
                loss = F.binary_cross_entropy_with_logits(offset[ids] + value.double(), labels[ids])
                loss.backward()
                if not torch.isfinite(loss) or any(v.grad is not None and not torch.isfinite(v.grad).all()
                                                   for v in model.parameters()):
                    raise FloatingPointError('Nonfinite smoke loss or gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                torch.cuda.synchronize('cuda:1')
                times.append(time.monotonic() - start); losses.append(float(loss.detach()))
                print('PAIR_RESOURCE_STEP', task, kind, step + 1, times[-1],
                      torch.cuda.max_memory_allocated('cuda:1') / 2**30, flush=True)
            dest = OUT / task / kind; dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            loaded, loaded_xs, loaded_graphs, _ = core.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True))
            query = pairs[:, folds[0]][:, :19]
            with torch.no_grad(), experiment.precision.deterministic_evaluation():
                model.eval(); loaded.eval()
                torch.cuda.synchronize('cuda:1'); start = time.monotonic()
                value = model(xs, graphs[0], query).double().cpu().numpy()
                torch.cuda.synchronize('cuda:1'); inference_seconds = time.monotonic() - start
                replay = loaded(loaded_xs, loaded_graphs[0], query.flip(1)).double().cpu().numpy()[::-1]
                chunks = np.concatenate([loaded(loaded_xs, loaded_graphs[0], query[:, i:i + 10])
                    .double().cpu().numpy() for i in range(0, query.shape[1], 10)])
            for actual in (replay, chunks):
                np.testing.assert_allclose(actual, value, atol=3e-5, rtol=3e-5)
            record = base.seal(dict(task=task, kind=kind, training_steps=2,
                nodes=sum(len(x) for x in xs), parameters=sum(v.numel() for v in model.parameters()),
                step_seconds=times, losses=losses, one_graph_inference_seconds=inference_seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved('cuda:1') / 2**30,
                replay_max_abs=float(max(abs(value-replay).max(), abs(value-chunks).max())),
                source_sha256={str(path): base.digest(path) for path in (Path(core.__file__),
                    Path(__file__), source, dest / 'model.pt')}, evaluation_outcomes_read=False,
                performance_metrics_computed=False, formal_fits=0))
            base.write_json(dest / 'RESULT.json', record)
            fits.append(record)
            print('PAIR_RESOURCE_FIT', task, kind, record, flush=True)
            del model, loaded, xs, graphs, folds, loaded_xs, loaded_graphs, optimizer, value, loss
            gc.collect(); torch.cuda.empty_cache()
        paths = [('historical', source)] + [(split, experiment.ROOT /
            'results/mechanism_validation_20260915/inputs' / task / str(split) / 'arrays.npz')
            for split in (61453, 7919, 104729)]
        for split, path in paths:
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            for fold in range(5):
                omitted = np.flatnonzero(training['folds'] == fold)
                graph = core.context(training, omitted, 'cpu')
                query = torch.as_tensor(training['pairs_train'][:, omitted] + training['pair_offset'][:, None])
                u, v = graph.global_to_local[query]
                overlap = int((graph.physical_relation[u, v] != 0).sum() +
                              graph.observed_zero[u, v].sum())
                if overlap:
                    raise ValueError('Omitted target fact appears in pair tokens')
                changed = {k: v.copy() for k, v in training.items()}
                changed['y_train'][omitted] = 1 - changed['y_train'][omitted]
                other = core.context(changed, omitted, 'cpu')
                for name in ('physical_relation', 'observed_zero', 'diagonal', 'global_to_local'):
                    torch.testing.assert_close(getattr(graph, name), getattr(other, name), atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, forbidden_overlap=overlap,
                    target_labels_invariant=True, nodes=len(graph.physical_relation)))
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), fits=fits,
        fold_audits=audits, core_parameters=p, device='cuda:1', evaluation_outcomes_read=False,
        performance_metrics_computed=False, formal_fits=0, confirmed_innovations=0,
        source_sha256={str(path): base.digest(path) for path in
                      (Path(core.__file__), Path(__file__))}))
    base.write_json(HERE / 'PAIR_RESOURCE_SMOKE_V1.json', result)
    print('PAIR_RESOURCE_COMPLETE', len(fits), len(audits), flush=True)


if __name__ == '__main__':
    main()
