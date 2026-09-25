"""Common explicit driver for new typed-graph candidate modules.

Model modules declare setup(a, protocol, kind, device). Fixed-step fitting is
strictly separated from historical checkpoint selection and rejects query data.
"""
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve import graph_prediction_precision as precision

TRAIN_KEYS = frozenset(('pairs_train', 'y_train', 'edges', 'relation', 'target_codes',
                       'node_counts', 'pair_offset', 'nd', 'folds', 'x_0', 'x_1', 'x_2', 'x_3'))


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if fixed_epochs is not None and set(a) - TRAIN_KEYS:
        raise ValueError('Fixed-step fitting accepts training inputs only')
    if kind not in p['kinds']:
        raise ValueError('Unknown registered graph control')
    library = importlib.import_module(p['executor_module'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, xs, graphs, folds = library.setup(a, p, kind, device)
    pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(a['y_train'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
    generator = torch.Generator(device=device).manual_seed(seed + 29015)
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one fitting step is required')
    query = (torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
             if fixed_epochs is None else None)
    best, state, selected_epoch = -np.inf, None, epochs
    trace, start = [], time.monotonic()
    for step in range(1, epochs + 1):
        fold = (step - 1) % len(graphs)
        ids = folds[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(xs, graphs[fold], pairs[:, ids])
        loss = F.binary_cross_entropy_with_logits(pred, labels[ids])
        if p.get('auxiliary_weight', 0):
            auxiliary, _ = model.auxiliary(xs, graphs[fold], generator)
            loss = loss + p['auxiliary_weight'] * auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite graph candidate loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if hasattr(model, 'update_teacher'):
            model.update_teacher()
        if step % p['check_interval'] == 0 or step == epochs:
            row = dict(epoch=step, loss=float(loss.detach()))
            if query is not None:
                value = precision.geometry_prediction(model, xs, graphs, query).cpu().numpy()
                composed = (1 - p['strong_weight']) * a['anchor_' + str(seed)] + p['strong_weight'] * value
                mm = common.base.metrics(a['y_val'], composed, a['pairs_val'])
                raw = common.base.metrics(a['y_val'], value, a['pairs_val'])
                row.update(anchor_composed=mm, standalone=raw)
                criterion = (mm['auc'] + mm['ap']) / 2
                if criterion > best:
                    best, selected_epoch = criterion, step
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if step % max(100, p['check_interval']) == 0:
                    print('TRAIN', kind, lr, seed, step, round(raw['auc'], 6), round(raw['ap'], 6),
                          round(time.monotonic() - start, 1), flush=True)
            trace.append(row)
    if state is not None:
        model.load_state_dict(state)
    prediction = None if query is None else precision.geometry_prediction(model, xs, graphs, query).cpu().numpy()
    detail = dict(epoch=selected_epoch, seconds=time.monotonic() - start,
                  parameters=sum(v.numel() for v in model.parameters()),
                  fresh_query_outcomes_used_for_fitting=False,
                  historical_checkpoint_selection=fixed_epochs is None)
    return model, prediction, trace, detail


def run(out, task, device, stage):
    out = Path(out)
    p = common.protocol(out)
    a = common.load(out / 'inputs' / task / 'arrays.npz')
    library = importlib.import_module(p['executor_module'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ph = common.base.digest(out / 'protocol.json')
    selection = None if stage == 'screen' else json.loads((out / 'SELECTION.json').read_text())
    if selection is not None and selection['protocol_sha256'] != ph:
        raise ValueError('Selection artifact changed')
    seeds = [p['screen_seed']] if stage == 'screen' else p['confirm_seeds']
    for kind in p['kinds']:
        for lr in p['lrs'] if stage == 'screen' else [selection['chosen'][kind]['lr']]:
            for seed in seeds:
                dest = out / 'runs' / task / kind / ('lr' + str(lr)) / ('s' + str(seed))
                if (dest / 'result.json').exists():
                    common.verify(dest, ph)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                model, pred, trace, detail = fit(a, p, kind, lr, seed, device)
                torch.save(model.state_dict(), dest / 'model.pt')
                common.base.write_json(dest / 'trace.json', trace)
                common.base.write_json(dest / 'training_detail.json', detail)
                loaded, xs, graphs, _ = library.setup(a, p, kind, device)
                loaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
                query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
                replay = precision.geometry_prediction(loaded, xs, graphs, query.flip(1)).cpu().numpy()[::-1]
                chunk = np.concatenate([precision.geometry_prediction(loaded, xs, graphs, query[:, i:i + 800]).cpu().numpy()
                                        for i in range(0, query.shape[1], 800)])
                for value in (replay, chunk):
                    np.testing.assert_allclose(pred, value, atol=3e-5, rtol=3e-5)
                scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=pred)
                for name in ('gnn', 'tabicl', 'anchor'):
                    weight = p['gnn_weight'] if name == 'gnn' else p['strong_weight']
                    scores[name] = a[name + '_' + str(seed)]
                    scores[name + '_composed'] = (1 - weight) * scores[name] + weight * pred
                np.savez_compressed(dest / 'scores.npz', **scores)
                result = dict(task=task, kind=kind, lr=lr, seed=seed, **detail, protocol_sha256=ph,
                    replay_max_abs=float(max(abs(pred - replay).max(), abs(pred - chunk).max())),
                    metrics={n: common.base.metrics(scores['y'], v, scores['pairs'])
                             for n, v in scores.items() if n not in ('y', 'pairs')},
                    files={n: common.base.digest(dest / n) for n in ('model.pt', 'trace.json', 'training_detail.json', 'scores.npz')})
                common.base.write_json(dest / 'result.json', result)
                print('VERIFIED', task, kind, lr, seed, detail['epoch'], result['metrics']['anchor_composed'], flush=True)
                del model, loaded, xs, graphs
                torch.cuda.empty_cache()
