"""Fixed-curvature geometric candidate; shared data/reference/selection audits.

The imported overlap experiment supplies only reference IO and reporting.
No variational/block model is trained or used by this geometry experiment.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import non_tabular_experiment as base
from improve import overlap_block_experiment as common
from improve import overlap_block_models as graph_lib
from improve import lorentz_graph_models as method

OUT = base.ROOT/'results/lorentz_graph_20260915'
PARENT = common.OUT
KINDS = ('full', 'euclidean', 'attention_only', 'distance_only')


def prepare(out, smoke=False):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a fresh geometry output directory')
    parent = common.protocol(PARENT)
    if parent['smoke']:
        raise ValueError('The shared reference source cannot be a smoke run')
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    own = ('lorentz_graph_experiment.py', 'lorentz_graph_models.py', 'test_lorentz_graph.py',
           'overlap_block_experiment.py', 'overlap_block_models.py', 'non_tabular_experiment.py')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=smoke,
             tasks=list(base.TASKS), kinds=list(KINDS), screen_seed=666, confirm_seeds=[0, 1],
             width=32 if smoke else 64, depth=1 if smoke else 3, heads=4, dropout=.1,
             radius_sq=1., expander_degree=0, overlay_seed=20260915, folds=5,
             epochs=2 if smoke else 400, check_interval=1 if smoke else 20,
             lrs=[.001] if smoke else [.001, .003], weight_decay=.001,
             strong_weight=.1, gnn_weight=.5, lnc_absolute_tolerance=.001,
             source_sha256={n:base.digest(base.ROOT/n) for n in own}, inputs={},
             literature=dict(title='Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space',
                             doi='10.1145/3637528.3672039', arxiv='2407.01290'),
             adaptation='Typed sparse attention with Lorentz chordal energies, normalized Lorentz centroids and residual midpoint, HRC-style spatial transformations and time reconstruction, separate RNA/drug distance readout. Fixed curvature -1; no expander, dense global attention, or official Hypformer checkpoint. Not a full reproduction of the paper linear-attention architecture.',
             controls='Euclidean attention/aggregation/readout; curved attention only; curved distance readout only. All controls have the same parameters and see the derived radial coordinate. Thus added nonlinear radial features are controlled.',
             boundary=parent['boundary'], selection=parent['selection'], gate=parent['gate'],
             success_scope=parent['success_scope'], no_original_test_access=True,
             no_original_outer_query_label_access=True, existing_default_replaced=False,
             parent_reference_protocol_sha256=base.digest(PARENT/'protocol.json'))
    for n in own:
        shutil.copy2(base.ROOT/n, out/'source_snapshot'/n)
    for task in p['tasks']:
        source = PARENT/'inputs'/task/'arrays.npz'
        dest = out/'inputs'/task
        dest.mkdir(parents=True)
        shutil.copy2(source, dest/'arrays.npz')
        for path in (source, dest/'arrays.npz'):
            p['inputs'][str(path)] = base.digest(path)
    base.write_json(out/'protocol.json', p)
    print('FROZEN', out, flush=True)


def setup(a, p, kind, device):
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max()+1)
    overlay = graph_lib.expander_edges(n, p['expander_degree'], p['overlay_seed'])
    graphs, indices = [], []
    for f in range(p['folds']):
        ix = np.flatnonzero(a['folds'] == f)
        edges, relations = graph_lib.masked_physical(a, ix)
        graphs.append(graph_lib.graph_inputs(edges, relations, overlay, n, nr, device))
        indices.append(torch.as_tensor(ix, device=device))
    xs = [torch.as_tensor(a['x_'+str(i)], device=device) for i in range(len(a['node_counts']))]
    model = method.LorentzGraph([x.shape[1] for x in xs], list(map(int,a['node_counts'])), nr,
                p['width'], p['depth'], p['heads'], p['dropout'],
                'lorentz' if kind == 'full' else kind, p['radius_sq']).to(device)
    return model, xs, graphs, indices


def train(a, p, kind, lr, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, xs, graphs, indices = setup(a, p, kind, device)
    pt = torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None], device=device)
    pv = torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None], device=device)
    y = torch.as_tensor(a['y_train'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
    best, best_epoch, state, trace = -np.inf, None, None, []
    start = time.monotonic()
    for epoch in range(1, p['epochs']+1):
        f = (epoch-1) % len(graphs)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(xs, graphs[f], pt[:,indices[f]])
        loss = F.binary_cross_entropy_with_logits(pred, y[indices[f]])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite geometric objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if epoch % p['check_interval'] == 0 or epoch == p['epochs']:
            candidate = method.predict(model, xs, graphs, pv).cpu().numpy()
            mixed = (1-p['strong_weight'])*a['anchor_'+str(seed)] + p['strong_weight']*candidate
            mm = base.metrics(a['y_val'], mixed, a['pairs_val'])
            raw = base.metrics(a['y_val'], candidate, a['pairs_val'])
            score = (mm['auc']+mm['ap'])/2
            trace.append(dict(epoch=epoch, loss=float(loss.detach()), anchor_composed=mm, standalone=raw))
            if score > best:
                best, best_epoch = score, epoch
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch % max(100,p['check_interval']) == 0:
                print('TRAIN', kind, lr, seed, epoch, round(raw['auc'],6), round(raw['ap'],6),
                      round(time.monotonic()-start,1), flush=True)
    model.load_state_dict(state)
    pred = method.predict(model, xs, graphs, pv).cpu().numpy()
    return model, pred, trace, dict(epoch=best_epoch, seconds=time.monotonic()-start,
        parameters=sum(v.numel() for v in model.parameters()))


def run(out, task, device, stage):
    out = Path(out)
    p = common.protocol(out)
    a = common.load(out/'inputs'/task/'arrays.npz')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol_hash = base.digest(out/'protocol.json')
    selected = None if stage == 'screen' else json.loads((out/'SELECTION.json').read_text())
    seeds = [p['screen_seed']] if stage == 'screen' else p['confirm_seeds']
    for kind in p['kinds']:
        lrs = p['lrs'] if stage == 'screen' else [selected['chosen'][kind]['lr']]
        for lr in lrs:
            for seed in seeds:
                dest = out/'runs'/task/kind/('lr'+str(lr))/('s'+str(seed))
                if (dest/'result.json').exists():
                    common.verify(dest, protocol_hash)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                model, pred, trace, detail = train(a, p, kind, lr, seed, device)
                torch.save(model.state_dict(), dest/'model.pt')
                replay_model, xs, graphs, _ = setup(a, p, kind, device)
                replay_model.load_state_dict(torch.load(dest/'model.pt', map_location=device,
                                                        weights_only=True), strict=True)
                pv = torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None], device=device)
                replay = method.predict(replay_model, xs, graphs, pv).cpu().numpy()
                reverse = method.predict(replay_model, xs, graphs, pv.flip(1)).cpu().numpy()[::-1]
                for value in (replay, reverse):
                    np.testing.assert_allclose(pred, value, atol=3e-5, rtol=3e-5)
                scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=pred)
                for ref in ('gnn', 'tabicl', 'anchor'):
                    scores[ref] = a[ref+'_'+str(seed)]
                    weight = p['gnn_weight'] if ref == 'gnn' else p['strong_weight']
                    scores[ref+'_composed'] = (1-weight)*scores[ref] + weight*pred
                np.savez_compressed(dest/'scores.npz', **scores)
                base.write_json(dest/'trace.json', trace)
                result = dict(task=task, kind=kind, lr=lr, seed=seed, **detail,
                    protocol_sha256=protocol_hash,
                    replay_max_abs=float(max(np.max(np.abs(pred-replay)), np.max(np.abs(pred-reverse)))),
                    metrics={name:base.metrics(scores['y'], value, scores['pairs'])
                             for name,value in scores.items() if name not in ('y','pairs')},
                    files={n:base.digest(dest/n) for n in ('model.pt','scores.npz','trace.json')})
                base.write_json(dest/'result.json', result)
                print('VERIFIED', task, kind, lr, seed, result['epoch'],
                      result['metrics']['anchor_composed'], flush=True)
                del model, replay_model, xs, graphs
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare','screen','select','confirm','report'))
    parser.add_argument('--out-dir', default=str(OUT))
    parser.add_argument('--task', choices=base.TASKS)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args()
    if a.command == 'prepare':
        prepare(a.out_dir, a.smoke)
    elif a.command in ('screen','confirm'):
        run(a.out_dir, a.task, a.device, a.command)
    elif a.command == 'select':
        common.select(a.out_dir)
    else:
        common.report(a.out_dir)


if __name__ == '__main__':
    main()
