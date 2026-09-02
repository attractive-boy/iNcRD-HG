# The author's own hyperparameter search, redone properly.
#
# `find_best_params.py` sweeps lr x epochs x channels x aggr and selects on the
# VALIDATION fold by (AUC+AUPR)/2.  So the paper's S_val row is not a single
# model's score -- it is the argmax of an 840-cell sweep, and the only honest
# way to move it is to run the same sweep better.  Three things make the
# author's version noisier than it needs to be:
#
#   * it scores one seed (666), so the argmax is partly a seed draw;
#   * it re-trains from scratch for each value of `epochs`, and reads the LAST
#     epoch of each run, so the epoch axis costs 6x and is still coarse;
#   * it never sees `posw2`, the one mechanism in improve/RESULTS.md that is
#     positive on both ncRNA types.
#
# Here the epoch axis is free -- one run to `--max-epochs` records every epoch,
# so the author's {51,101,...,301} checkpoints and the best-epoch rule both
# fall out of the same trace -- the selection statistic is a multi-seed mean,
# and the arm is a fifth axis.
#
# Both selection rules are recorded per config so the choice can be stated:
#   ep{E}_*  -- the value at epoch E, the author's rule (`epochs` = E)
#   bv_*     -- the value at the best-validation epoch, which is what
#               predict_association.py's own printout tracks
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.model import Model
from improve.run import ARMS, TASKS, compute_loss, fwd, load_splits, set_seed

# find_best_params.py's grid, verbatim.  `epoch_options` is not a search axis
# here -- it is read off the trace.
LR_OPTIONS = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
EPOCH_CKPTS = [51, 101, 151, 201, 251, 301]
CHANNEL_OPTIONS = [32, 64, 128, 256, 512]
AGGR_OPTIONS = ['mean', 'sum', 'max', 'min']


@torch.no_grad()
def score(model, data, key):
    model.eval()
    pred = fwd(model, data, data[key].edge_label_index).cpu().numpy()
    y = data[key].edge_label.cpu().numpy()
    return roc_auc_score(y, pred), average_precision_score(y, pred)


def trace_one(splits, task, arm, lr, channels, aggr, seed, max_epochs, device,
              layers=2, dropout=0.0, weight_decay=0.0, optimizer='rmsprop'):
    """Train one config and return the full per-epoch val/test trace."""
    cfg = ARMS[arm]
    t = TASKS[task]
    key = (t['rna'], task, 'drug')
    set_seed(seed)
    train_data, val_data, test_data = splits

    model = Model(train_data, channels, aggr, t['rna'],
                  decoder=cfg.get('decoder', 'concat'),
                  encoder=cfg.get('encoder', 'hetero'),
                  layers=layers, dropout=dropout).to(device)
    # RMSprop(alpha=0.99) is the released optimiser and the default here.
    if optimizer == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    elif optimizer == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    else:
        opt = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99,
                                  weight_decay=weight_decay)

    label_index = train_data[key].edge_label_index
    label = train_data[key].edge_label
    pos_index = label_index[:, label == 1]
    n_pos = pos_index.size(1)
    n_src = train_data[t['rna']].num_nodes
    n_dst = train_data['drug'].num_nodes

    rows = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        if cfg.get('neg_resample'):
            neg = negative_sampling(
                pos_index, num_nodes=(n_src, n_dst),
                num_neg_samples=cfg.get('neg_ratio', 2) * n_pos,
                method='sparse')
            cur_i = torch.cat([pos_index, neg], dim=1)
            cur_l = torch.cat([torch.ones(n_pos, device=device),
                               torch.zeros(neg.size(1), device=device)])
        else:
            cur_i, cur_l = label_index, label
        opt.zero_grad()
        loss = compute_loss(fwd(model, train_data, cur_i), cur_l, cfg)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        va, vp = score(model, val_data, key)
        ta, tp = score(model, test_data, key)
        rows.append((epoch, va, vp, ta, tp))
    return pd.DataFrame(rows, columns=['epoch', 'val_auc', 'val_aupr',
                                       'test_auc', 'test_aupr'])


def summarise(h, task, arm, lr, channels, aggr, seed, **extra):
    """Collapse a trace to one row: the author's epoch checkpoints + best-val."""
    r = dict(task=task, arm=arm, lr=lr, channels=channels, aggr=aggr, seed=seed,
             n_epochs=len(h), **extra)
    if len(h) == 0:
        return r
    for e in EPOCH_CKPTS:
        # find_best_params.py runs `for epoch in range(1, E)`, i.e. E-1 steps.
        row = h[h['epoch'] == e - 1]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        for m in ('val_auc', 'val_aupr', 'test_auc', 'test_aupr'):
            r[f'ep{e}_{m}'] = row[m]
    # Best-validation-epoch rule, each metric read at its own best epoch --
    # the same rule improve/run.py and improve/grid.py use.
    ba = h.loc[h['val_auc'].idxmax()]
    bp = h.loc[h['val_aupr'].idxmax()]
    r.update(bv_epoch=int(ba['epoch']), bv_val_auc=ba['val_auc'],
             bv_test_auc=ba['test_auc'],
             bp_epoch=int(bp['epoch']), bv_val_aupr=bp['val_aupr'],
             bv_test_aupr=bp['test_aupr'])
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--arms', nargs='+', default=['base', 'posw2'])
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--lrs', nargs='+', type=float, default=LR_OPTIONS)
    p.add_argument('--channels', nargs='+', type=int, default=CHANNEL_OPTIONS)
    p.add_argument('--aggrs', nargs='+', default=AGGR_OPTIONS)
    # Axes find_best_params.py does not have.  Two layers, no regularisation
    # and RMSprop are untested constants in the release, not tuned choices.
    p.add_argument('--layers', nargs='+', type=int, default=[2])
    p.add_argument('--dropouts', nargs='+', type=float, default=[0.0])
    p.add_argument('--weight-decays', nargs='+', type=float, default=[0.0])
    p.add_argument('--optimizers', nargs='+', default=['rmsprop'],
                   choices=['rmsprop', 'adam', 'adamw'])
    p.add_argument('--max-epochs', type=int, default=301)
    p.add_argument('--out', required=True)
    p.add_argument('--variant', default='full',
                   choices=['full', 'wo_lncmi', 'wo_attribute', 'wo_both'],
                   help='ablation_predict_association.py evaluates its variants '
                        "at the FULL model's hyperparameters (lines 155-164), "
                        'which overstates what the removed component was worth. '
                        'This re-searches for each variant.')
    p.add_argument('--trace-dir', default=None,
                   help='optional: dump the full per-epoch trace per config')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # The features are identical across arms here, so the splits load once.
    if args.variant == 'full':
        splits = load_splits(args.run_dir, args.task, {}, device)
    else:
        from improve.final import load_variant
        splits = load_variant(args.run_dir, args.task,
                              TASKS[args.task]['rna'], args.variant, device)

    combos = list(itertools.product(args.arms, args.lrs, args.channels,
                                    args.aggrs, args.layers, args.dropouts,
                                    args.weight_decays, args.optimizers))
    rows, t0 = [], time.time()
    for i, (arm, lr, ch, aggr, nl, dp, wd, optn) in enumerate(combos, 1):
        extra = dict(layers=nl, dropout=dp, weight_decay=wd, optimizer=optn,
                     variant=args.variant)
        for seed in args.seeds:
            h = trace_one(splits, args.task, arm, lr, ch, aggr, seed,
                          args.max_epochs, device, layers=nl, dropout=dp,
                          weight_decay=wd, optimizer=optn)
            rows.append(summarise(h, args.task, arm, lr, ch, aggr, seed,
                                  **extra))
            if args.trace_dir:
                os.makedirs(args.trace_dir, exist_ok=True)
                h.to_csv(f'{args.trace_dir}/{args.task}_{arm}_lr{lr}_ch{ch}_'
                         f'{aggr}_l{nl}_d{dp}_w{wd}_{optn}_s{seed}.csv',
                         index=False)
        r = rows[-1]
        el = time.time() - t0
        print(f"[{i}/{len(combos)}] {arm:9s} lr={lr:<8g} ch={ch:<5d} "
              f"{aggr:5s} L{nl} d{dp} w{wd:g} {optn:7s} "
              f"bv_val={r.get('bv_val_auc', float('nan')):.4f}/"
              f"{r.get('bv_val_aupr', float('nan')):.4f} "
              f"[{el/60:.1f}m, eta {el/i*(len(combos)-i)/60:.0f}m]", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
