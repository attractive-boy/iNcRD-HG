# Ensemble across CONFIGURATIONS, not just across seeds.
#
# Every ensemble in this repo so far averaged 10 seeds of ONE configuration, and
# it was worth +0.0002 to +0.0006 AUC.  That is the expected return: 10 runs of
# the same architecture on the same fold make correlated errors, so averaging
# them cancels almost nothing.  Models from *different* configurations -- widths,
# learning rates, dropout rates, aggregations -- fail on different edges, and
# that is where an ensemble actually pays.
#
# Two combiners are reported:
#
#   uniform -- z-score each member's scores on the fold, average.  No selection,
#              so nothing is fitted to the fold it is scored on.
#   greedy  -- Caruana et al. 2004 ensemble selection: repeatedly add the member
#              (with replacement) that most improves VALIDATION AUC.  This is
#              fitted on validation, so its validation figure is optimistic by
#              construction and only the test column is honest.  Both are
#              printed side by side for exactly that reason.
#
# The library is built from an hpsearch CSV: take the top-K configurations by
# the validation criterion and train each with `--seeds` seeds.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.final import PAPER, load_variant
from improve.hpselect import robust_rule, set_keys
from improve.run import ARMS, TASKS, fwd, set_seed
from improve.model import Model

CFG_COLS = ['arm', 'lr', 'channels', 'aggr', 'layers', 'dropout',
            'weight_decay', 'optimizer']
DEFAULTS = dict(layers=2, dropout=0.0, weight_decay=0.0, optimizer='rmsprop')


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def rk(v):
    """Rank-normalise to [0,1].  Scale-free, so a member with a heavy-tailed
    score distribution cannot dominate the average the way it can under z."""
    order = v.argsort().argsort().astype(np.float64)
    return order / max(len(v) - 1, 1)


SCORER = {'auc': roc_auc_score, 'aupr': average_precision_score}


def capture_scores(splits, task, cfg, seed, device, max_epochs=301,
                   snapshots=3):
    """Train one member; return score vectors at several selected epochs.

    Returns `{'auc': [one member], 'aupr': [up to `snapshots` members]}`.

    Capturing per metric matters: reading AUPR off the best-val-AUC epoch costs
    ~0.007 val AUPR on LncDrug, because the two metrics peak at different
    epochs.  The AUPR snapshots are the top-`snapshots` validation-AUPR epochs
    of the same run -- extra ensemble members at zero extra training, and
    decorrelated from each other because they are different points on the
    trajectory (snapshot ensembling, Huang et al. 2017).
    """
    arm = cfg['arm']
    acfg = ARMS[arm]
    t = TASKS[task]
    key = (t['rna'], task, 'drug')
    set_seed(seed)
    train_data, val_data, test_data = splits
    from torch_geometric.utils import negative_sampling
    from improve.run import compute_loss

    model = Model(train_data, int(cfg['channels']), cfg['aggr'], t['rna'],
                  decoder=acfg.get('decoder', 'concat'),
                  encoder=acfg.get('encoder', 'hetero'),
                  layers=int(cfg['layers']), dropout=float(cfg['dropout'])
                  ).to(device)
    wd, lr = float(cfg['weight_decay']), float(cfg['lr'])
    if cfg['optimizer'] == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif cfg['optimizer'] == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99,
                                  weight_decay=wd)

    li = train_data[key].edge_label_index
    lb = train_data[key].edge_label
    pos_index = li[:, lb == 1]
    n_pos = pos_index.size(1)
    n_src, n_dst = train_data[t['rna']].num_nodes, train_data['drug'].num_nodes
    y = {f: d[key].edge_label.cpu().numpy()
         for f, d in (('val', val_data), ('test', test_data))}

    def test_scores():
        return fwd(model, test_data,
                   test_data[key].edge_label_index).cpu().numpy()

    best_auc, cap_auc, pool = -1.0, None, []
    for _ in range(1, max_epochs + 1):
        model.train()
        if acfg.get('neg_resample'):
            neg = negative_sampling(
                pos_index, num_nodes=(n_src, n_dst),
                num_neg_samples=acfg.get('neg_ratio', 2) * n_pos,
                method='sparse')
            cur_i = torch.cat([pos_index, neg], dim=1)
            cur_l = torch.cat([torch.ones(n_pos, device=device),
                               torch.zeros(neg.size(1), device=device)])
        else:
            cur_i, cur_l = li, lb
        opt.zero_grad()
        loss = compute_loss(fwd(model, train_data, cur_i), cur_l, acfg)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            sv = fwd(model, val_data,
                     val_data[key].edge_label_index).cpu().numpy()
            a = roc_auc_score(y['val'], sv)
            if a > best_auc:
                best_auc, cap_auc = a, {'val': sv, 'test': test_scores()}
            u = average_precision_score(y['val'], sv)
            # Keep the top-`snapshots` validation-AUPR epochs.  Test is only
            # computed when an epoch actually enters the pool.
            if len(pool) < snapshots or u > min(p[0] for p in pool):
                pool.append((u, {'val': sv, 'test': test_scores()}))
                pool.sort(key=lambda p: -p[0])
                del pool[snapshots:]
    return {'auc': [cap_auc] if cap_auc else [],
            'aupr': [p[1] for p in pool]}, y


def greedy_select(members, y_val, n_steps, metric='auc'):
    """Caruana et al. 2004 ensemble selection with replacement, maximising the
    named VALIDATION metric.  Fitted on validation by construction."""
    score = SCORER[metric]
    chosen, cur = [], None
    for _ in range(n_steps):
        best_i, best_s, best_v = None, -1.0, None
        for i, m in enumerate(members):
            cand = m if cur is None else (cur * len(chosen) + m) / (len(chosen) + 1)
            s = score(y_val, cand)
            if s > best_s:
                best_i, best_s, best_v = i, s, cand
        chosen.append(best_i)
        cur = best_v
    return chosen, best_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--from-csv', nargs='+', required=True)
    p.add_argument('--top-k', type=int, default=12)
    p.add_argument('--diverse-by', nargs='*', default=None,
                   help='build the library for SPREAD instead of quality: take '
                        'the best config within each group of these columns '
                        "(e.g. aggr arm channels).  Ensemble error falls with "
                        'member decorrelation, not member accuracy, and the '
                        'plain top-K is 12 near-identical configurations.')
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--greedy-steps', type=int, default=25)
    p.add_argument('--snapshots', type=int, default=3,
                   help='top-N validation-AUPR epochs kept per run as extra '
                        'ensemble members (free: no extra training)')
    p.add_argument('--max-epochs', type=int, default=301)
    p.add_argument('--dump-scores', default=None,
                   help='npz of every combiner\'s ensemble scores, for '
                        'blending with a non-GNN model (improve/gbdt.py)')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.concat([pd.read_csv(c) for c in args.from_csv], ignore_index=True)
    set_keys(df)
    ranked = robust_rule(df)
    if args.diverse_by:
        # Best config per group, then the groups in criterion order.  Keeps the
        # library's individual quality as high as possible *subject to* covering
        # every distinct setting of the named columns.
        top = (ranked.groupby(args.diverse_by, sort=False).head(1)
               .sort_values('criterion', ascending=False).head(args.top_k))
        print(f'diverse library over {args.diverse_by}: '
              f'{len(top)} groups, criterion {top.criterion.min():.4f}'
              f'..{top.criterion.max():.4f}')
    else:
        top = ranked.head(args.top_k)
    for c, v in DEFAULTS.items():
        if c not in top.columns:
            top[c] = v
    cfgs = top[CFG_COLS].to_dict('records')
    print(f'{args.task}: library = {len(cfgs)} configs x {len(args.seeds)} seeds')

    splits = load_variant(args.run_dir, args.task, TASKS[args.task]['rna'],
                          'full', device)
    key = (TASKS[args.task]['rna'], args.task, 'drug')
    y = {f: d[key].edge_label.cpu().numpy()
         for f, d in (('val', splits[1]), ('test', splits[2]))}
    # One library per metric: members are captured at that metric's own
    # validation-selected epochs, so the AUPR library is not read off the
    # AUC-selected epoch.
    lib = {m: {'z': {'val': [], 'test': []}, 'r': {'val': [], 'test': []},
               'cfg_rank': []} for m in ('auc', 'aupr')}
    meta = []
    for ci, cfg in enumerate(cfgs):
        for seed in args.seeds:
            caps, _ = capture_scores(splits, args.task, cfg, seed, device,
                                     args.max_epochs, args.snapshots)
            for m in ('auc', 'aupr'):
                for c in caps[m]:
                    for f in ('val', 'test'):
                        lib[m]['z'][f].append(zs(c[f]))
                        lib[m]['r'][f].append(rk(c[f]))
                    lib[m]['cfg_rank'].append(ci)
            if not caps['auc']:
                continue
            meta.append(dict(
                cfg_rank=ci, seed=seed, **cfg,
                val_auc=roc_auc_score(y['val'], caps['auc'][0]['val']),
                test_auc=roc_auc_score(y['test'], caps['auc'][0]['test']),
                val_aupr=average_precision_score(y['val'], caps['aupr'][0]['val']),
                test_aupr=average_precision_score(y['test'],
                                                  caps['aupr'][0]['test'])))
            m0 = meta[-1]
            print(f"  [{len(meta):3d}] cfg{ci} {cfg['arm']:9s} lr={cfg['lr']:g} "
                  f"ch={int(cfg['channels'])} {cfg['aggr']:5s} "
                  f"d{cfg['dropout']} s={seed:<4d} "
                  f"val {m0['val_auc']:.4f}/{m0['val_aupr']:.4f} "
                  f"test {m0['test_auc']:.4f}/{m0['test_aupr']:.4f}", flush=True)
    meta = pd.DataFrame(meta)
    meta.to_csv(args.out.replace('.csv', '_members.csv'), index=False)
    for m in ('auc', 'aupr'):
        print(f"library[{m}]: {len(lib[m]['z']['val'])} members")

    rows = []
    pa = PAPER[args.task]
    dumped = {'y_val': y['val'], 'y_test': y['test']}

    def add(metric, label, idx, norm='z'):
        L = lib[metric][norm]
        r = dict(task=args.task, metric=metric, combiner=label, norm=norm,
                 n_members=len(idx))
        for f in ('val', 'test'):
            e = np.mean([L[f][i] for i in idx], axis=0)
            r[f'{f}_score'] = SCORER[metric](y[f], e)
            r[f'paper_{f}'] = pa[f][0 if metric == 'auc' else 1]
            dumped[f'{metric}_{norm}_{label}_{f}'] = e
        r['d_val'] = r['val_score'] - r['paper_val']
        r['d_test'] = r['test_score'] - r['paper_test']
        rows.append(r)

    for metric in ('auc', 'aupr'):
        ranks = np.array(lib[metric]['cfg_rank'])
        n = len(ranks)
        add(metric, 'seeds_of_top_cfg', list(np.where(ranks == 0)[0]))
        for norm in ('z', 'r'):
            add(metric, 'uniform_all', list(range(n)), norm)
            for k in (4, 8):
                idx = list(np.where(ranks < k)[0])
                if idx:
                    add(metric, f'uniform_top{k}cfg', idx, norm)
            chosen, _ = greedy_select(lib[metric][norm]['val'], y['val'],
                                      args.greedy_steps, metric)
            add(metric, 'greedy_on_val', chosen, norm)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    if args.dump_scores:
        np.savez(args.dump_scores, **dumped)
        print(f'wrote {args.dump_scores}')
    print(f'\n=== {args.task} (paper val {pa["val"][0]:.4f}/{pa["val"][1]:.4f}, '
          f'test {pa["test"][0]:.4f}/{pa["test"][1]:.4f}) ===')
    for metric in ('auc', 'aupr'):
        print(f'\n-- {metric.upper()} --')
        print(out[out.metric == metric][
            ['combiner', 'norm', 'n_members', 'val_score', 'd_val',
             'test_score', 'd_test']].to_string(
                index=False, float_format=lambda v: f'{v:+.4f}'))


if __name__ == '__main__':
    main()
