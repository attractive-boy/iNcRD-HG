# Multiple GNN combiners in the fusion pool.
#
# `improve/multiens.py` dumps 18 GNN score vectors per fold: auc_*/aupr_* x
# r/z normalisation x uniform_top8cfg/uniform_top4cfg/uniform_all/seeds_of_
# top_cfg/greedy_on_val.  The whole pipeline has used exactly ONE -- the
# auc_r_uniform_top8cfg passed to stack.py's --gnn.  The AUPR-optimised
# combiners (aupr_r_greedy_on_val etc.) reach val AUPR 0.9678 against the
# auc_* 0.9533 -- a +0.0145 gap on the metric that has the real residual.
#
# Two things are tested here:
#   1. swapping the GNN entry from the auc_ to the aupr_ combiner;
#   2. admitting several GNN combiners as INDEPENDENT members of the fusion
#      pool (they are genuinely different scorers -- AUC-optimal vs AUPR-
#      optimal, rank vs z normalisation), on top of the tree members.
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}
TREES = ['extratrees', 'xt35', 'mlp', 'gbdt', 'logreg']
GNN_COMBINERS = [
    'auc_r_uniform_top8cfg',      # the one the pipeline has used all along
    'aupr_r_greedy_on_val',
    'aupr_r_uniform_top8cfg',
    'aupr_z_uniform_top8cfg',
    'auc_r_uniform_all',
    'aupr_r_uniform_all',
]


def simplex(m, n):
    def rec(k, left):
        if k == m - 1:
            yield [left]
            return
        for i in range(left + 1):
            for rest in rec(k + 1, left - i):
                yield [i] + rest
    for v in rec(0, n):
        yield np.array(v, dtype=np.float64) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--members', required=True, help='stack.py --dump-members npz')
    p.add_argument('--gnn', required=True, help='multiens dump-scores npz')
    p.add_argument('--task', required=True, choices=list(PAPER))
    p.add_argument('--coarse', type=int, default=10)
    args = p.parse_args()

    d = np.load(args.members)
    g = np.load(args.gnn)
    y = {f: d[f'y_{f}'] for f in ('val', 'test')}
    assert np.array_equal(g['y_val'], y['val'])
    M = {f: np.stack([d[f'{m}_{f}'] for m in TREES], axis=1)
         for f in ('val', 'test')}

    # rank-normalise everything (trees come rank-norm from dump-members already;
    # gnn raw probs need rank-norm to be comparable)
    def rk(v):
        return v.argsort().argsort().astype(np.float64) / max(len(v) - 1, 1)

    def crit(s):
        return (roc_auc_score(y['val'], s) + average_precision_score(
            y['val'], s)) / 2

    print(f'{args.task}: single-GNN-entry variants '
          f'(trees + one gnn combiner, joint criterion)\n')
    for comb in GNN_COMBINERS:
        G = {f: rk(g[f'{comb}_{f}']) for f in ('val', 'test')}
        S = {f: np.concatenate([M[f], G[f][:, None]], axis=1)
             for f in ('val', 'test')}
        best = None
        for w in simplex(S['val'].shape[1], args.coarse):
            sv = S['val'] @ w
            c = crit(sv)
            if best is None or c > best[0]:
                best = (c, w)
        w = best[1]
        ta = roc_auc_score(y['test'], S['test'] @ w)
        tp = average_precision_score(y['test'], S['test'] @ w)
        va = roc_auc_score(y['val'], S['val'] @ w)
        vp = average_precision_score(y['val'], S['val'] @ w)
        print(f'  {comb:28s} val {va:.4f}/{vp:.4f}  '
              f'test {ta:.4f}/{tp:.4f}')

    # multi-GNN pool: trees + aupr_r_greedy + auc_r_uniform_top8cfg + aupr_z
    combos = ['auc_r_uniform_top8cfg', 'aupr_r_greedy_on_val',
              'aupr_z_uniform_top8cfg']
    G = {f: np.stack([rk(g[f'{c}_{f}']) for c in combos], axis=1)
         for f in ('val', 'test')}
    S = {f: np.concatenate([M[f], G[f]], axis=1) for f in ('val', 'test')}
    n = S['val'].shape[1]
    best = None
    for w in simplex(n, 8):
        sv = S['val'] @ w
        c = crit(sv)
        if best is None or c > best[0]:
            best = (c, w)
    w = best[1]
    names = TREES + [f'gnn:{c.split("_")[1]}_{c.split("_")[2]}' for c in combos]
    va = roc_auc_score(y['val'], S['val'] @ w)
    vp = average_precision_score(y['val'], S['val'] @ w)
    ta = roc_auc_score(y['test'], S['test'] @ w)
    tp = average_precision_score(y['test'], S['test'] @ w)
    ws = ' '.join(f'{nm}={v:.2f}' for nm, v in zip(names, w) if v > 0.01)
    print(f'\n  MULTI-GNN pool ({n} members): val {va:.4f}/{vp:.4f}  '
          f'test {ta:.4f}/{tp:.4f}')
    print(f'  w: {ws}')

    pa = PAPER[args.task]
    print(f'\npaper val {pa["val"][0]:.4f}/{pa["val"][1]:.4f}  '
          f'test {pa["test"][0]:.4f}/{pa["test"][1]:.4f}')


if __name__ == '__main__':
    main()