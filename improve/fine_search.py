# Finer weight search on the current best pool.
#
# `improve/stack.py`'s `val_grid` combiner searches non-negative weights on a
# simplex grid with --weight-step 0.1 (the default used for the headline), so
# the reported weights are quantized to multiples of 1/10.  This re-searches the
# same 6-member pool (extratrees, xt35, mlp, gbdt, logreg, gnn) on the author's
# (AUC+AUPR)/2 criterion with a multi-resolution scheme: a coarse simplex pass,
# then a local refinement around the argmax at 100x the resolution, on the SAME
# member scores.  S_val stays the selection fold (optimistic, as always); S_test
# is the honest column.  No member is refitted.
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}
MEMBERS = ['extratrees', 'xt35', 'mlp', 'gbdt', 'logreg', 'gnn']


def crit(s, y):
    return (roc_auc_score(y, s) + average_precision_score(y, s)) / 2


def crit_aupr(s, y):
    return average_precision_score(y, s)


def crit_auc(s, y):
    return roc_auc_score(y, s)


def simplex(m, n):
    """Non-negative weight vectors summing to 1 on a grid of 1/n."""
    def rec(k, left):
        if k == m - 1:
            yield [left]
            return
        for i in range(left + 1):
            for rest in rec(k + 1, left - i):
                yield [i] + rest
    for v in rec(0, n):
        yield np.array(v, dtype=np.float64) / n


def refine(m, S, y, center, scale, crit_sel):
    """Local search: perturb each weight by +-scale, keep feasible, iterate."""
    w = center.copy()
    best = crit_sel(S @ w, y)
    improved = True
    while improved:
        improved = False
        for i in range(m):
            for delta in (scale, -scale):
                w2 = w.copy()
                w2[i] += delta
                if w2[i] < 0:
                    w2[i] = 0
                s2 = w2.sum()
                if s2 <= 0:
                    continue
                w2 = w2 / s2
                if (w2 < 0).any():
                    continue
                g = crit(S @ w2, y)
                if g > best:
                    w, best = w2, g
                    improved = True
    return w, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--members', required=True)
    p.add_argument('--task', required=True, choices=list(PAPER))
    p.add_argument('--coarse', type=int, default=20,
                   help='coarse simplex resolution (default 20 = step 0.05)')
    p.add_argument('--fine-scale', type=float, default=0.002,
                   help='refinement step around the coarse argmax')
    p.add_argument('--objective', default='joint', choices=['joint', 'aupr', 'auc'],
                   help='criterion to select weights on the validation fold')
    args = p.parse_args()

    crit_sel = {'joint': crit, 'aupr': crit_aupr, 'auc': crit_auc}[args.objective]

    d = np.load(args.members)
    y = {f: d[f'y_{f}'] for f in ('val', 'test')}
    S = {f: np.stack([d[f'{m}_{f}'] for m in MEMBERS], axis=1)
         for f in ('val', 'test')}

    # reproduce stack.py's val_grid at step 0.1 first (sanity check)
    bw = None
    for w in simplex(len(MEMBERS), 10):
        sv = S['val'] @ w
        g = crit_sel(sv, y['val'])
        if bw is None or g > crit_sel(S['val'] @ bw, y['val']):
            bw = w
    print(f'{args.task}: val_grid step=0.1 (stack.py headline, '
          f'objective={args.objective})')
    for f in ('val', 'test'):
        s = S[f] @ bw
        print(f'  {f} AUC {roc_auc_score(y[f], s):.4f}  '
              f'AUPR {average_precision_score(y[f], s):.4f}')
    print(f'  w: {" ".join(f"{m}={w:.2f}" for m, w in zip(MEMBERS, bw) if w > 0)}')

    # multi-resolution: coarse simplex then local refinement
    best = None
    for w0 in simplex(len(MEMBERS), args.coarse):
        sv = S['val'] @ w0
        g = crit_sel(sv, y['val'])
        if best is None or g > best[0]:
            best = (g, w0)
    w = best[1]
    for _ in range(4):
        w, _ = refine(len(MEMBERS), S['val'], y['val'], w, args.fine_scale,
                      crit_sel)
    print(f'\n{args.task}: fine search (coarse {args.coarse} + refine '
          f'{args.fine_scale}, objective={args.objective})')
    for f in ('val', 'test'):
        s = S[f] @ w
        print(f'  {f} AUC {roc_auc_score(y[f], s):.4f}  '
              f'AUPR {average_precision_score(y[f], s):.4f}')
    print(f'  w: {" ".join(f"{m}={w:.4f}" for m, w in zip(MEMBERS, w) if w > 0)}')

    pa = PAPER[args.task]
    print(f'\npaper val {pa["val"][0]:.4f}/{pa["val"][1]:.4f}  '
          f'test {pa["test"][0]:.4f}/{pa["test"][1]:.4f}')


if __name__ == '__main__':
    main()