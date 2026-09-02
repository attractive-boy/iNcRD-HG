# Multi-view similarity: the ncRNA side has one view, the drug side has one view.
#
# `heter_data_prepare.py` uses exactly one similarity matrix per node type --
# `LncSimMat_ken.csv` / `MiSimMat_ken.csv` (Kendall tau over the 938-sample
# expression profile) and `drugSimMat.csv` (Tanimoto over Morgan fingerprints,
# built by `drug_similarity.py`).  Every feature in `gbdt.py` that is not pure
# degree reads one of those two matrices, so a single correlation coefficient
# decides the entire notion of "similar" for the whole method.  It was never
# compared against an alternative: `heter_data_prepare.py` refers to `pea`/`spe`
# variants the data release does not ship, and nothing downstream uses them.
#
# This is the one remaining route that adds INFORMATION rather than using the
# existing information better -- which is what every arm since the hyperparameter
# search has been doing.  It is aimed at MiDrug: `feature_groups.py` measured
# that task gaining +0.0019 from continuous similarity against LncDrug's +0.0003,
# so MiDrug is the one whose score depends on what the similarity matrix says.
#
# The views, all computed from inputs the model already receives:
#
#   ken       -- the released matrix.  The control.
#   pearson   -- linear correlation of the same expression profiles (`x`, which
#                is the 938-sample vector the GNN gets as a node attribute).
#   spearman  -- rank correlation of the same profiles.  Kendall and Spearman
#                are both rank statistics, so where they disagree it is the
#                concordant-pair vs rank-difference weighting, not the data.
#   cosine    -- unranked, uncentred.  For the drug side this is ChemBERTa
#                cosine, a genuinely different molecular view from Tanimoto
#                over Morgan fingerprints.
#   gip       -- Gaussian interaction profile kernel over the rows of the
#                association matrix.  The standard kernel in this literature and
#                the only view derived from `A` rather than from attributes,
#                so it is the only one that has to sit inside the K-fold guard.
#
# Fusion is a min-max-then-mean over the chosen views, which is the cheapest
# thing that is scale-invariant; SNF would add a hyperparameter to a component
# that currently has none, and the point here is whether the VIEW matters, not
# whether a fusion rule can be tuned.
#
# `--mode replace` swaps the fused matrix in for S_r / S_d and leaves the 29
# features untouched, so any delta is attributable to the similarity definition
# alone.  Per-view arms say which view carries it.
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.altmodels import fitted as _alt_fitted
from improve.gbdt import EPS, build_inputs, featurise
from improve.poolplus import fitted as _pool_fitted
from improve.run import TASKS


def fitted(name, seed):
    """The two member factories in this repo cover disjoint names."""
    try:
        return _alt_fitted(name, seed)
    except ValueError:
        return _pool_fitted(name, seed)

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}
VIEWS = ('ken', 'pearson', 'spearman', 'cosine', 'gip')


def _clean(M):
    M = np.nan_to_num(np.asarray(M, dtype=np.float32), nan=0.0)
    np.fill_diagonal(M, 0.0)          # the S @ A guard from gbdt.sim_matrix
    return M


def _minmax(M):
    """Scale off-diagonal magnitudes to [0, 1].

    The views are not commensurable -- Kendall tau on this expression data sits
    near 0.1 while a GIP kernel sits near 0.6 -- so averaging them raw would be
    a weighting rather than a fusion.
    """
    M = np.abs(M)
    hi = M.max()
    return M / hi if hi > 0 else M


def view(name, X, A, released):
    if name == 'ken':
        return _clean(released)
    if name == 'pearson':
        return _clean(np.corrcoef(X))
    if name == 'spearman':
        R = np.apply_along_axis(lambda v: v.argsort().argsort(), 1, X)
        return _clean(np.corrcoef(R))
    if name == 'cosine':
        U = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
        return _clean(U @ U.T)
    if name == 'gip':
        # exp(-gamma ||A_i - A_j||^2), gamma normalised by the mean squared
        # profile norm -- the standard parameterisation.  A function of A, so
        # it is rebuilt per fold from that fold's association matrix.
        sq = (A * A).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2 * (A @ A.T)
        g = A.shape[0] / max(sq.sum(), EPS)
        return _clean(np.exp(-g * np.maximum(d2, 0)))
    raise ValueError(name)


def fuse(names, X, A, released, cache):
    out = None
    for n in names:
        if n == 'gip':                                  # A-dependent, no cache
            M = _minmax(view(n, X, A, released))
        else:
            if n not in cache:
                cache[n] = _minmax(view(n, X, A, released))
            M = cache[n]
        out = M if out is None else out + M
    return (out / len(names)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--rna-views', nargs='+', default=None,
                   help='views fused for S_r; default: sweep each alone')
    p.add_argument('--drug-views', nargs='+', default=['ken'],
                   help="'ken' means the released Tanimoto matrix")
    p.add_argument('--model', default='extratrees')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--dump-scores', default=None,
                   help='npz of the seed-averaged scores, keyed `sim`, for '
                        'improve/boot_pair.py. Requires a single arm.')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    splits, key, A, S_r0, S_d0, bridge, attrs = build_inputs(
        args.run_dir, args.net_dir, args.task, 'cpu')
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(('train', 'val', 'test'), splits)}
    y = {f: rows[f][1] for f in ('val', 'test')}
    tr_idx, tr_y = rows['train']
    Xr, Xd = attrs
    cache_r, cache_d = {}, {}

    arms = ([dict(rna=[v], drug=args.drug_views) for v in VIEWS] +
            [dict(rna=['ken', 'pearson', 'spearman'], drug=args.drug_views),
             dict(rna=['ken', 'gip'], drug=args.drug_views),
             dict(rna=list(VIEWS), drug=args.drug_views),
             dict(rna=['ken'], drug=['ken', 'cosine']),
             dict(rna=['ken', 'gip'], drug=['ken', 'cosine', 'gip'])]
            if args.rna_views is None else
            [dict(rna=args.rna_views, drug=args.drug_views)])

    fold = np.random.RandomState(0).randint(0, args.folds, size=tr_idx.shape[1])
    res = []
    for arm in arms:
        def sims(Amat):
            return (fuse(arm['rna'], Xr, Amat, S_r0, cache_r),
                    fuse(arm['drug'], Xd, Amat.T, S_d0, cache_d))

        S_r, S_d = sims(A)
        X = {f: featurise(rows[f][0], A, S_r, S_d, bridge, Xr=Xr, Xd=Xd)
             for f in ('val', 'test')}
        parts = []
        for k in range(args.folds):
            m = fold == k
            Ak = A.copy()
            held = tr_idx[:, m & (tr_y == 1)]
            Ak[held[0], held[1]] = 0.0
            Sk_r, Sk_d = sims(Ak)       # gip is rebuilt from Ak, not from A
            f = featurise(tr_idx[:, m], Ak, Sk_r, Sk_d, bridge, Xr=Xr, Xd=Xd)
            f.index = np.where(m)[0]
            parts.append(f)
        X['train'] = pd.concat(parts).sort_index()

        acc = {f: {'auc': [], 'aupr': []} for f in ('val', 'test')}
        raw = {f: [] for f in ('val', 'test')}
        for seed in args.seeds:
            clf = fitted(args.model, seed)
            clf.fit(X['train'].values, tr_y)
            for f in ('val', 'test'):
                s = clf.predict_proba(X[f].values)[:, 1]
                raw[f].append(s)
                acc[f]['auc'].append(roc_auc_score(y[f], s))
                acc[f]['aupr'].append(average_precision_score(y[f], s))
        if args.dump_scores:
            assert len(arms) == 1, '--dump-scores needs a single arm'
            np.savez(args.dump_scores, y_val=y['val'], y_test=y['test'],
                     **{f'sim_{f}': np.mean(raw[f], axis=0)
                        for f in ('val', 'test')})
            print(f'wrote {args.dump_scores}')
        r = dict(task=args.task, model=args.model,
                 rna='+'.join(arm['rna']), drug='+'.join(arm['drug']))
        for f in ('val', 'test'):
            for m in ('auc', 'aupr'):
                r[f'{f}_{m}'] = float(np.mean(acc[f][m]))
                r[f'{f}_{m}_sd'] = float(np.std(acc[f][m]))
        res.append(r)
        print(f"  S_r={r['rna']:28s} S_d={r['drug']:16s} "
              f"val {r['val_auc']:.4f}/{r['val_aupr']:.4f} "
              f"test {r['test_auc']:.4f}/{r['test_aupr']:.4f}", flush=True)

    df = pd.DataFrame(res)
    df.to_csv(args.out, index=False)
    pa = PAPER[args.task]
    print(f"\n{args.task} {args.model}: released views are S_r=ken S_d=ken. "
          f"paper val {pa['val'][0]:.4f}/{pa['val'][1]:.4f}")
    b = df.loc[df.val_auc.idxmax()]
    print(f"best on val: S_r={b['rna']} S_d={b['drug']}  "
          f"val {b.val_auc:.4f}/{b.val_aupr:.4f} "
          f"test {b.test_auc:.4f}/{b.test_aupr:.4f}")


if __name__ == '__main__':
    main()
