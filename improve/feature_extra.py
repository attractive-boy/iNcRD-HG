# Continuous-feature enrichment, measured against the existing 29.
#
# `improve/feature_groups.py` nested the feature set (degree -> graph -> all)
# and found the only place continuous signal pays is MiDrug's `all` block
# (+0.0019 over `graph`).  This tests whether MORE continuous structure exists
# beyond the current set, on the strongest tree member (extratrees):
#
#   katz          -- geometric-decay weighted path count, path3*0.5 + path5*0.25.
#                    A bounded aggregation of the two existing path counts, the
#                    standard Katz truncation without the exploding path7.
#   deg_gap       -- |log_deg_r - log_deg_d|.  A drug with degree near its
#                    ncRNA's degree ranks differently from one an order of
#                    magnitude above or below it.
#   deg_ratio     -- log_deg_r / (log_deg_d + eps), the signed side of deg_gap.
#   wpath3        -- (S_r A A^T S_d)[r,d] normalised.  A similarity-weighted
#                    3-path: S_r A A^T reaches ncRNAs similar to r that share a
#                    drug with ncRNAs similar to d.  The existing collab_both is
#                    S_r A S_d (one similarity side); this weights both.
#   adamic_adar_r -- over d's ncRNA neighbours, sum 1/log(1+deg_r(n)).  A refined
#                    version of the drug-degree prior: popular drugs get down-
#                    weighted when their neighbours are themselves popular, which
#                    is exactly the cross-drug axis the metric reads.
#   adamic_adar_d -- the mirror on the ncRNA side.
#
# Every new feature is a function of the same (A, S_r, S_d) the existing set
# uses, so the same K-fold guard in `gbdt.py` covers it automatically.
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.altmodels import fitted
from improve.gbdt import EPS, build_inputs, featurise
from improve.run import TASKS

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}


def extra_columns(pairs, A, S_r, S_d):
    r, d = pairs[0], pairs[1]
    deg_r, deg_d = A.sum(1), A.sum(0)
    lr, ld = np.log1p(deg_r), np.log1p(deg_d)

    AtA = A.T @ A
    AAtA = A @ AtA
    A5 = AAtA @ AtA
    srow, dcol = S_r.sum(1), S_d.sum(1)

    wpath = (S_r @ AAtA) @ S_d
    katz = (0.5 * AAtA + 0.25 * A5) / (lr[:, None] * ld[None, :] + EPS)

    aa_drug = (A / (np.log1p(deg_r)[:, None] + EPS)).sum(0)   # per drug, over ncRNA nbrs
    aa_rna = (A / (np.log1p(deg_d)[None, :] + EPS)).sum(1)    # per ncRNA, over drug nbrs

    return pd.DataFrame({
        'katz': katz[r, d],
        'deg_gap': np.abs(lr[r] - ld[d]),
        'deg_ratio': lr[r] / (ld[d] + EPS),
        'wpath3': wpath[r, d] / (srow[r] * dcol[d] + EPS),
        'adamic_adar_r': aa_drug[d],
        'adamic_adar_d': aa_rna[r],
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--model', default='extratrees',
                   choices=['extratrees', 'gbdt', 'logreg', 'mlp'])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    p.add_argument('--out', required=True)
    args = p.parse_args()

    splits, key, A, S_r, S_d, bridge, attrs = build_inputs(
        args.run_dir, args.net_dir, args.task, 'cpu')
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(('train', 'val', 'test'), splits)}
    y = {f: rows[f][1] for f in ('val', 'test')}

    base = {f: featurise(rows[f][0], A, S_r, S_d, bridge,
                         Xr=attrs[0], Xd=attrs[1]) for f in ('val', 'test')}
    extra = {f: extra_columns(rows[f][0], A, S_r, S_d) for f in ('val', 'test')}
    X = {f: pd.concat([base[f], extra[f]], axis=1) for f in ('val', 'test')}

    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, args.folds, size=tr_idx.shape[1])
    parts = []
    for k in range(args.folds):
        m = fold == k
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        f = pd.concat([featurise(tr_idx[:, m], Ak, S_r, S_d, bridge,
                                 Xr=attrs[0], Xd=attrs[1]),
                       extra_columns(tr_idx[:, m], Ak, S_r, S_d)], axis=1)
        f.index = np.where(m)[0]
        parts.append(f)
    X['train'] = pd.concat(parts).sort_index()

    new_cols = list(extra['val'].columns)
    for label, cols in (('base29', list(base['val'].columns)),
                        ('+6', list(X['train'].columns))):
        acc = {f: {'auc': [], 'aupr': []} for f in ('val', 'test')}
        for seed in args.seeds:
            clf = fitted(args.model, seed)
            clf.fit(X['train'][cols].values, tr_y)
            for f in ('val', 'test'):
                s = clf.predict_proba(X[f][cols].values)[:, 1]
                acc[f]['auc'].append(roc_auc_score(y[f], s))
                acc[f]['aupr'].append(average_precision_score(y[f], s))
        r = dict(task=args.task, model=args.model, set=label, n_feat=len(cols))
        for f in ('val', 'test'):
            for m in ('auc', 'aupr'):
                r[f'{f}_{m}'] = float(np.mean(acc[f][m]))
                r[f'{f}_{m}_std'] = float(np.std(acc[f][m]))
        res = {k: v for k, v in r.items() if k != 'set'}
        res['set'] = label
        print(f"  {label:6s} ({len(cols):2d} feat)  "
              f"val {r['val_auc']:.4f}/{r['val_aupr']:.4f}  "
              f"test {r['test_auc']:.4f}/{r['test_aupr']:.4f}")

    pa, pp = PAPER[args.task]['val']
    print(f"\n{args.task} {args.model}: paper S_val {pa:.4f}/{pp:.4f}")
    print('new features:', ', '.join(new_cols))


if __name__ == '__main__':
    main()