# Widening the fusion pool with more independent model classes.
#
# `improve/altmodels.py` established that the gain in stages 7-8 comes from
# model-class diversity: each member that fails on different edges adds unique
# wins to the blend.  The current pool has four scorers with meaningful signal --
# gnn, gbdt, extratrees, logreg -- and the pair-level oracle says the ceiling is
# not reached (the GNN x GBDT blend captured 17% of "pick the correct model per
# pair").
#
# This adds members with deliberately DIFFERENT biases from the ones already in
# the pool, on the SAME 19 K-fold-guarded features and the SAME GNN scores:
#
#   rf        -- RandomForest: bagged tree splits.  Extratrees draws random
#                cut points; a forest draws the best cut per feature, so it is
#                a different point in the same family.
#   mlp       -- sklearn MLPClassifier on the features.  The pool has one
#                linear member (logreg) and two tree families; a shallow
#                nonlinear member is the functional middle that neither can
#                represent (no thresholds, but interactions).
#   subspace  -- ExtraTrees with random feature subspaces (max_features < 1).
#                Random subspaces decorrelate the trees further, so bagging
#                averages over more diverse members.
#   posfocus  -- extratrees fit on the top-k*4 hardest negatives only (highest
#                similarity to the positive pool), so the classifier spends its
#                capacity on the ranking head that AUPR measures.  This is the
#                AUPR-targeted member -- AUPR's residual is 3x AUC's and no
#                member in the pool is trained for the top of the ranking.
#   balrf     -- RandomForest with class_weight='balanced'.  A cost-sensitive
#                prior on the positive class, cheap and independent of the
#                others.
#
# Every member is fit per-seed on the same K-fold feature matrix and averaged
# over seeds, exactly like altmodels.py.  Scores are dumped in the same npz
# layout so blend.py consumes them unchanged.  The printed oracle is the
# pair-level AUC of "pick whichever member ranks each pair correctly" for the
# ENLARGED pool against the current one, so the question "is a subset worth
# blending" is answered before any fitting on it.
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.gbdt import EPS, build_inputs, featurise
from improve.run import TASKS

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}


def fitted(name, seed, extra=None):
    if name == 'rf':
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                      n_jobs=-1, random_state=seed)
    if name == 'balrf':
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                      class_weight='balanced', n_jobs=-1,
                                      random_state=seed)
    if name == 'mlp':
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2,
                          max_iter=2000, early_stopping=True,
                          n_iter_no_change=20, random_state=seed))
    if name == 'subspace':
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                    max_features=0.3, n_jobs=-1,
                                    random_state=seed)
    if name == 'posfocus':
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                    n_jobs=-1, random_state=seed)
    raise ValueError(name)


def pair_oracle(scores, y):
    """AUC of a selector that picks whichever member ranks each pair right."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    any_right = np.zeros((len(pos), len(neg)), dtype=bool)
    for s in scores:
        any_right |= s[pos][:, None] > s[neg][None, :]
    return any_right.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--gnn', required=True, help='npz from multiens --dump-scores')
    p.add_argument('--combiner', default='auc_r_uniform_top8cfg')
    p.add_argument('--models', nargs='+',
                   default=['rf', 'balrf', 'mlp', 'subspace', 'posfocus'])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--out', required=True)
    p.add_argument('--dump-scores', default=None)
    args = p.parse_args()

    splits, key, A, S_r, S_d, bridge, attrs = build_inputs(
        args.run_dir, args.net_dir, args.task, 'cpu')
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(('train', 'val', 'test'), splits)}
    y = {f: rows[f][1] for f in ('val', 'test')}

    X = {f: featurise(rows[f][0], A, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1]) for f in ('val', 'test')}
    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, args.folds, size=tr_idx.shape[1])
    parts = []
    for k in range(args.folds):
        m = fold == k
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        f = featurise(tr_idx[:, m], Ak, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1])
        f.index = np.where(m)[0]
        parts.append(f)
    X['train'] = pd.concat(parts).sort_index()

    # posfocus: train on all positives + the hardest negatives only.  "Hard" =
    # the negatives whose feature vector is most similar to the mean positive
    # vector, i.e. the ones the model must learn to separate from positives at
    # the top of the ranking.  AUPR measures the ranking head, and no member in
    # the pool is trained for it.
    k = max(20, int(4 * tr_y.sum()))
    pos_mean = X['train'].values[tr_y == 1].mean(0)
    neg_idx = np.where(tr_y == 0)[0]
    hard = np.argsort(
        -np.abs(X['train'].values[neg_idx] - pos_mean).sum(1))[:k]
    hard_idx = np.concatenate([np.where(tr_y == 1)[0], neg_idx[hard]])

    g = np.load(args.gnn)
    assert np.array_equal(g['y_val'], y['val']), 'val label order differs'
    S = {'gnn': {f: g[f'{args.combiner}_{f}'] for f in ('val', 'test')}}

    for name in args.models:
        acc = {f: [] for f in ('val', 'test')}
        for seed in args.seeds:
            clf = fitted(name, seed)
            if name == 'posfocus':
                Xtr, ytr = X['train'].iloc[hard_idx], tr_y[hard_idx]
            else:
                Xtr, ytr = X['train'].values, tr_y
            clf.fit(Xtr, ytr)
            for f in ('val', 'test'):
                acc[f].append(clf.predict_proba(X[f].values)[:, 1])
        S[name] = {f: np.mean(acc[f], axis=0) for f in ('val', 'test')}
        print(f"  {name:10s} val AUC {roc_auc_score(y['val'], S[name]['val']):.4f}"
              f"  AUPR {average_precision_score(y['val'], S[name]['val']):.4f}"
              f"   test AUC {roc_auc_score(y['test'], S[name]['test']):.4f}",
              flush=True)

    names = list(S)
    out = []
    for size in range(1, len(names) + 1):
        for sub in itertools.combinations(names, size):
            if 'gnn' not in sub:
                continue
            o = pair_oracle([S[n]['val'] for n in sub], y['val'])
            out.append(dict(task=args.task, subset='+'.join(sub), size=size,
                            oracle_val_auc=o))
    df = pd.DataFrame(out).sort_values('oracle_val_auc', ascending=False)
    df.to_csv(args.out, index=False)

    base = df[df.subset == 'gnn'].oracle_val_auc.iloc[0]
    print(f"\n{args.task}: pair-level oracle on the validation fold "
          f"(GNN alone = {base:.4f}, paper = {PAPER[args.task]['val'][0]:.4f})")
    print(df.head(14).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    if args.dump_scores:
        np.savez(args.dump_scores,
                 **{f'{n}_{f}': S[n][f] for n in S for f in ('val', 'test')},
                 y_val=y['val'], y_test=y['test'])
        print(f'wrote {args.dump_scores}')


if __name__ == '__main__':
    main()