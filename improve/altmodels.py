# More model classes, and the pair-level ceiling of the pool they form.
#
# The GNN x GBDT blend captured 17% of what "pick the correct model per pair"
# was worth (0.9846 -> 0.9856 against a 0.9904 oracle).  The remaining 83% says
# model-class diversity is not exhausted, so this adds four more scorers with
# deliberately different biases and measures the pool's ceiling BEFORE anything
# is tuned on it:
#
#   labelprop -- S_r @ A @ S_d, similarity propagation with no fitting at all.
#                The classic ncRNA-drug heuristic, and the only member with no
#                learned parameters.
#   mf        -- truncated SVD reconstruction of A.  Pure collaborative
#                filtering: no features, no similarity, no graph.
#   logreg    -- a LINEAR model on the same features the GBDT uses.  Same
#                inputs, opposite bias -- no interactions, no thresholds -- so
#                where it disagrees with the GBDT the disagreement is the
#                functional form, not the information.
#   extratrees -- bagged random splits rather than boosted residuals.
#
# `--oracle` reports, for every subset, the AUC of an omniscient selector that
# takes whichever member ranks each (positive, negative) pair correctly.  That
# is not achievable -- it needs the labels -- but it bounds every combiner, so
# it says up front whether a subset is worth blending.
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.gbdt import EPS, build_inputs, featurise
from improve.run import TASKS

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}


def fitted(name, seed):
    if name == 'gbdt':
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
    if name == 'extratrees':
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                    n_jobs=-1, random_state=seed)
    if name == 'logreg':
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
    raise ValueError(name)


def heuristic(name, pairs, A, S_r, S_d, k=24):
    """Unsupervised scorers: nothing is fitted, so nothing can leak."""
    r, d = pairs[0], pairs[1]
    if name == 'labelprop':
        M = S_r @ A @ S_d
        return M[r, d] / (S_r.sum(1)[r] * S_d.sum(1)[d] + EPS)
    if name == 'mf':
        U, sig, Vt = np.linalg.svd(A, full_matrices=False)
        kk = min(k, len(sig))
        return ((U[:, :kk] * sig[:kk]) @ Vt[:kk])[r, d]
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
                   default=['gbdt', 'extratrees', 'logreg', 'labelprop', 'mf'])
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

    g = np.load(args.gnn)
    assert np.array_equal(g['y_val'], y['val']), 'val label order differs'
    S = {'gnn': {f: g[f'{args.combiner}_{f}'] for f in ('val', 'test')}}

    for name in args.models:
        if name in ('labelprop', 'mf'):
            S[name] = {f: heuristic(name, rows[f][0], A, S_r, S_d)
                       for f in ('val', 'test')}
        else:
            acc = {f: [] for f in ('val', 'test')}
            for seed in args.seeds:
                clf = fitted(name, seed)
                clf.fit(X['train'].values, tr_y)
                for f in ('val', 'test'):
                    acc[f].append(clf.predict_proba(X[f].values)[:, 1])
            S[name] = {f: np.mean(acc[f], axis=0) for f in ('val', 'test')}
        print(f"  {name:11s} val AUC {roc_auc_score(y['val'], S[name]['val']):.4f}"
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
    print(df.head(12).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    if args.dump_scores:
        np.savez(args.dump_scores,
                 **{f'{n}_{f}': S[n][f] for n in S for f in ('val', 'test')},
                 y_val=y['val'], y_test=y['test'])
        print(f'wrote {args.dump_scores}')


if __name__ == '__main__':
    main()
