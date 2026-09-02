# Fit the fusion on the TRAINING fold, not on the fold being reported.
#
# `improve/poolplus.py` closed the mechanism search with a result that points
# somewhere else.  `mlp` is the highest single-member test AUC on BOTH tasks
# (LncDrug 0.9910, MiDrug 0.9593), above every GNN configuration and every tree
# in the pool, and `xt35` improves LncDrug on all four cells -- and the author's
# `(AUC + AUPR)/2` validation rule assigns both of them weight 0.00.  So the
# binding constraint is no longer "is there a better member", it is "the
# selector cannot see one".
#
# The selector is a 1263-row fold (LncDrug; 3234 for MiDrug) on which the
# bootstrap sd of AUC and AUPR is 0.002-0.006 -- an order of magnitude above the
# differences it is being asked to rank.  Fitting four weights on it is fitting
# noise, and it is also the caveat that travels with every number in RESULTS.md.
#
# Three changes, all of them symmetric across the two ncRNA types:
#
#   1. OOF STACKING.  Every member already gets K-fold-guarded features
#      (`gbdt.py`: for fold k, features come from the association matrix with
#      fold k's positives deleted).  Nest one more level -- fit the member on
#      folds != k and predict fold k -- and the training fold yields honest
#      out-of-fold member scores.  The meta-learner is then fitted on 3801 rows
#      (LncDrug) / 9714 (MiDrug) instead of 1263 / 3234, and the validation fold
#      is not touched at all.  That is both a 3x-quieter selector and the end of
#      "weights fitted on the reported fold".
#
#   2. DEGREE-GATED WEIGHTS.  `altmodels.py`'s pair oracle says "pick whichever
#      member ranks each pair correctly" is worth 0.9904 against a 0.9846 blend,
#      i.e. 83% of the pool's ceiling is unrecovered -- and a constant weight
#      vector cannot recover any of it by construction.  The degree-stratified
#      residual (RESULTS.md: test AUC 0.937 / 0.912 / 0.550 across drug-degree
#      quartiles) says where the members should differ.  Weights become a
#      function of the drug's (and optionally the ncRNA's) training degree, via
#      bin-interaction terms.  This is only affordable BECAUSE of change 1: it
#      triples the parameter count, which 1263 rows cannot pay for.
#
#   3. EXTRA NEGATIVES.  Every tree in this repo is fitted on the frozen 2:1
#      labelled set -- 3801 rows out of 1322 x 216 = 285,552 possible pairs.
#      The GNN arm `neg` (resample negatives each epoch) was tried and is not
#      this: it reshuffles a fixed budget, it does not enlarge it.  Extra
#      negatives are sampled from pairs absent from the TRAINING association
#      matrix, featurised through the same per-fold guard, and used to fit the
#      base members only -- the meta-learner still sees the protocol's 2:1
#      balance, and the evaluation negatives are untouched.
#
#      Sampling is strict by default: only training positives are excluded, so
#      nothing about the val/test edge sets is read.  `--exclude-all-known`
#      reproduces what `RandomLinkSplit` itself does (it avoids every true edge
#      when it samples the frozen negatives); it is offered for comparability
#      and is not the headline.
#
# The GNN enters as a single scalar weight selected on validation, because its
# training-fold scores would require retraining it K times.  That still cuts the
# quantity fitted on the reported fold from four weights to one; `--no-gnn`
# removes even that and makes the whole predictor free of validation fitting.
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.feature_extra import extra_columns
from improve.gbdt import build_inputs, featurise
from improve.run import TASKS

PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}
FOLDS = ('train', 'val', 'test')


def member(name, seed):
    if name == 'gbdt':
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
    if name in ('extratrees', 'xt35'):
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                    n_jobs=-1, random_state=seed)
    if name == 'rf':
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                      n_jobs=-1, random_state=seed)
    if name == 'logreg':
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
    if name == 'mlp':
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=2000,
                          early_stopping=True, n_iter_no_change=20,
                          random_state=seed))
    raise ValueError(name)


def rk(v):
    """Rank-normalise to [0, 1].

    Applied per column and per dataset.  OOF scores come from members fitted on
    80% of the rows while val/test scores come from members fitted on 100%, so
    the two are not on a common scale; ranks remove the part of that difference
    which is monotone, which is most of it.
    """
    return v.argsort().argsort().astype(np.float64) / max(len(v) - 1, 1)


def sample_negatives(A, n, rng, forbidden=None):
    """Uniform (ncRNA, drug) pairs that are not edges of `A`."""
    n_r, n_d = A.shape
    block = set(np.flatnonzero(A.ravel() > 0).tolist())
    if forbidden is not None:
        block |= set(forbidden.tolist())
    out, want = [], int(n)
    while len(out) < want:
        draw = rng.randint(0, n_r * n_d, size=max(want * 2, 1024))
        out.extend(x for x in draw.tolist() if x not in block)
        block |= set(out)                       # sample without replacement
    flat = np.array(out[:want], dtype=np.int64)
    return np.stack([flat // n_d, flat % n_d])


def gate_matrix(Z, gate_vals, edges):
    """Expand member scores into per-bin blocks: weights become a function of
    degree instead of a constant.  With `edges` empty this is the identity, so
    the flat and gated combiners run through exactly the same code path."""
    if edges is None or len(edges) == 0:
        return Z
    b = np.digitize(gate_vals, edges)           # 0 .. len(edges)
    blocks = [Z * (b == i)[:, None] for i in range(len(edges) + 1)]
    ind = np.stack([(b == i).astype(float) for i in range(len(edges) + 1)], 1)
    return np.concatenate(blocks + [ind], axis=1)


def fit_meta(kind, M_tr, y_tr, M_ev):
    if kind == 'nnls':
        w, _ = nnls(M_tr, y_tr.astype(np.float64))
        return {f: M_ev[f] @ w for f in M_ev}, w
    if kind == 'logreg':
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=1.0))
        clf.fit(M_tr, y_tr)
        return ({f: clf.decision_function(M_ev[f]) for f in M_ev},
                clf[-1].coef_.ravel())
    if kind == 'gbdt':
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=False, random_state=0)
        clf.fit(M_tr, y_tr)
        return {f: clf.predict_proba(M_ev[f])[:, 1] for f in M_ev}, None
    raise ValueError(kind)


def scores(y, s):
    return roc_auc_score(y, s), average_precision_score(y, s)


def simplex_grid(m, step):
    """Non-negative weight vectors summing to 1, on a grid of the given step.

    This is the search the current fusion runs on the validation fold, written
    out so the OOF meta-learners can be compared against it on identical
    member scores rather than against a number quoted from a previous run.
    """
    n = int(round(1.0 / step))

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
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--gnn', default=None,
                   help='npz from multiens --dump-scores; blended in with one '
                        'val-selected scalar (the GNN has no OOF train scores)')
    p.add_argument('--combiner', default='auc_r_uniform_top8cfg')
    p.add_argument('--models', nargs='+',
                   default=['extratrees', 'xt35', 'mlp', 'gbdt', 'logreg'])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--extra-neg', type=float, default=0.0,
                   help='extra sampled negatives, as a multiple of the number '
                        'of training positives (change 3; 0 disables)')
    p.add_argument('--exclude-all-known', action='store_true',
                   help='also exclude val/test positives when sampling extra '
                        'negatives, as RandomLinkSplit itself does. Reads the '
                        'held-out edge SET (not its labels); off by default.')
    p.add_argument('--balance-extra', action='store_true',
                   help='downweight negatives so their total mass matches the '
                        "protocol's 2:1 ratio. Without this the enlarged pool "
                        'shifts the class prior, which costs the linear member "'
                        '0.044 val AUPR on MiDrug while the trees still gain.')
    p.add_argument('--gate', default='none',
                   choices=['none', 'drug', 'rna', 'both'])
    p.add_argument('--gate-bins', type=int, default=3)
    p.add_argument('--metas', nargs='+', default=['nnls', 'logreg', 'gbdt'])
    p.add_argument('--weight-step', type=float, default=0.1,
                   help='simplex step for the val_grid control combiner')
    p.add_argument('--dump-scores', default=None)
    p.add_argument('--dump-members', default=None,
                   help='also save the per-member rank-normalised val/test '
                        'scores, so a finer weight search can reuse them '
                        'without refitting')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    splits, key, A, S_r, S_d, bridge, attrs = build_inputs(
        args.run_dir, args.net_dir, args.task, 'cpu')
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(FOLDS, splits)}
    y = {f: rows[f][1] for f in FOLDS}
    tr_idx, tr_y = rows['train']
    n_real = tr_idx.shape[1]
    real = np.arange(n_real)          # the protocol's own rows, before change 3

    # ---- change 3: enlarge the negative pool the base members are fitted on.
    rng = np.random.RandomState(0)
    if args.extra_neg > 0:
        forbidden = None
        if args.exclude_all_known:
            keep = [rows[f][0][:, y[f] == 1] for f in FOLDS]
            allp = np.concatenate(keep, axis=1)
            forbidden = allp[0] * A.shape[1] + allp[1]
        ex = sample_negatives(A, args.extra_neg * tr_y.sum(), rng, forbidden)
        tr_idx = np.concatenate([tr_idx, ex], axis=1)
        tr_y = np.concatenate([tr_y, np.zeros(ex.shape[1], dtype=tr_y.dtype)])
        print(f'+{ex.shape[1]} sampled negatives '
              f'({n_real} -> {tr_idx.shape[1]} training rows)')

    # ---- features, with gbdt.py's K-fold guard, for base + xt35 column sets.
    fold = np.random.RandomState(0).randint(0, args.folds, size=tr_idx.shape[1])

    def build(pairs, Amat):
        base = featurise(pairs, Amat, S_r, S_d, bridge,
                         Xr=attrs[0], Xd=attrs[1]).reset_index(drop=True)
        return base, pd.concat(
            [base, extra_columns(pairs, Amat, S_r, S_d)], axis=1)

    X, X35 = {}, {}
    for f in ('val', 'test'):
        X[f], X35[f] = build(rows[f][0], A)
    parts, parts35 = [], []
    for k in range(args.folds):
        m = fold == k
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        b, b35 = build(tr_idx[:, m], Ak)
        b.index = b35.index = np.where(m)[0]
        parts.append(b)
        parts35.append(b35)
    X['train'] = pd.concat(parts).sort_index()
    X35['train'] = pd.concat(parts35).sort_index()

    # ---- change 1: out-of-fold member scores on the training rows.
    #
    # Extra negatives enlarge the pool the members see but must not move the
    # class prior they are calibrated to: the evaluation folds are 2:1 and a
    # member fitted at 12:1 is answering a different question.  Downweighting
    # the negatives so their total mass is unchanged keeps the prior fixed
    # while still letting every sampled pair inform the decision boundary.
    # MLPClassifier takes no sample_weight, so it is fitted unweighted -- which
    # is also the member the raw enlargement helps most.
    sw = None
    if args.balance_extra and args.extra_neg > 0:
        sw = np.ones(len(tr_y))
        neg = tr_y == 0
        sw[neg] = (n_real - tr_y[:n_real].sum()) / max(neg.sum(), 1)

    def fit(clf, name, Xv, yv, w):
        if w is None or name == 'mlp':
            return clf.fit(Xv, yv)
        if hasattr(clf, 'steps'):                 # sklearn Pipeline
            return clf.fit(Xv, yv, **{f'{clf.steps[-1][0]}__sample_weight': w})
        return clf.fit(Xv, yv, sample_weight=w)

    Z = {f: {} for f in FOLDS}
    for name in args.models:
        Xm = X35 if name == 'xt35' else X
        oof = np.zeros(tr_idx.shape[1])
        ev = {f: [] for f in ('val', 'test')}
        for seed in args.seeds:
            for k in range(args.folds):
                m = fold == k
                clf = member(name, seed)
                fit(clf, name, Xm['train'].values[~m], tr_y[~m],
                    None if sw is None else sw[~m])
                oof[m] += clf.predict_proba(Xm['train'].values[m])[:, 1]
            clf = member(name, seed)            # full fit for val / test
            fit(clf, name, Xm['train'].values, tr_y, sw)
            for f in ('val', 'test'):
                ev[f].append(clf.predict_proba(Xm[f].values)[:, 1])
        Z['train'][name] = oof / len(args.seeds)
        for f in ('val', 'test'):
            Z[f][name] = np.mean(ev[f], axis=0)
        # OOF AUC is reported over the protocol's own rows only, so it stays
        # comparable across --extra-neg settings.
        a_oof = roc_auc_score(y['train'][real], Z['train'][name][real])
        av, pv = scores(y['val'], Z['val'][name])
        at, _ = scores(y['test'], Z['test'][name])
        print(f'  {name:11s} oof {a_oof:.4f} | val {av:.4f}/{pv:.4f} '
              f'| test {at:.4f}', flush=True)

    if args.gnn:
        g = np.load(args.gnn)
        assert np.array_equal(g['y_val'], y['val']), 'val label order differs'
        for f in ('val', 'test'):
            Z[f]['gnn'] = g[f'{args.combiner}_{f}']

    # ---- rank-normalise per column per dataset, then assemble design matrices.
    names = list(args.models)
    R = {f: {n: rk(Z[f][n]) for n in Z[f]} for f in FOLDS}
    M = {f: np.stack([R[f][n] for n in names], axis=1) for f in FOLDS}

    # ---- change 2: weights as a function of degree.
    deg_r, deg_d = A.sum(1), A.sum(0)
    gv = {}
    for f in FOLDS:
        idx = tr_idx if f == 'train' else rows[f][0]
        gv[f] = {'drug': np.log1p(deg_d[idx[1]]),
                 'rna': np.log1p(deg_r[idx[0]])}
    edges = None
    if args.gate != 'none':
        keys = ['drug', 'rna'] if args.gate == 'both' else [args.gate]
        q = np.linspace(0, 100, args.gate_bins + 1)[1:-1]
        # one combined gate variable: bin index of the first key, offset by the
        # second, so `both` is a product grid rather than two additive gates.
        codes = {}
        for f in FOLDS:
            c = np.zeros(len(gv[f][keys[0]]))
            for j, kk in enumerate(keys):
                e = np.percentile(gv['train'][kk], q)
                c = c * (args.gate_bins) + np.digitize(gv[f][kk], e)
            codes[f] = c
        edges = np.arange(0.5, codes['train'].max() + 0.5)
        gvv = codes
    else:
        gvv = {f: np.zeros(M[f].shape[0]) for f in FOLDS}

    # The meta-learner sees the protocol's own 2:1 rows only: extra negatives
    # widen the BASE members' view, they must not reweight the combiner.
    D = {f: gate_matrix(M[f], gvv[f], edges) for f in FOLDS}

    res = []
    grid = np.linspace(0, 1, 51)
    dumps = {}
    for kind in args.metas:
        s, w = fit_meta(kind, D['train'][real], y['train'][real],
                        {f: D[f] for f in ('val', 'test')})
        row = dict(task=args.task, meta=kind, gate=args.gate,
                   extra_neg=args.extra_neg, members='+'.join(names),
                   n_train=n_real, n_fit=tr_idx.shape[1])
        for f in ('val', 'test'):
            row[f'{f}_auc'], row[f'{f}_aupr'] = scores(y[f], s[f])
        dumps[f'stack_{kind}_val'], dumps[f'stack_{kind}_test'] = s['val'], s['test']

        if args.gnn:
            # one scalar, chosen on validation by the author's own criterion.
            sv, st = rk(s['val']), rk(s['test'])
            gvl, gtl = rk(Z['val']['gnn']), rk(Z['test']['gnn'])
            crit = [np.mean(scores(y['val'], (1 - a) * sv + a * gvl))
                    for a in grid]
            a = float(grid[int(np.argmax(crit))])
            row['gnn_alpha'] = a
            for f, ss, gg in (('val', sv, gvl), ('test', st, gtl)):
                row[f'{f}g_auc'], row[f'{f}g_aupr'] = scores(
                    y[f], (1 - a) * ss + a * gg)
            dumps[f'stackgnn_{kind}_val'] = (1 - a) * sv + a * gvl
            dumps[f'stackgnn_{kind}_test'] = (1 - a) * st + a * gtl
        res.append(row)
        if w is not None and args.gate == 'none':
            row['weights'] = ' '.join(f'{n}={v:.3f}'
                                      for n, v in zip(names, np.asarray(w)))

    # ---- reference points measured on identical scores.
    #
    # `val_grid` is the combiner being replaced: non-negative weights on a
    # simplex grid, argmax of the author's own (AUC + AUPR)/2 criterion, fitted
    # on the validation fold.  It is the only fair control for the OOF metas --
    # same members, same scores, different fold to fit the weights on.  Its
    # val figure is fitted on the fold it reports and is not quotable; its test
    # figure is.
    ref_names = names + (['gnn'] if args.gnn else [])
    W = np.stack([R['val'][n] for n in ref_names], axis=1)
    Wt = np.stack([R['test'][n] for n in ref_names], axis=1)
    best, bw = -1.0, None
    for w in simplex_grid(len(ref_names), args.weight_step):
        sv = W @ w
        c = np.mean(scores(y['val'], sv))
        if c > best:
            best, bw = c, w
    refs = [('val_grid', {'val': W @ bw, 'test': Wt @ bw}, bw)]
    refs.append(('uniform', {f: M[f].mean(1) for f in ('val', 'test')}, None))
    b = max(names, key=lambda n: roc_auc_score(y['val'], R['val'][n]))
    refs.append((f'best_on_val({b})',
                 {f: R[f][b] for f in ('val', 'test')}, None))
    for lab, s, w in refs:
        row = dict(task=args.task, meta=lab, gate='-',
                   extra_neg=args.extra_neg, members='+'.join(ref_names
                                                              if w is not None
                                                              else names))
        for f in ('val', 'test'):
            row[f'{f}_auc'], row[f'{f}_aupr'] = scores(y[f], s[f])
        if w is not None:
            row['weights'] = ' '.join(f'{n}={v:.2f}'
                                      for n, v in zip(ref_names, w) if v > 0)
        res.append(row)
        dumps[f'{lab}_val'], dumps[f'{lab}_test'] = s['val'], s['test']

    df = pd.DataFrame(res)
    df.to_csv(args.out, index=False)
    pa = PAPER[args.task]
    print(f"\n{args.task}  paper val {pa['val'][0]:.4f}/{pa['val'][1]:.4f}  "
          f"test {pa['test'][0]:.4f}/{pa['test'][1]:.4f}")
    cols = [c for c in ('meta', 'gate', 'val_auc', 'val_aupr', 'test_auc',
                        'test_aupr', 'valg_auc', 'valg_aupr', 'testg_auc',
                        'testg_aupr', 'gnn_alpha') if c in df]
    print(df[cols].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    if args.dump_scores:
        np.savez(args.dump_scores, y_val=y['val'], y_test=y['test'], **dumps)
        print(f'wrote {args.dump_scores}')
    if args.dump_members:
        names_dump = [n for n in ref_names]
        np.savez(args.dump_members,
                 **{f'{n}_val': R['val'][n] for n in names_dump},
                 **{f'{n}_test': R['test'][n] for n in names_dump},
                 y_val=y['val'], y_test=y['test'])
        print(f'wrote {args.dump_members}')


if __name__ == '__main__':
    main()
