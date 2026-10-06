"""Offline probe 4: hollow (leave-one-out) target statistics of the labelled pairs sharing the query's miRNA / drug,
as stacking features on the F logit (fit on train+val pairs with LOO statistics, applied to test with full statistics)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
from sklearn.linear_model import LogisticRegression
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    feats_tr, feats_te = [], []
    for side in (0, 1):
        s = pd.DataFrame(dict(k=P[:, side], y=Y))
        S = s.groupby('k').y.agg(['sum', 'count'])
        sm = S['sum'].reindex(P[:, side]).values; ct = S['count'].reindex(P[:, side]).values
        prior = Y.mean(); a0 = 2.0
        loo = (sm - Y + a0 * prior) / (ct - 1 + a0)               # hollow: own label removed
        smt = S['sum'].reindex(te[:, side]).fillna(0).values; ctt = S['count'].reindex(te[:, side]).fillna(0).values
        full = (smt + a0 * prior) / (ctt + a0)
        from scipy.special import logit as lg
        feats_tr += [lg(np.clip(loo, 1e-3, 1 - 1e-3)), np.log1p(ct - 1)]; feats_te += [lg(np.clip(full, 1e-3, 1 - 1e-3)), np.log1p(ctt)]
        # residual version: mean residual of the other pairs
    Xtr = np.column_stack([L] + feats_tr); Xte = np.column_stack([lte] + feats_te)
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    for C in (0.01, 0.1, 1.0):
        c = LogisticRegression(C=C, max_iter=3000).fit(Xtr, Y); s = c.decision_function(Xte)
        res.setdefault(f'te_C{C}', []).append((AU(yte, s), AP(yte, s)))
    # drug-only
    c = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr[:, [0, 3, 4]], Y); s = c.decision_function(Xte[:, [0, 3, 4]])
    res.setdefault('te_drug', []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base': continue
    m = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])]); w2 = np.mean([x[1] > y[1] for x, y in zip(v, res['base'])])
    print(f'  {k:10s} {m[0]:.4f}/{m[1]:.4f}  d {m[0]-bs[0]:+.4f}/{m[1]-bs[1]:+.4f}  win {w:.1f}/{w2:.1f}')
