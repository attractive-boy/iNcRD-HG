"""Offline probe 6: is the F logit sufficient w.r.t. its own hollow inputs?  Cross-fitted (5 test folds) gradient-boosted
stacker on [logit, qis, ease, diff, entity residuals] vs LR on the logit alone."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
rows = []
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    te = z['te']
    X = np.column_stack([z['logit_te'], z['qis_te'], z['ease_te'], z['diff_te']])
    rows.append((f, z['yte'], X))
res = {}
for f in range(5):
    Xa = np.concatenate([r[2] for r in rows if r[0] != f]); ya = np.concatenate([r[1] for r in rows if r[0] != f])
    Xb = rows[f][2]; yb = rows[f][1]
    res.setdefault('base', []).append((AU(yb, Xb[:, 0]), AP(yb, Xb[:, 0])))
    for name, cols in (('gbt_logit', [0]), ('gbt_all', list(range(Xa.shape[1])))):
        c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0).fit(Xa[:, cols], ya)
        p = c.predict_proba(Xb[:, cols])[:, 1]; res.setdefault(name, []).append((AU(yb, p), AP(yb, p)))
    c = LogisticRegression(C=0.1, max_iter=5000).fit(Xa, ya); p = c.decision_function(Xb); res.setdefault('lr_all', []).append((AU(yb, p), AP(yb, p)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base': continue
    m = np.mean(v, 0); print(f'  {k:10s} {m[0]:.4f}/{m[1]:.4f}  d {m[0]-bs[0]:+.4f}/{m[1]-bs[1]:+.4f}')
