"""Offline probe 9: higher-order EASE (Steck & Liang 2021, 'Negative interactions... go higher') on the drug side:
column-wise ridge of A[:, d] on single-drug indicators and conjunctions A[m, i] * A[m, j] of frequent drug pairs, with
every feature that involves d removed (so the query cell never enters its own features).  Test pairs are never in
the graph, so their score is the plain prediction.  Stacked with the F logit, cross-fitted over the 5 test folds."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
from sklearn.linear_model import LogisticRegression
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
NPAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 400
rows = []
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    tp = z['train_pos']; nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max())) + 1
    A = np.zeros((nm, nd)); A[tp[:, 0], tp[:, 1]] = 1
    C = A.T @ A; np.fill_diagonal(C, 0)
    iu = np.triu_indices(nd, 1); cnt = C[iu]; top = np.argsort(-cnt)[:NPAIR]
    pi, pj = iu[0][top], iu[1][top]
    Xp = A[:, pi] * A[:, pj]                                   # [nm, NPAIR] conjunction features
    X = np.concatenate([A, Xp], 1)
    te = z['te']; out = {}
    for lam in (10.0, 50.0):
        sc = np.zeros(len(te)); sc1 = np.zeros(len(te))
        for d in np.unique(te[:, 1]):
            keep = np.ones(X.shape[1], bool); keep[d] = False; keep[nd:][(pi == d) | (pj == d)] = False
            Xd = X[:, keep]; G = Xd.T @ Xd + lam * np.eye(Xd.shape[1]); w = np.linalg.solve(G, Xd.T @ A[:, d])
            keep1 = np.zeros(X.shape[1], bool); keep1[:nd] = True; keep1[d] = False
            X1 = X[:, keep1]; w1 = np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ A[:, d])
            idx = np.where(te[:, 1] == d)[0]
            sc[idx] = Xd[te[idx, 0]] @ w; sc1[idx] = X1[te[idx, 0]] @ w1
        out[f'ho{lam:g}'] = sc; out[f'lin{lam:g}'] = sc1
    rows.append(pd.DataFrame(dict(fold=f, y=z['yte'], lg=z['logit_te'], **out)))
Xd_ = pd.concat(rows, ignore_index=True)


def cf(cols):
    o = []
    for f in range(5):
        a, b = Xd_[Xd_.fold != f], Xd_[Xd_.fold == f]
        c = LogisticRegression(C=1.0, max_iter=3000).fit(a[['lg'] + cols].values, a.y); p = c.decision_function(b[['lg'] + cols].values)
        o.append((AU(b.y, p), AP(b.y, p)))
    return np.mean(o, 0)


bs = np.mean([(AU(Xd_[Xd_.fold == f].y, Xd_[Xd_.fold == f].lg), AP(Xd_[Xd_.fold == f].y, Xd_[Xd_.fold == f].lg)) for f in range(5)], 0)
print(f'{ds} npair={NPAIR} base {bs[0]:.4f}/{bs[1]:.4f}')
for c in ('lin10', 'lin50', 'ho10', 'ho50'):
    alone = np.mean([AU(Xd_[Xd_.fold == f].y, Xd_[Xd_.fold == f][c]) for f in range(5)])
    r = cf([c]); print(f'  + {c:6s} alone AUC {alone:.4f} | stacked {r[0]:.4f}/{r[1]:.4f}  d {r[0]-bs[0]:+.4f}/{r[1]-bs[1]:+.4f}')
r = cf(['lin10', 'ho10']); print(f'  + lin10+ho10 stacked {r[0]:.4f}/{r[1]:.4f}  d {r[0]-bs[0]:+.4f}/{r[1]-bs[1]:+.4f}')
