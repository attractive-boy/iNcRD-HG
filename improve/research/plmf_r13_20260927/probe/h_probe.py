"""Offline probes on dumped F-model states (valsel epoch): is there signal left beyond the F logit?
Stacker = logistic regression on [F logit, candidate feature(s)], cross-fitted over the 5 test folds
(fit on 4 folds' test pairs, score the 5th) -- a diagnostic, not a model."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
F = [np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz') for f in range(5)]
rows = []
for f, z in enumerate(F):
    ptr = expit(z['logit_tr']); print(f'fold {f}: train AUC {AU(z["ytr"], z["logit_tr"]):.4f}  val AUC {AU(z["yva"], z["logit_va"]):.4f}  test AUC {AU(z["yte"], z["logit_te"]):.4f}')
    tr, ytr, te, yte = z['tr'], z['ytr'], z['te'], z['yte']
    va, yva = z['va'], z['yva']
    # candidate 1: kNN label average in h-space over train+val pairs (cosine)
    H = np.concatenate([z['h_tr'], z['h_va']]); Y = np.concatenate([ytr, yva])
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9); Tn = z['h_te'] / (np.linalg.norm(z['h_te'], axis=1, keepdims=True) + 1e-9)
    S = Tn @ Hn.T
    feats = {}
    for k in (10, 50):
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        feats[f'knn{k}'] = Y[idx].mean(1)
    # candidate 2: residual propagation: mean residual of labelled pairs (train+val, out-of-sample val preferred) sharing m / d
    P_all = np.concatenate([tr, va]); L_all = np.concatenate([z['logit_tr'], z['logit_va']])
    R = Y - expit(L_all)
    df = pd.DataFrame(dict(m=P_all[:, 0], d=P_all[:, 1], r=R, ispos=Y))
    rm = df.groupby('m').r.mean(); rd = df.groupby('d').r.mean()
    feats['res_m'] = pd.Series(te[:, 0]).map(rm).fillna(0).values
    feats['res_d'] = pd.Series(te[:, 1]).map(rd).fillna(0).values
    # val-only residuals (out of sample)
    dv = pd.DataFrame(dict(m=va[:, 0], d=va[:, 1], r=yva - expit(z['logit_va'])))
    feats['vres_d'] = pd.Series(te[:, 1]).map(dv.groupby('d').r.mean()).fillna(0).values
    feats['vres_m'] = pd.Series(te[:, 0]).map(dv.groupby('m').r.mean()).fillna(0).values
    d = pd.DataFrame(dict(fold=f, y=yte, lg=z['logit_te'], m=te[:, 0], d=te[:, 1], **feats))
    rows.append(d)
X = pd.concat(rows, ignore_index=True)
def cf(cols):
    o = []
    for f in range(5):
        a, b = X[X.fold != f], X[X.fold == f]
        Z = lambda q: q[['lg'] + cols].values
        c = LogisticRegression(C=1.0, max_iter=3000).fit(Z(a), a.y); p = c.decision_function(Z(b))
        o.append((AU(b.y, p), AP(b.y, p)))
    return np.mean(o, 0)
base = np.mean([(AU(X[X.fold == f].y, X[X.fold == f].lg), AP(X[X.fold == f].y, X[X.fold == f].lg)) for f in range(5)], 0)
print(f'{ds} F logit  {base[0]:.4f}/{base[1]:.4f}')
for cols in (['knn10'], ['knn50'], ['res_m'], ['res_d'], ['res_m', 'res_d'], ['vres_d'], ['vres_m'], ['knn50', 'res_m', 'res_d']):
    r = cf(cols); print(f'  + {"+".join(cols):20s} {r[0]:.4f}/{r[1]:.4f}  d {r[0]-base[0]:+.4f}/{r[1]-base[1]:+.4f}')
