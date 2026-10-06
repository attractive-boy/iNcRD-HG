# -*- coding: utf-8 -*-
"""Probe: closed-form EASE channels (drug side and miRNA side) as LR pair features, with the query
edge inside the fit (inclusive) vs exactly removed from the fit by a rank-2 Woodbury downdate
(exclusive).  Same folds / LR as feat_probe.py."""
import sys, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, '.')
from feat_probe import G, rowminmax, knn, stem, same, feats
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset


def ease_item(X, lam, K=None):
    """items = columns of X.  returns P = (X^T X + lam I + K)^-1."""
    Gm = X.T @ X + lam * np.eye(X.shape[1])
    if K is not None:
        Gm = Gm + K
    return np.linalg.inv(Gm)


def ease_scores(X, P, users, items, exclusive):
    """S[u,i] = sum_{j != i} X[u,j] B[j,i],  B[:,i] = -P[:,i]/P[i,i].
    exclusive: remove X[u,i] from the fit: Gram' = Gram - x_u e_i^T - e_i x_u^T + e_i e_i^T (if X[u,i]=1)."""
    x = X[users]                                   # [n, I]
    Pi = P[:, items].T                             # [n, I]   P e_i
    if not exclusive:
        col = -Pi / Pi[np.arange(len(items)), items][:, None]
        col[np.arange(len(items)), items] = 0
        return (x * col).sum(1)
    a = X[users, items]                            # 1 for observed pairs
    Pu = x @ P                                     # P x_u  (P symmetric)
    uPu = (x * Pu).sum(1); uPv = Pu[np.arange(len(items)), items]; vPv = Pi[np.arange(len(items)), items]
    out = np.zeros(len(items))
    for k in range(len(items)):
        if a[k] == 0:
            col = -Pi[k] / vPv[k]; col[items[k]] = 0; out[k] = x[k] @ col; continue
        Cinv = np.array([[-1.0, -1.0], [-1.0, 0.0]])
        M = Cinv + np.array([[uPu[k], uPv[k]], [uPv[k], vPv[k]]])
        W = np.stack([Pu[k], Pi[k]], 1)            # [I, 2]
        xcol = Pi[k] - W @ np.linalg.solve(M, np.array([uPv[k], vPv[k]]))
        col = -xcol / xcol[items[k]]; col[items[k]] = 0
        xu = x[k].copy(); xu[items[k]] = 0
        out[k] = xu @ col
    return out


def main(ds):
    g = np.load(G.format(ds), allow_pickle=True)
    mir, dru, pos, neg = load_dataset(ds)
    names = mir['miRNA_Name'].astype(str).values; seqs = mir['miRNA_Sequence'].astype(str).str.upper().str.replace('T', 'U').tolist()
    xr = rowminmax(g['mirna_kmer'].astype(float)); xd = rowminmax(g['drug_x'].astype(float))
    Sr, Sd = knn(xr, 10), knn(xd, 10)
    rels = [same([s[1:8] for s in seqs]), same([stem(n) for n in names])]
    rows = []
    for f in range(5):
        tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
        A = np.zeros((len(names), len(dru))); A[tr[ytr == 1, 0], tr[ytr == 1, 1]] = 1
        base_tr = feats(A, Sr, Sd, rels, tr[:, 0], tr[:, 1], True); base_te = feats(A, Sr, Sd, rels, te[:, 0], te[:, 1], True)
        for lam in (10.0, 50.0):
            Pd = ease_item(A, lam); Pr = ease_item(A.T, lam)
            for ex in (False, True):
                ftr = np.stack([ease_scores(A, Pd, tr[:, 0], tr[:, 1], ex), ease_scores(A.T, Pr, tr[:, 1], tr[:, 0], ex)], 1)
                fte = np.stack([ease_scores(A, Pd, te[:, 0], te[:, 1], ex), ease_scores(A.T, Pr, te[:, 1], te[:, 0], ex)], 1)
                for name, Xtr, Xte in (('ease', ftr, fte), ('qis+ease', np.hstack([base_tr, ftr]), np.hstack([base_te, fte])), ('qis', base_tr, base_te)):
                    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                    m = LogisticRegression(C=1.0, max_iter=3000).fit((Xtr - mu) / sd, ytr)
                    s = m.predict_proba((Xte - mu) / sd)[:, 1]
                    rows.append(dict(ds=ds, fold=f, lam=lam, feats=name, exclusive=ex, auc=roc_auc_score(yte, s), aupr=average_precision_score(yte, s)))
    d = pd.DataFrame(rows)
    print(d.groupby(['ds', 'lam', 'feats', 'exclusive'])[['auc', 'aupr']].mean().round(4))


if __name__ == '__main__':
    for ds in ('MDR', 'MDS'):
        main(ds)
