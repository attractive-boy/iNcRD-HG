# -*- coding: utf-8 -*-
"""Does a closed-form EASE item-item channel add information on top of the internal LR reference
(ctx+diff x LR, K=5 cross-fitting)?  5cv seed 42, both datasets."""
import os, sys
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
for p in ('plmf_innov_20260921', 'mphgnn_plmf_20260921', 'dlstmda_strongest_20260920', 'plmf_warm_20260922'):
    sys.path.insert(0, os.path.join(BASE, p))
from run_probe import load_nodes, make_bank  # noqa
from warm_scores import cached_xy  # noqa
from plmf_splits import get_split  # noqa


def ease_feats(sup, q, nm, nd, lams=(50.0, 200.0)):
    A = np.zeros((nm, nd)); A[sup[:, 0], sup[:, 1]] = 1
    A = torch.as_tensor(A, dtype=torch.float64)
    cols = []
    for lam in lams:
        for side, X in (('d', A), ('r', A.t())):
            G = X.t() @ X; P = torch.linalg.inv(G + lam * torch.eye(G.shape[0], dtype=torch.float64))
            B = -P / torch.diag(P)[None, :]; B.fill_diagonal_(0); S = X @ B
            S = S if side == 'd' else S.t()
            v = S[q[:, 0], q[:, 1]].numpy()
            rz = (v - S[q[:, 0]].mean(1).numpy()) / (S[q[:, 0]].std(1).numpy() + 1e-8)
            cz = (v - S[:, q[:, 1]].mean(0).numpy()) / (S[:, q[:, 1]].std(0).numpy() + 1e-8)
            cols += [v, rz, cz]
    return np.stack(cols, 1)


def xy_ease(tr, ytr, te, nm, nd, seed):
    Xtr = None
    for fit, qy in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, ytr.astype(int)):
        f = ease_feats(tr[fit[ytr[fit] == 1]], tr[qy], nm, nd)
        if Xtr is None:
            Xtr = np.zeros((len(ytr), f.shape[1]))
        Xtr[qy] = f
    Xte = ease_feats(tr[ytr == 1], te, nm, nd)
    both = np.concatenate([Xtr, Xte]); both = np.sign(both) * np.log1p(np.abs(both) * 10)
    Xtr, Xte = both[:len(ytr)], both[len(ytr):]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


for ds in ('MDR', 'MDS'):
    xr, xd = load_nodes(ds); bank = make_bank(xr, xd, 'kernel', 10)
    res = {'base': [], 'ease': []}
    for f in range(5):
        Xtr, Xte, tr, ytr, te, yte = cached_xy(bank, ds, 42, f, {'ctx', 'diff'}, 'kernel', 10)
        Etr, Ete = xy_ease(tr, ytr, te, xr.shape[0], xd.shape[0], 42 * 100 + f)
        for k, (A_, B_) in (('base', (Xtr, Xte)), ('ease', (np.hstack([Xtr, Etr]), np.hstack([Xte, Ete])))):
            s = LogisticRegression(C=0.3, max_iter=3000).fit(A_, ytr).predict_proba(B_)[:, 1]
            res[k].append((roc_auc_score(yte, s), average_precision_score(yte, s)))
    b, e = np.array(res['base']), np.array(res['ease'])
    print(f'{ds} ctx+diff LR {b.mean(0).round(4)}  +EASE {e.mean(0).round(4)}  delta {(e - b).mean(0).round(4)} wins {(e[:, 0] > b[:, 0]).sum()}/5', flush=True)
