# -*- coding: utf-8 -*-
"""Cross-fitting hygiene: inner folds K (support = (K-1)/K of train positives) vs the standard 5;
and repeated cross-fitting (average features over R inner splits)."""
import os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'plmf_innov_20260921'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'mphgnn_plmf_20260921'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'dlstmda_strongest_20260920'))
from run_probe import load_nodes, make_bank
from plmf_splits import get_split

def build_xy(bank, tr, ytr, te, fams, seed, K, R):
    Xtr = None
    for rep in range(R):
        Xr = None
        for fit, qy in StratifiedKFold(K, shuffle=True, random_state=seed + rep).split(tr, ytr.astype(int)):
            sup = tr[fit[ytr[fit] == 1]]
            f = bank.transform(sup, tr[qy], fams)
            if Xr is None: Xr = np.zeros((len(ytr), f.shape[1]), np.float32)
            Xr[qy] = f.values
        Xtr = Xr if Xtr is None else Xtr + Xr
    Xtr = Xtr / R
    Xte = bank.transform(tr[ytr == 1], te, fams).values
    both = np.concatenate([Xtr, Xte], 0); both = np.log1p(np.abs(both)) * np.sign(both)
    Xtr, Xte = both[:len(ytr)], both[len(ytr):]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd

ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='MDR'); ap.add_argument('--K', type=int, default=5)
ap.add_argument('--R', type=int, default=1); ap.add_argument('--seed', type=int, default=42); a = ap.parse_args()
xr, xd = load_nodes(a.dataset); bank = make_bank(xr, xd, 'kernel', 10); fams = {'ctx', 'diff'}
res = []
for f in range(5):
    tr, ytr, te, yte = get_split(a.dataset, '5cv', f, a.seed)
    Xtr, Xte = build_xy(bank, tr, ytr, te, fams, a.seed * 100 + f, a.K, a.R)
    m = LogisticRegression(max_iter=5000, C=0.3).fit(Xtr, ytr.astype(int)); s = m.predict_proba(Xte)[:, 1]
    res.append((roc_auc_score(yte, s), average_precision_score(yte, s)))
res = np.array(res)
line = f'## {a.dataset} s{a.seed} inner K={a.K} R={a.R}: AUC {res[:,0].mean():.4f}±{res[:,0].std():.4f} AUPR {res[:,1].mean():.4f}±{res[:,1].std():.4f}'
print(line); open(os.path.join(HERE, 'results', 'hygiene.txt'), 'a').write(line + '\n')
