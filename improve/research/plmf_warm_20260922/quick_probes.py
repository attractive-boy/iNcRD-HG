# -*- coding: utf-8 -*-
"""Cheap warm-face probes on the cached ctx+diff features:
  Q1 readout objectives: LR (ref), pairwise AUC (RankNet-style) loss, AP-surrogate loss, LR + hidden-positive-cleaned negatives
  Q3 remove training negatives documented as positives elsewhere (in_other / in_midrug)
"""
import os, sys, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset
torch.manual_seed(0)

def load(ds, seed, f):
    z = np.load(os.path.join(HERE, 'cache', f'{ds}_s{seed}_f{f}_ctx+diff_kernel10.npz'))
    return z['Xtr'], z['Xte'], z['tr'], z['ytr'], z['te'], z['yte']

def rank_model(Xtr, ytr, Xte, loss='auc', epochs=300, lr=0.05, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(Xtr); y = torch.tensor(ytr).float(); Xt = torch.tensor(Xte)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    P = X[y == 1]; N = X[y == 0]
    for ep in range(epochs):
        opt.zero_grad()
        sp = P @ w + b; sn = N @ w + b
        if loss == 'auc':   # RankNet / logistic pairwise loss over all pos-neg pairs (subsample negs)
            idx = torch.randint(0, len(sn), (4096,))
            diff = sp[:, None] - sn[idx][None, :]
            l = torch.nn.functional.softplus(-diff).mean()
        elif loss == 'ap':  # smooth AP surrogate (sigmoid ranking, Qin et al. style)
            s = torch.cat([sp, sn]); yy = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
            idx = torch.randperm(len(s))[:3000]
            s = s[idx]; yy = yy[idx]
            D = torch.sigmoid((s[None, :] - s[:, None]) / 0.1)  # D[i,j]=P(j ranked above i)
            rank_all = 1 + D.sum(1) - torch.diagonal(D)
            rank_pos = 1 + (D * yy[None, :]).sum(1) - torch.diagonal(D) * yy
            l = -((rank_pos / rank_all) * yy).sum() / yy.sum()
        else:
            raise ValueError
        l.backward(); opt.step()
    with torch.no_grad():
        return (Xt @ w + b).numpy()

def run(ds, seed=42):
    mir, dru, pos, neg = load_dataset(ds)
    flags = pd.read_csv(os.path.join(HERE, 'results', f'negflags_{ds}_s{seed}.csv'))
    fl = set(zip(flags.r, flags.d))
    fl_other = set(zip(flags[flags.in_other == 1].r, flags[flags.in_other == 1].d))
    fl_any = set(zip(flags[(flags.in_other == 1) | (flags.in_midrug == 1)].r, flags[(flags.in_other == 1) | (flags.in_midrug == 1)].d))
    res = {}
    for f in range(5):
        Xtr, Xte, tr, ytr, te, yte = load(ds, seed, f)
        def ev(name, s):
            res.setdefault(name, []).append((roc_auc_score(yte, s), average_precision_score(yte, s)))
        m = LogisticRegression(max_iter=3000, C=0.3).fit(Xtr, ytr.astype(int)); ev('lr0.3', m.predict_proba(Xte)[:, 1])
        for C in (0.1, 1.0, 3.0):
            m = LogisticRegression(max_iter=3000, C=C).fit(Xtr, ytr.astype(int)); ev(f'lr{C}', m.predict_proba(Xte)[:, 1])
        ev('rank_auc', rank_model(Xtr, ytr, Xte, 'auc', seed=f))
        ev('rank_ap', rank_model(Xtr, ytr, Xte, 'ap', seed=f))
        # Q3: drop documented-positive negatives from TRAINING only
        keep = np.array([not ((ytr[i] == 0) and ((tr[i, 0], tr[i, 1]) in fl_any)) for i in range(len(ytr))])
        m = LogisticRegression(max_iter=3000, C=0.3).fit(Xtr[keep], ytr[keep].astype(int)); ev('lr0.3_cleanneg', m.predict_proba(Xte)[:, 1])
        # Q3b: relabel them as positives in training
        y2 = ytr.copy(); y2[~keep] = 1
        m = LogisticRegression(max_iter=3000, C=0.3).fit(Xtr, y2.astype(int)); ev('lr0.3_relabelneg', m.predict_proba(Xte)[:, 1])
        print(f'  {ds} fold {f} done', flush=True)
    print(f'\n### {ds} seed {seed}')
    for k, v in res.items():
        v = np.array(v); print(f'  {k:18s} AUC {v[:,0].mean():.4f}±{v[:,0].std():.4f}  AUPR {v[:,1].mean():.4f}±{v[:,1].std():.4f}')

if __name__ == '__main__':
    for ds in ('MDR', 'MDS'):
        run(ds)
