"""Leak control for the EB head on every dumped board (lam 10): intercepts refitted with the labelled negatives replaced by
an equal number of cells drawn uniformly from all cells outside the message graph and the training/val pairs."""
import sys, glob, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
lam = 10.0


def fit(L, Y, m, d, nm, nd):
    L = torch.tensor(L, dtype=torch.float64); Y = torch.tensor(Y, dtype=torch.float64); m = torch.tensor(m); d = torch.tensor(d)
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([bm, bd], max_iter=500, line_search_fn='strong_wolfe')

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(L + bm[m] + bd[d], Y, reduction='sum') + lam * (bm.pow(2).sum() + bd.pow(2).sum())
        l.backward(); return l
    opt.step(cl)
    return bm.detach().numpy(), bd.detach().numpy()


rows = []
for ds in ('MDR', 'MDS'):
    for split, seeds in (('5cv', (42, 43, 44)), ('single', (0, 1, 2))):
        for sd in seeds:
            for fn in sorted(glob.glob(f'cache/dump_FE3_{ds}_{split}_s{sd}_f*.npz')):
                z = np.load(fn)
                nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max(), z['tr'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max(), z['tr'][:, 1].max())) + 1
                Lf = np.full((nm, nd), np.nan)
                for part in ('u', 'tr', 'va', 'te'):
                    Q = z[part]; Lf[Q[:, 0], Q[:, 1]] = z[f'logit_{part}']
                P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
                te, yte, lte = z['te'], z['yte'], z['logit_te']
                bm, bd = fit(L, Y, P[:, 0], P[:, 1], nm, nd); s = lte + bm[te[:, 0]] + bd[te[:, 1]]
                cand = np.concatenate([z['u'], te]); rng = np.random.RandomState(sd * 10 + int(fn.split('_f')[-1][0]))
                nneg = int((Y == 0).sum()); pick = cand[rng.choice(len(cand), nneg, replace=False)]
                Ps = np.concatenate([P[Y == 1], pick]); Ys = np.r_[np.ones((Y == 1).sum()), np.zeros(nneg)]; Ls = np.r_[L[Y == 1], Lf[pick[:, 0], pick[:, 1]]]
                bm2, bd2 = fit(Ls, Ys, Ps[:, 0], Ps[:, 1], nm, nd); s2 = lte + bm2[te[:, 0]] + bd2[te[:, 1]]
                rows.append(dict(ds=ds, split=split, seed=sd, F=(AU(yte, lte), AP(yte, lte)), EB=(AU(yte, s), AP(yte, s)), EBstrict=(AU(yte, s2), AP(yte, s2))))
X = pd.DataFrame(rows)
for (ds, split), x in X.groupby(['ds', 'split']):
    F = np.array(x.F.tolist()); E = np.array(x.EB.tolist()); S = np.array(x.EBstrict.tolist())
    print(f'{ds} {split} n={len(x)}: EB d {E[:,0].mean()-F[:,0].mean():+.4f}/{E[:,1].mean()-F[:,1].mean():+.4f} | EB with strict negatives d {S[:,0].mean()-F[:,0].mean():+.4f}/{S[:,1].mean()-F[:,1].mean():+.4f}')
