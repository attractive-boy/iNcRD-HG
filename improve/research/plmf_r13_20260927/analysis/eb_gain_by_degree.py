"""Where does the EB head gain?  Per-positive / per-negative rank-loss change (F -> F + EB, lam 3) by train-degree bins,
from the --dump caches (seed 42 5cv by default)."""
import sys, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score as AU
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'; lam = 3.0


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


def rank_loss(x, col):
    sp = np.sort(x[col][x.y == 1].values); sn = np.sort(x[col][x.y == 0].values); isp = x.y.values == 1
    l = np.zeros(len(x)); l[isp] = 1 - np.searchsorted(sn, x[col].values[isp]) / len(sn); l[~isp] = np.searchsorted(sp, x[col].values[~isp], side='right') / len(sp)
    return l


rows = []
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    tp = z['train_pos']; nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max(), z['tr'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max(), z['tr'][:, 1].max())) + 1
    A = np.zeros((nm, nd)); A[tp[:, 0], tp[:, 1]] = 1
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    bm, bd = fit(L, Y, P[:, 0], P[:, 1], nm, nd)
    te = z['te']
    x = pd.DataFrame(dict(m=te[:, 0], d=te[:, 1], y=z['yte'], F=z['logit_te'], EB=z['logit_te'] + bm[te[:, 0]] + bd[te[:, 1]]))
    x['dm'] = A.sum(1)[x.m]; x['dd'] = A.sum(0)[x.d]
    x['lF'] = rank_loss(x, 'F'); x['lEB'] = rank_loss(x, 'EB'); x['fold'] = f
    rows.append(x)
X = pd.concat(rows)
X['mbin'] = pd.cut(X.dm, [-1, 0, 1, 3, 7, 15, 100]); X['dbin'] = pd.cut(X.dd, [-1, 5, 20, 50, 150, 1000])
print(f'{ds}: 1-AUC  F {X.lF.mean():.5f}  EB {X.lEB.mean():.5f}')
for side in ('mbin', 'dbin'):
    for yv, nmv in ((1, 'pos'), (0, 'neg')):
        x = X[X.y == yv]; n = len(x)
        t = x.groupby(side).apply(lambda q: pd.Series(dict(frac=len(q) / n, F=q.lF.sum() / n, EB=q.lEB.sum() / n)))
        t['delta'] = t.EB - t.F
        print(f'--- {nmv} by {side}'); print(t.round(5).to_string())
