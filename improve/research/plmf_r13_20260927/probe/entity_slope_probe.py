"""Offline probe 3: entity random effects beyond a bias -- per-drug slope on log(1 + hollow miRNA degree) and per-miRNA
slope on log(1 + hollow drug degree) ("which popularity regime does this entity live in")."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
def fit(L, Y, dI, mI, xm, xd, nd, nm, lb, lm, ls_d, ls_m, use_sd, use_sm):
    T = lambda v: torch.tensor(v, dtype=torch.float64)
    L, Y, xm, xd = T(L), T(Y), T(xm), T(xd); dI = torch.tensor(dI); mI = torch.tensor(mI)
    ps = {k: torch.zeros(n, dtype=torch.float64, requires_grad=True) for k, n in (('b', nd), ('bm', nm), ('sd', nd), ('sm', nm))}
    opt = torch.optim.LBFGS(list(ps.values()), max_iter=500, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        z = L + ps['b'][dI] + ps['bm'][mI] + (ps['sd'][dI] * xm if use_sd else 0) + (ps['sm'][mI] * xd if use_sm else 0)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction='sum') + lb * (ps['b'] ** 2).sum() + lm * (ps['bm'] ** 2).sum()
                + ls_d * (ps['sd'] ** 2).sum() + ls_m * (ps['sm'] ** 2).sum())
        loss.backward(); return loss
    opt.step(closure); return {k: v.detach().numpy() for k, v in ps.items()}
res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    tp = z['train_pos']; nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max())) + 1
    A = np.zeros((nm, nd)); A[tp[:, 0], tp[:, 1]] = 1; dgm = A.sum(1); dgd = A.sum(0)
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    q = A[P[:, 0], P[:, 1]]                       # hollow: the pair's own edge removed
    xm = np.log1p(dgm[P[:, 0]] - q); xd = np.log1p(dgd[P[:, 1]] - q)
    mu_m, mu_d = xm.mean(), xd.mean(); xm -= mu_m; xd -= mu_d
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    xmt = np.log1p(dgm[te[:, 0]]) - mu_m; xdt = np.log1p(dgd[te[:, 1]]) - mu_d
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    for key, sd_, sm_, lsd, lsm in (('bias', False, False, 0, 0), ('bias+slope_d', True, False, 10, 0), ('bias+slope_m', False, True, 0, 10),
                                    ('bias+both10', True, True, 10, 10), ('bias+both3', True, True, 3, 3), ('bias+both30', True, True, 30, 30)):
        p = fit(L, Y, P[:, 1], P[:, 0], xm, xd, nd, nm, 10, 10, lsd, lsm, sd_, sm_)
        s = lte + p['b'][te[:, 1]] + p['bm'][te[:, 0]] + (p['sd'][te[:, 1]] * xmt if sd_ else 0) + (p['sm'][te[:, 0]] * xdt if sm_ else 0)
        res.setdefault(key, []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base': continue
    m = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])]); w2 = np.mean([x[1] > y[1] for x, y in zip(v, res['base'])])
    print(f'  {k:16s} {m[0]:.4f}/{m[1]:.4f}  d {m[0]-bs[0]:+.4f}/{m[1]-bs[1]:+.4f}  win {w:.1f}/{w2:.1f}')
