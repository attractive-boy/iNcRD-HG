"""Offline probe 11: drug-specific (and miRNA-specific) random slopes on the penultimate state h (low-rank per-entity heads):
logit' = logit + b_m + b_d + h . (V e_d) with V [H x r], e_d per-drug r-vector, ridge on both -- fitted on labelled pairs."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'


def fit(L, Hh, Y, m, d, nm, nd, r, lam_b=10.0, lam_s=10.0, seed=0):
    torch.manual_seed(seed)
    T = lambda v: torch.tensor(v, dtype=torch.float64)
    L, Hh, Y = T(L), T(Hh), T(Y); m = torch.tensor(m); d = torch.tensor(d)
    Hh = (Hh - Hh.mean(0)) / (Hh.std(0) + 1e-6)
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    V = (0.01 * torch.randn(Hh.shape[1], max(r, 1), dtype=torch.float64)).requires_grad_(True)
    E = (0.01 * torch.randn(nd, max(r, 1), dtype=torch.float64)).requires_grad_(True)
    ps = [bm, bd] + ([V, E] if r else [])
    opt = torch.optim.LBFGS(ps, max_iter=600, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = L + bm[m] + bd[d] + (((Hh @ V) * E[d]).sum(1) if r else 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction='sum') + lam_b * (bm.pow(2).sum() + bd.pow(2).sum())
        if r:
            loss = loss + lam_s * (V.pow(2).sum() + E.pow(2).sum())
        loss.backward(); return loss
    opt.step(closure)
    mu, sdv = None, None
    return bm.detach(), bd.detach(), V.detach(), E.detach()


res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max())) + 1
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    Hh = np.concatenate([z['h_tr'], z['h_va']]); mu, sd = Hh.mean(0), Hh.std(0) + 1e-6
    te, yte, lte = z['te'], z['yte'], z['logit_te']; Ht = torch.tensor((z['h_te'] - mu) / sd, dtype=torch.float64)
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    for r, ls in ((0, 10), (1, 10), (2, 10), (2, 100), (4, 100)):
        bm, bd, V, E = fit(L, Hh, Y, P[:, 0], P[:, 1], nm, nd, r, 10.0, ls)
        s = torch.tensor(lte, dtype=torch.float64) + bm[te[:, 0]] + bd[te[:, 1]] + (((Ht @ V) * E[te[:, 1]]).sum(1) if r else 0)
        s = s.numpy(); res.setdefault(f'r{r}_ls{ls}', []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base':
        continue
    m_ = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])])
    print(f'  {k:10s} {m_[0]:.4f}/{m_[1]:.4f}  d {m_[0]-bs[0]:+.4f}/{m_[1]-bs[1]:+.4f}  winAUC {w:.1f}')
