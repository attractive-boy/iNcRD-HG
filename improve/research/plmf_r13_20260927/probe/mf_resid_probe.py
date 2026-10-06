"""Offline probe 8: beyond entity biases, does a low-rank residual interaction (rank-k MF with ridge, fitted on the
labelled pairs with the F logit as offset) add anything?"""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'


def fit(L, Y, m, d, nm, nd, k, lam_b=10.0, lam_f=10.0, seed=0):
    torch.manual_seed(seed)
    L = torch.tensor(L, dtype=torch.float64); Y = torch.tensor(Y, dtype=torch.float64); m = torch.tensor(m); d = torch.tensor(d)
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    U = (0.01 * torch.randn(nm, max(k, 1), dtype=torch.float64)).requires_grad_(True); V = (0.01 * torch.randn(nd, max(k, 1), dtype=torch.float64)).requires_grad_(True)
    ps = [bm, bd] + ([U, V] if k else [])
    opt = torch.optim.LBFGS(ps, max_iter=800, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = L + bm[m] + bd[d] + ((U[m] * V[d]).sum(1) if k else 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction='sum') + lam_b * (bm.pow(2).sum() + bd.pow(2).sum())
        if k:
            loss = loss + lam_f * (U.pow(2).sum() + V.pow(2).sum())
        loss.backward(); return loss
    opt.step(closure)
    return [p.detach().numpy() for p in (bm, bd, U, V)]


res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max())) + 1
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    for k, lf in ((0, 10), (2, 3), (2, 10), (4, 10), (4, 30), (8, 30)):
        bm, bd, U, V = fit(L, Y, P[:, 0], P[:, 1], nm, nd, k, 10.0, lf)
        s = lte + bm[te[:, 0]] + bd[te[:, 1]] + ((U[te[:, 0]] * V[te[:, 1]]).sum(1) if k else 0)
        res.setdefault(f'k{k}_lf{lf}', []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for kk, v in res.items():
    if kk == 'base':
        continue
    m_ = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])])
    print(f'  {kk:10s} {m_[0]:.4f}/{m_[1]:.4f}  d {m_[0]-bs[0]:+.4f}/{m_[1]-bs[1]:+.4f}  winAUC {w:.1f}')
