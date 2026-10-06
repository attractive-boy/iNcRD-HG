"""Offline probe: per-entity affine calibration of the F logit fitted on the labelled training (+val) pairs with
ridge shrinkage toward identity (a=1, b=0): logit' = a_d * logit + b_d (+ b_m).  Test labels only for scoring."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
def fit(L, Y, dI, mI, nd, nm, lam_b, lam_a, lam_m, use_m, epochs=300):
    L = torch.tensor(L, dtype=torch.float64); Y = torch.tensor(Y, dtype=torch.float64)
    dI = torch.tensor(dI); mI = torch.tensor(mI)
    b = torch.zeros(nd, dtype=torch.float64, requires_grad=True); a = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([b, a, bm], max_iter=epochs, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        z = (1 + a[dI]) * L + b[dI] + (bm[mI] if use_m else 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction='sum') + lam_b * (b ** 2).sum() + lam_a * (a ** 2).sum() + lam_m * (bm ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return a.detach().numpy(), b.detach().numpy(), bm.detach().numpy()
res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max(), z['tr'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max(), z['tr'][:, 1].max())) + 1
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    for lam_b in (3, 10, 30):
        for lam_a, use_m, lam_m in ((1e9, False, 0), (10, False, 0), (1e9, True, 10), (1e9, True, 30)):
            a, b, bm = fit(L, Y, P[:, 1], P[:, 0], nd, nm, lam_b, lam_a, lam_m, use_m)
            s = (1 + a[te[:, 1]]) * lte + b[te[:, 1]] + (bm[te[:, 0]] if use_m else 0)
            res.setdefault(f'b{lam_b}_a{lam_a:g}_m{int(use_m)}{lam_m}', []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base': continue
    m = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])]); w2 = np.mean([x[1] > y[1] for x, y in zip(v, res['base'])])
    print(f'  {k:22s} {m[0]:.4f}/{m[1]:.4f}  d {m[0]-bs[0]:+.4f}/{m[1]-bs[1]:+.4f}  win {w:.1f}/{w2:.1f}')
