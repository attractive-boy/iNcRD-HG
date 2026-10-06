"""Offline probe 10: post-hoc entity random effects -- fitted on train only vs train+val, and combined with the column
quota calibration of probe/quota_probe.py."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
from scipy.special import expit
from scipy.optimize import brentq
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'


def fit(L, Y, m, d, nm, nd, lam=10.0):
    L = torch.tensor(L, dtype=torch.float64); Y = torch.tensor(Y, dtype=torch.float64); m = torch.tensor(m); d = torch.tensor(d)
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([bm, bd], max_iter=500, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(L + bm[m] + bd[d], Y, reduction='sum') + lam * (bm.pow(2).sum() + bd.pow(2).sum())
        loss.backward(); return loss
    opt.step(closure)
    return bm.detach().numpy(), bd.detach().numpy()


res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    tp = z['train_pos']; nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max(), z['tr'][:, 0].max())) + 1
    nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max(), z['tr'][:, 1].max())) + 1
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    bm, bd = fit(z['logit_tr'], z['ytr'], z['tr'][:, 0], z['tr'][:, 1], nm, nd)
    s = lte + bm[te[:, 0]] + bd[te[:, 1]]; res.setdefault('eb_train', []).append((AU(yte, s), AP(yte, s)))
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    bm, bd = fit(L, Y, P[:, 0], P[:, 1], nm, nd)
    s = lte + bm[te[:, 0]] + bd[te[:, 1]]; res.setdefault('eb_trval', []).append((AU(yte, s), AP(yte, s)))
    # + column quota on the bias-corrected full matrix
    Lf = np.full((nm, nd), np.nan)
    for part in ('u', 'tr', 'va', 'te'):
        Q = z[part]; Lf[Q[:, 0], Q[:, 1]] = z[f'logit_{part}']
    Lf = Lf + bm[:, None] + bd[None, :]
    A = np.zeros((nm, nd)); A[tp[:, 0], tp[:, 1]] = 1; NG = A == 0; hid = 0.25
    c = brentq(lambda c: expit(Lf[NG] + c).sum() - hid * A.sum(), -30, 30)
    B = np.zeros(nd)
    for dd in range(nd):
        ng = NG[:, dd]; tgt = max(hid * A[:, dd].sum(), 0.25)
        B[dd] = brentq(lambda b: expit(Lf[ng, dd] + c + b).sum() - tgt, -40, 40)
    for lam in (0.25, 0.5):
        s2 = s + lam * B[te[:, 1]]; res.setdefault(f'eb_trval+quota{lam}', []).append((AU(yte, s2), AP(yte, s2)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base':
        continue
    m_ = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])]); w2 = np.mean([x[1] > y[1] for x, y in zip(v, res['base'])])
    print(f'  {k:20s} {m_[0]:.4f}/{m_[1]:.4f}  d {m_[0]-bs[0]:+.4f}/{m_[1]-bs[1]:+.4f}  win {w:.1f}/{w2:.1f}')
