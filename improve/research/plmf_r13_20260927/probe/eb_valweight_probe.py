"""Offline probe 14: EB head -- weight the out-of-sample validation pairs more than the (in-sample) training pairs."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'; lam = 3.0
res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max(), z['tr'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max(), z['tr'][:, 1].max())) + 1
    P = np.concatenate([z['tr'], z['va']]); Y = torch.tensor(np.concatenate([z['ytr'], z['yva']]), dtype=torch.float64)
    L = torch.tensor(np.concatenate([z['logit_tr'], z['logit_va']]), dtype=torch.float64); m = torch.tensor(P[:, 0]); d = torch.tensor(P[:, 1])
    isva = torch.tensor(np.r_[np.zeros(len(z['tr'])), np.ones(len(z['va']))], dtype=torch.float64)
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    for kappa in (0.0, 1.0, 2.0, 4.0, 8.0):
        w = 1 + (kappa - 1) * isva if kappa > 0 else 1 - isva
        bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([bm, bd], max_iter=500, line_search_fn='strong_wolfe')

        def cl():
            opt.zero_grad()
            l = (w * torch.nn.functional.binary_cross_entropy_with_logits(L + bm[m] + bd[d], Y, reduction='none')).sum() + lam * (bm.pow(2).sum() + bd.pow(2).sum())
            l.backward(); return l
        opt.step(cl)
        s = lte + bm.detach().numpy()[te[:, 0]] + bd.detach().numpy()[te[:, 1]]
        res.setdefault(f'kappa{kappa:g}', []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}  (kappa = weight of val pairs; 0 = train only)')
for k, v in res.items():
    if k == 'base':
        continue
    m_ = np.mean(v, 0); print(f'  {k:8s} {m_[0]:.4f}/{m_[1]:.4f}  d {m_[0]-bs[0]:+.4f}/{m_[1]-bs[1]:+.4f}')
