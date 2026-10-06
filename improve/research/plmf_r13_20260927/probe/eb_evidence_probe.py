"""Offline probe 16: choose the EB prior variances by the Laplace-approximated marginal likelihood of the labelled
(train+val) pairs -- a fully data-driven empirical-Bayes head, no test data and no validation split needed.
Prior b_m ~ N(0, 1/(2 lam_m)), b_d ~ N(0, 1/(2 lam_d)); diagonal (per-intercept) Hessian in the log-determinant."""
import sys, glob, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2]; split = sys.argv[3]; seed = sys.argv[4]
GRID = [(lm, ld) for lm in (3.0, 10.0, 30.0) for ld in (1.0, 3.0, 10.0, 30.0)]


def fit(L, Y, m, d, nm, nd, lm, ld):
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([bm, bd], max_iter=500, line_search_fn='strong_wolfe')

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(L + bm[m] + bd[d], Y, reduction='sum') + lm * bm.pow(2).sum() + ld * bd.pow(2).sum()
        l.backward(); return l
    opt.step(cl)
    with torch.no_grad():
        z = L + bm[m] + bd[d]; p = torch.sigmoid(z); w = p * (1 - p)
        ll = -torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction='sum')
        Hm = torch.zeros(nm, dtype=torch.float64).index_add_(0, m, w) + 2 * lm; Hd = torch.zeros(nd, dtype=torch.float64).index_add_(0, d, w) + 2 * ld
        # log evidence (Laplace): ll + log prior(b*) - 0.5 log det H + 0.5 * n * log(2 pi)  (the 2pi terms cancel)
        logprior = (-lm * bm.pow(2).sum() + 0.5 * nm * np.log(2 * lm)) + (-ld * bd.pow(2).sum() + 0.5 * nd * np.log(2 * ld))
        ev = ll + logprior - 0.5 * (torch.log(Hm).sum() + torch.log(Hd).sum())
    return bm.detach().numpy(), bd.detach().numpy(), float(ev)


res = {}; chosen = []
for fn in sorted(glob.glob(f'cache/dump_{tag}_{ds}_{split}_s{seed}_f*.npz')):
    z = np.load(fn)
    nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max(), z['tr'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max(), z['tr'][:, 1].max())) + 1
    P = np.concatenate([z['tr'], z['va']]); Y = torch.tensor(np.concatenate([z['ytr'], z['yva']]), dtype=torch.float64)
    L = torch.tensor(np.concatenate([z['logit_tr'], z['logit_va']]), dtype=torch.float64); m = torch.tensor(P[:, 0]); d = torch.tensor(P[:, 1])
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    evs = {}
    for lm, ld in GRID:
        bm, bd, ev = fit(L, Y, m, d, nm, nd, lm, ld)
        s = lte + bm[te[:, 0]] + bd[te[:, 1]]
        res.setdefault(f'{lm:g}/{ld:g}', []).append((AU(yte, s), AP(yte, s))); evs[(lm, ld)] = ev
    b = max(evs, key=evs.get); chosen.append(b)
    res.setdefault('evidence', []).append(res[f'{b[0]:g}/{b[1]:g}'][-1])
bs = np.mean(res['base'], 0)
print(f'{ds} {split} s{seed} base {bs[0]:.4f}/{bs[1]:.4f}  evidence-chosen (lam_m, lam_d): {chosen}')
for k in ('10/10', '3/3', '10/3', '30/3', 'evidence'):
    m_ = np.mean(res[k], 0); print(f'  {k:9s} d {m_[0]-bs[0]:+.4f}/{m_[1]-bs[1]:+.4f}')
