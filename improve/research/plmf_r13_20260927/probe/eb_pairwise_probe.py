"""Offline probe 17: entity intercepts fitted with a global pairwise (AUC-surrogate) objective instead of BCE:
minimise mean_{pos p, neg n} log(1 + exp(-(s_p - s_n))) + lam * ||b||^2 / N over sampled labelled pos-neg pairs,
s = logit + b_m + b_d.  Compared with the BCE-fitted EB head."""
import sys, glob, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'FE3'; seed = sys.argv[3] if len(sys.argv) > 3 else '42'


def fit_bce(L, Y, m, d, nm, nd, lam=10.0):
    L = torch.tensor(L, dtype=torch.float64); Y = torch.tensor(Y, dtype=torch.float64); m = torch.tensor(m); d = torch.tensor(d)
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([bm, bd], max_iter=500, line_search_fn='strong_wolfe')

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(L + bm[m] + bd[d], Y, reduction='sum') + lam * (bm.pow(2).sum() + bd.pow(2).sum())
        l.backward(); return l
    opt.step(cl)
    return bm.detach().numpy(), bd.detach().numpy()


def fit_pair(L, Y, m, d, nm, nd, lam, n_pairs=400000, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = torch.tensor(L, dtype=torch.float64); m = torch.tensor(m); d = torch.tensor(d)
    pi = torch.tensor(np.where(Y == 1)[0]); ni = torch.tensor(np.where(Y == 0)[0])
    a = pi[torch.randint(0, len(pi), (n_pairs,), generator=g)]; b = ni[torch.randint(0, len(ni), (n_pairs,), generator=g)]
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([bm, bd], max_iter=300, line_search_fn='strong_wolfe')
    N = len(Y)

    def cl():
        opt.zero_grad()
        s = L + bm[m] + bd[d]
        l = torch.nn.functional.softplus(-(s[a] - s[b])).mean() + lam / N * (bm.pow(2).sum() + bd.pow(2).sum())
        l.backward(); return l
    opt.step(cl)
    return bm.detach().numpy(), bd.detach().numpy()


res = {}
for fn in sorted(glob.glob(f'cache/dump_{tag}_{ds}_5cv_s{seed}_f*.npz')):
    z = np.load(fn)
    nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max(), z['tr'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max(), z['tr'][:, 1].max())) + 1
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    bm, bd = fit_bce(L, Y, P[:, 0], P[:, 1], nm, nd); s = lte + bm[te[:, 0]] + bd[te[:, 1]]; res.setdefault('EB_bce', []).append((AU(yte, s), AP(yte, s)))
    for lam in (3.0, 10.0, 30.0):
        bm, bd = fit_pair(L, Y, P[:, 0], P[:, 1], nm, nd, lam); s = lte + bm[te[:, 0]] + bd[te[:, 1]]
        res.setdefault(f'EB_pair{lam:g}', []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} s{seed} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base':
        continue
    m_ = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])])
    print(f'  {k:10s} d {m_[0]-bs[0]:+.4f}/{m_[1]-bs[1]:+.4f}  winAUC {w:.1f}')
