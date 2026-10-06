"""Offline probe 15: choose the EB ridge strength on the validation pairs (fit on train pairs only, pick the lam with the
best val AUC, refit on train + val), instead of fixing it."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'; split = sys.argv[3] if len(sys.argv) > 3 else '5cv'; seed = sys.argv[4] if len(sys.argv) > 4 else '42'
LAMS = (1.0, 3.0, 10.0, 30.0)


def fit(L, Y, m, d, nm, nd, lam):
    L = torch.tensor(L, dtype=torch.float64); Y = torch.tensor(Y, dtype=torch.float64); m = torch.tensor(m); d = torch.tensor(d)
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([bm, bd], max_iter=500, line_search_fn='strong_wolfe')

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(L + bm[m] + bd[d], Y, reduction='sum') + lam * (bm.pow(2).sum() + bd.pow(2).sum())
        l.backward(); return l
    opt.step(cl)
    return bm.detach().numpy(), bd.detach().numpy()


res = {}; chosen = []
import glob
files = sorted(glob.glob(f'cache/dump_{tag}_{ds}_{split}_s{seed}_f*.npz'))
for fn in files:
    z = np.load(fn)
    nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max(), z['tr'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max(), z['tr'][:, 1].max())) + 1
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    va_auc = {}
    for lam in LAMS:
        bm, bd = fit(z['logit_tr'], z['ytr'], z['tr'][:, 0], z['tr'][:, 1], nm, nd, lam)
        va_auc[lam] = AU(z['yva'], z['logit_va'] + bm[z['va'][:, 0]] + bd[z['va'][:, 1]])
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    for lam in LAMS:
        bm, bd = fit(L, Y, P[:, 0], P[:, 1], nm, nd, lam)
        s = lte + bm[te[:, 0]] + bd[te[:, 1]]; res.setdefault(f'lam{lam:g}', []).append((AU(yte, s), AP(yte, s)))
    lb = max(va_auc, key=va_auc.get); chosen.append(lb)
    res.setdefault('lam_valsel', []).append(res[f'lam{lb:g}'][-1])
bs = np.mean(res['base'], 0); print(f'{ds} {split} s{seed} base {bs[0]:.4f}/{bs[1]:.4f}  val-chosen lams {chosen}')
for k, v in res.items():
    if k == 'base':
        continue
    m_ = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])]); w2 = np.mean([x[1] > y[1] for x, y in zip(v, res['base'])])
    print(f'  {k:10s} {m_[0]:.4f}/{m_[1]:.4f}  d {m_[0]-bs[0]:+.4f}/{m_[1]-bs[1]:+.4f}  win {w:.1f}/{w2:.1f}')
