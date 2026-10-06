"""Offline probe 12: is the post-hoc random-intercept gain just a smooth popularity correction?  Compare
(a) free per-entity intercepts (ridge), (b) intercepts restricted to a + b*log(1+deg) per side, (c) (b) + free residual
intercepts -- all fitted on the labelled train+val pairs with the F logit as offset."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    tp = z['train_pos']; nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max())) + 1
    A = np.zeros((nm, nd)); A[tp[:, 0], tp[:, 1]] = 1
    P = np.concatenate([z['tr'], z['va']]); Y = torch.tensor(np.concatenate([z['ytr'], z['yva']]), dtype=torch.float64)
    q = A[P[:, 0], P[:, 1]]                                             # hollow degrees for the labelled pairs
    hdm = torch.tensor(np.log1p(A.sum(1)[P[:, 0]] - q)); hdd = torch.tensor(np.log1p(A.sum(0)[P[:, 1]] - q))
    tdm = torch.tensor(np.log1p(A.sum(1)[z['te'][:, 0]])); tdd = torch.tensor(np.log1p(A.sum(0)[z['te'][:, 1]]))
    L = torch.tensor(np.concatenate([z['logit_tr'], z['logit_va']]), dtype=torch.float64); m = torch.tensor(P[:, 0]); d = torch.tensor(P[:, 1])
    te, yte, lte = z['te'], z['yte'], z['logit_te']; mt = torch.tensor(te[:, 0]); dt = torch.tensor(te[:, 1])
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    for kind in ('free', 'deg', 'deg+free'):
        bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
        c = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        ps = ([bm, bd] if kind != 'deg' else []) + ([c] if kind != 'free' else [])
        opt = torch.optim.LBFGS(ps, max_iter=500, line_search_fn='strong_wolfe')

        def off(mi, di):
            o = 0
            if kind != 'free':
                xm, xd = (hdm, hdd) if len(mi) == len(P) else (tdm, tdd)
                o = o + c[0] + c[1] * xm + c[2] * xd + c[3] * xm * xd
            if kind != 'deg':
                o = o + bm[mi] + bd[di]
            return o

        def cl():
            opt.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(L + off(m, d), Y, reduction='sum')
            if kind != 'deg':
                l = l + 10 * (bm.pow(2).sum() + bd.pow(2).sum())
            l.backward(); return l
        opt.step(cl)
        with torch.no_grad():
            s = (torch.tensor(lte, dtype=torch.float64) + off(mt, dt)).numpy()
        res.setdefault(kind, []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base':
        continue
    m_ = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])])
    print(f'  {k:10s} {m_[0]:.4f}/{m_[1]:.4f}  d {m_[0]-bs[0]:+.4f}/{m_[1]-bs[1]:+.4f}  winAUC {w:.1f}')
