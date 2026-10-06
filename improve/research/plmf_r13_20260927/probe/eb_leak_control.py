"""Leak control for the post-hoc EB head.  The benchmark's training negatives are drawn from the official negative pool
(= complement of ALL positives), so the number of training negatives of an entity carries (weak) information on how many
test positives it has.  Control: refit the intercepts with the training(+val) negatives replaced by an equal number of
cells drawn uniformly from every cell outside the message graph and the training/val pairs (test cells included, no
knowledge of which cells are positives).  If the gain survives, it is entity idiosyncrasy, not the sampling artifact."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'; lam = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0


def fit(L, Y, m, d, nm, nd):
    L = torch.tensor(L, dtype=torch.float64); Y = torch.tensor(Y, dtype=torch.float64); m = torch.tensor(m); d = torch.tensor(d)
    bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([bm, bd], max_iter=500, line_search_fn='strong_wolfe')

    def cl():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(L + bm[m] + bd[d], Y, reduction='sum') + lam * (bm.pow(2).sum() + bd.pow(2).sum())
        l.backward(); return l
    opt.step(cl)
    return bm.detach().numpy(), bd.detach().numpy()


res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max(), z['tr'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max(), z['tr'][:, 1].max())) + 1
    Lf = np.full((nm, nd), np.nan)
    for part in ('u', 'tr', 'va', 'te'):
        Q = z[part]; Lf[Q[:, 0], Q[:, 1]] = z[f'logit_{part}']
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    bm, bd = fit(L, Y, P[:, 0], P[:, 1], nm, nd)
    s = lte + bm[te[:, 0]] + bd[te[:, 1]]; res.setdefault('eb_official_negs', []).append((AU(yte, s), AP(yte, s)))
    # strict negatives: uniform over cells outside graph and outside train/val pairs (u and test cells)
    cand = np.concatenate([z['u'], te])
    for rep in range(3):
        rng = np.random.RandomState(100 * f + rep)
        nneg = int((Y == 0).sum())
        pick = cand[rng.choice(len(cand), nneg, replace=False)]
        Ps = np.concatenate([P[Y == 1], pick]); Ys = np.concatenate([np.ones((Y == 1).sum()), np.zeros(nneg)])
        Ls = np.concatenate([L[Y == 1], Lf[pick[:, 0], pick[:, 1]]])
        bm2, bd2 = fit(Ls, Ys, Ps[:, 0], Ps[:, 1], nm, nd)
        s2 = lte + bm2[te[:, 0]] + bd2[te[:, 1]]; res.setdefault(f'eb_strict_negs_r{rep}', []).append((AU(yte, s2), AP(yte, s2)))
    # positives-only information: intercepts fitted with negatives drawn uniformly from ALL cells except graph edges
bs = np.mean(res['base'], 0); print(f'{ds} lam={lam} base {bs[0]:.4f}/{bs[1]:.4f}')
for k, v in res.items():
    if k == 'base':
        continue
    m_ = np.mean(v, 0); w = np.mean([x[0] > y[0] for x, y in zip(v, res['base'])]); w2 = np.mean([x[1] > y[1] for x, y in zip(v, res['base'])])
    print(f'  {k:20s} {m_[0]:.4f}/{m_[1]:.4f}  d {m_[0]-bs[0]:+.4f}/{m_[1]-bs[1]:+.4f}  win {w:.1f}/{w2:.1f}')
