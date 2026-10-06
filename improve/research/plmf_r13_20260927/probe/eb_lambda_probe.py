"""Offline probe 13: separate ridge strengths for miRNA and drug random intercepts (post-hoc EB head on Fd dumps)."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'Fd'
grid = [(lm, ld) for lm in (0.3, 1, 3) for ld in (1, 3, 10, 30)]
res = {}
for f in range(5):
    z = np.load(f'cache/dump_{tag}_{ds}_5cv_s42_f{f}.npz')
    nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max())) + 1; nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max())) + 1
    P = np.concatenate([z['tr'], z['va']]); Y = torch.tensor(np.concatenate([z['ytr'], z['yva']]), dtype=torch.float64)
    L = torch.tensor(np.concatenate([z['logit_tr'], z['logit_va']]), dtype=torch.float64); m = torch.tensor(P[:, 0]); d = torch.tensor(P[:, 1])
    te, yte, lte = z['te'], z['yte'], z['logit_te']
    res.setdefault('base', []).append((AU(yte, lte), AP(yte, lte)))
    for lm, ld in grid:
        bm = torch.zeros(nm, dtype=torch.float64, requires_grad=True); bd = torch.zeros(nd, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([bm, bd], max_iter=500, line_search_fn='strong_wolfe')

        def cl():
            opt.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(L + bm[m] + bd[d], Y, reduction='sum') + lm * bm.pow(2).sum() + ld * bd.pow(2).sum()
            l.backward(); return l
        opt.step(cl)
        s = lte + bm.detach().numpy()[te[:, 0]] + bd.detach().numpy()[te[:, 1]]
        res.setdefault((lm, ld), []).append((AU(yte, s), AP(yte, s)))
bs = np.mean(res['base'], 0); print(f'{ds} base {bs[0]:.4f}/{bs[1]:.4f}   rows: lam_m, cols: lam_d = 1 3 10 30  (dAUC/dAUPR)')
for lm in (0.3, 1, 3):
    print(f'  lam_m={lm:4.1f}: ' + '  '.join(f'{np.mean(res[(lm, ld)], 0)[0]-bs[0]:+.4f}/{np.mean(res[(lm, ld)], 0)[1]-bs[1]:+.4f}' for ld in (1, 3, 10, 30)))
