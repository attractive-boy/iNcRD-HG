"""Headroom check for entity-level corrections: intercepts fitted on the OTHER test folds' labelled pairs (same seed;
the 5 test folds are disjoint pair sets) and applied to the held-out test fold, on top of F (+ the train-fitted EB head).
This uses test labels of other folds -- an oracle-style diagnostic, never a model."""
import sys, glob, numpy as np, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
ds = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'FE3'; seed = sys.argv[3] if len(sys.argv) > 3 else '42'


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


Z = [np.load(fn) for fn in sorted(glob.glob(f'cache/dump_{tag}_{ds}_5cv_s{seed}_f*.npz'))]
nm = int(max(max(z['u'][:, 0].max(), z['te'][:, 0].max()) for z in Z)) + 1; nd = int(max(max(z['u'][:, 1].max(), z['te'][:, 1].max()) for z in Z)) + 1
# EB-corrected test logits per fold (train+val fit, lam 10)
EBL = []
for z in Z:
    P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
    bm, bd = fit(L, Y, P[:, 0], P[:, 1], nm, nd, 10.0)
    EBL.append(z['logit_te'] + bm[z['te'][:, 0]] + bd[z['te'][:, 1]])
out = {'F': [], 'F+EB': [], 'F+oracle': [], 'F+EB+oracle': []}
for f, z in enumerate(Z):
    oth = [g for g in range(5) if g != f]
    for base_name, base in (('F', [zz['logit_te'] for zz in Z]), ('F+EB', EBL)):
        P = np.concatenate([Z[g]['te'] for g in oth]); Y = np.concatenate([Z[g]['yte'] for g in oth]); L = np.concatenate([base[g] for g in oth])
        bm, bd = fit(L, Y, P[:, 0], P[:, 1], nm, nd, 10.0)
        s = base[f] + bm[z['te'][:, 0]] + bd[z['te'][:, 1]]
        out[base_name].append((AU(z['yte'], base[f]), AP(z['yte'], base[f])))
        out[base_name.replace('F+EB', 'F+EB+oracle') if base_name == 'F+EB' else 'F+oracle'].append((AU(z['yte'], s), AP(z['yte'], s)))
for k, v in out.items():
    m_ = np.mean(v, 0); print(f'{ds} s{seed} {k:12s} {m_[0]:.4f}/{m_[1]:.4f}')
