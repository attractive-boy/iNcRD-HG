# -*- coding: utf-8 -*-
"""Round-13 main tables.  F = R12 final (valsel protocol); F+EB = F with the post-hoc empirical-Bayes random-intercept
head (ridge lam, fitted on the labelled train+val pairs with the network logit as offset).  The EB numbers are computed
from the --dump caches (identical to the in-run ebias_post) so that any lam can be tabulated without retraining; runs
without a dump (80/20 seeds 3-4, tag FE10) use their in-run ebp_* columns.

usage: python make_tables.py [lam]    -> tables_main.md"""
import os, sys, glob
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP

HERE = os.path.dirname(os.path.abspath(__file__))
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0


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


def fold_rows(tag, ds, split, seed):
    out = []
    for fn in sorted(glob.glob(os.path.join(HERE, 'cache', f'dump_{tag}_{ds}_{split}_s{seed}_f*.npz'))):
        z = np.load(fn); f = int(fn.split('_f')[-1].split('.')[0])
        nm = int(max(z['u'][:, 0].max(), z['te'][:, 0].max(), z['tr'][:, 0].max())) + 1
        nd = int(max(z['u'][:, 1].max(), z['te'][:, 1].max(), z['tr'][:, 1].max())) + 1
        P = np.concatenate([z['tr'], z['va']]); Y = np.concatenate([z['ytr'], z['yva']]); L = np.concatenate([z['logit_tr'], z['logit_va']])
        bm, bd = fit(L, Y, P[:, 0], P[:, 1], nm, nd, LAM)
        te, yte, lte = z['te'], z['yte'], z['logit_te']; s = lte + bm[te[:, 0]] + bd[te[:, 1]]
        out.append(dict(ds=ds, split=split, seed=seed, fold=f, F_auc=AU(yte, lte), F_aupr=AP(yte, lte), EB_auc=AU(yte, s), EB_aupr=AP(yte, s)))
    return out


rows = []
for ds in ('MDR', 'MDS'):
    for sd in (42, 43, 44):
        rows += fold_rows('FE3', ds, '5cv', sd)
    for sd in (0, 1, 2):
        rows += fold_rows('FE3', ds, 'single', sd)
    if LAM == 10.0:
        for sd in (3, 4):
            f = os.path.join(HERE, 'results', f'FE10_{ds}_single_seed{sd}.csv')
            if os.path.exists(f):
                d = pd.read_csv(f)
                for _, r in d.iterrows():
                    rows.append(dict(ds=ds, split='single', seed=sd, fold=int(r.fold), F_auc=r.valsel_auc, F_aupr=r.valsel_aupr, EB_auc=r.ebp_auc, EB_aupr=r.ebp_aupr))
X = pd.DataFrame(rows)
X.to_csv(os.path.join(HERE, f'per_fold_eb_lam{LAM:g}.csv'), index=False)
L_ = [f'# R13 main table (valsel protocol), EB head lam = {LAM:g}', '',
      '| board | n folds | F AUC / AUPR | F+EB AUC / AUPR | delta AUC / AUPR | wins AUC / AUPR |', '|---|---|---|---|---|---|']
for ds in ('MDR', 'MDS'):
    for split, name in (('5cv', '5cv x seeds 42-44'), ('single', '80/20 x seeds 0-4' if LAM == 10.0 else '80/20 x seeds 0-2')):
        x = X[(X.ds == ds) & (X.split == split)]
        if not len(x):
            continue
        L_.append(f'| {ds} {name} | {len(x)} | {x.F_auc.mean():.4f} / {x.F_aupr.mean():.4f} | {x.EB_auc.mean():.4f} / {x.EB_aupr.mean():.4f} | '
                  f'{(x.EB_auc - x.F_auc).mean():+.4f} / {(x.EB_aupr - x.F_aupr).mean():+.4f} | {int((x.EB_auc > x.F_auc).sum())} / {int((x.EB_aupr > x.F_aupr).sum())} |')
open(os.path.join(HERE, f'tables_main_lam{LAM:g}.md'), 'w').write('\n'.join(L_) + '\n')
print('\n'.join(L_))
