# -*- coding: utf-8 -*-
"""Unlabeled-pool negative augmentation (PU-style): add `mult` x n_pos extra unlabeled pairs from the released negative pool
(excluding every train/test pair of the protocol) as extra training negatives; class-balanced sample weights; test set unchanged.
Features: ctx+diff (bank, K=10 cross-fit) + PHL (prov + mprior). Seed-42 5cv."""
import os, sys, time, argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
for d in ('plmf_warm_20260922', 'plmf_innov_20260921', 'mphgnn_plmf_20260921', 'dlstmda_strongest_20260920'): sys.path.insert(0, os.path.join(BASE, d))
from run_probe import load_nodes, make_bank
from warm_scores import build_xy_k
from plmf_splits import get_split, load_dataset
from mprior import MPrior, build_extra
from prov import Prov, crossfit
from xrel import XRel
ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='MDR'); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--mult', type=float, default=1.0)
ap.add_argument('--inner', type=int, default=10); ap.add_argument('--C', type=float, default=0.3); ap.add_argument('--hidden', default='keep', help='keep|drop documented hidden positives from the extra negatives'); ap.add_argument('--prov', default='incrd', help='incrd|full provenance records'); ap.add_argument('--xrel', default='', help='extra label-free relation groups, e.g. cluster,geo'); ap.add_argument('--split', default='5cv'); a = ap.parse_args()
xr, xd = load_nodes(a.dataset); bank = make_bank(xr, xd, 'kernel', 10)
import prov as provmod
if a.prov == 'full': provmod.HERE = os.path.join(HERE, 'data', f'full_{a.dataset}')
pv = Prov(a.dataset, False, None, groups=['pmid', 'costudy', 'ctx', 'rec']); mp = MPrior(a.dataset)
provmod.HERE = os.path.join(BASE, 'plmf_warm_20260922')
xr = XRel(a.dataset, groups=a.xrel.split(',')) if a.xrel else None
mir, dru, pos, neg = load_dataset(a.dataset)
res = []; rows = []
for f in (range(5) if a.split == '5cv' else [0]):
    t0 = time.time()
    tr, ytr, te, yte = get_split(a.dataset, a.split, f, a.seed)
    used = set(map(tuple, tr)) | set(map(tuple, te))
    rng = np.random.RandomState(a.seed * 1000 + f)
    pool = neg[rng.permutation(len(neg))]
    extra = np.array([p for p in map(tuple, pool) if p not in used][:int(a.mult * (ytr == 1).sum())], dtype=np.int64).reshape(-1, 2)
    tr2 = np.concatenate([tr, extra]) if len(extra) else tr; ytr2 = np.concatenate([ytr, np.zeros(len(extra))])
    Xtr, Xte = build_xy_k(bank, tr2, ytr2, te, {'ctx', 'diff'}, a.seed * 100 + f, a.inner)
    Ptr, Pte = crossfit(pv, tr2, ytr2, te, a.seed * 100 + f, lambda o, s, q: o.transform(s, q), K=a.inner)
    Etr, Ete, _ = build_extra(mp, tr2, ytr2, te, {'fam', 'seed', 'hmdd', 'xdeg'}, a.seed * 100 + f, K=a.inner)
    Xtr = np.concatenate([Xtr, Ptr, Etr], 1); Xte = np.concatenate([Xte, Pte, Ete], 1)
    if xr is not None:
        Rtr, Rte = crossfit(xr, tr2, ytr2, te, a.seed * 100 + f, lambda o, s_, q: o.transform(s_, q), K=a.inner)
        Xtr = np.concatenate([Xtr, Rtr], 1); Xte = np.concatenate([Xte, Rte], 1)
    w = np.ones(len(ytr2)); w[ytr2 == 0] = (ytr2 == 1).sum() / (ytr2 == 0).sum()
    m = LogisticRegression(max_iter=5000, C=a.C).fit(Xtr, ytr2.astype(int), sample_weight=w); s = m.predict_proba(Xte)[:, 1]
    res.append((roc_auc_score(yte, s), average_precision_score(yte, s)))
    rows.append(pd.DataFrame({'fold': f, 'r': te[:, 0], 'd': te[:, 1], 'y': yte.astype(int), 'score': s}))
    print(f'  fold {f}: n_train {len(ytr2)} (extra {len(extra)}) AUC {res[-1][0]:.4f} AUPR {res[-1][1]:.4f} ({time.time() - t0:.0f}s)', flush=True)
res = np.array(res)
pd.concat(rows).to_csv(os.path.join(HERE, 'results', f'scores_{a.dataset}_{"single" if a.split == "single" else ""}s{a.seed}_K{a.inner}_{"phlf" if a.prov == "full" else "phl"}{("_" + a.xrel.replace(",", "-")) if a.xrel else ""}_extraneg{a.mult:g}.csv'), index=False)
line = f'## {a.dataset} {"single" if a.split == "single" else ""}s{a.seed} K{a.inner} {"phlf" if a.prov == "full" else "phl"}{("+" + a.xrel) if a.xrel else ""} + extra negatives x{a.mult:g}: AUC {res[:,0].mean():.4f}±{res[:,0].std():.4f} AUPR {res[:,1].mean():.4f}±{res[:,1].std():.4f}'
print(line, flush=True); open(os.path.join(HERE, 'results', 'extraneg.txt'), 'a').write(line + '\n')
