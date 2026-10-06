# -*- coding: utf-8 -*-
"""Study-grouped 5-fold split (same construction as plmf_warm_20260922/study_split.py) with external relations (XRel) on top of PHL.
Variants: base | phl | phl_<groups> | phl_x | *_shuf."""
import os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE); sys.path.insert(0, HERE)
for d in ('plmf_warm_20260922', 'plmf_innov_20260921', 'mphgnn_plmf_20260921', 'dlstmda_strongest_20260920'): sys.path.insert(0, os.path.join(BASE, d))
from study_split import study_folds
from run_probe import load_nodes, make_bank
from warm_scores import build_xy_k
from mprior import MPrior, build_extra
from prov import Prov, crossfit
from xrel import XRel
from final2 import parse_variant, PHL
ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='MDR'); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--variant', default='phl_x'); ap.add_argument('--inner', type=int, default=10); ap.add_argument('--records', default='incrd', help='incrd|full: which provenance records define the study components'); ap.add_argument('--group', default='component', help='component|primary: connected components of the pair-PMID graph, or GroupKFold on each pair\'s primary (smallest) study'); a = ap.parse_args()
head, groups, shuf = parse_variant(a.variant)
import study_split as ssmod
if a.records == 'full': ssmod.HERE = os.path.join(HERE, 'data', f'full_{a.dataset}')      # components from the full ncRNADrug records (95% of positives)
def primary_folds(ds, seed, n_folds=5):
    """Each positive is assigned to its primary study = the PMID (among its records) that reports the fewest pairs (most specific study);
    pairs without records get their own singleton group. Groups are dealt to folds greedily by size (largest first) for balance."""
    import pickle
    from plmf_splits import load_dataset, _balanced_pairs
    mir, dru, pos, neg = load_dataset(ds)
    names = mir['miRNA_Name'].astype(str).str.upper().values; dbs = dru['DrugBank_ID'].astype(str).values
    recs = pickle.load(open(os.path.join(ssmod.HERE, 'data', 'pair_records.pkl'), 'rb'))
    pm = {k: recs[(names[i], dbs[j])]['pmids'] for k, (i, j) in enumerate(map(tuple, pos)) if (names[i], dbs[j]) in recs}
    size = {}
    for v in pm.values():
        for p in v: size[p] = size.get(p, 0) + 1
    grp = np.array([min(pm[k], key=lambda p: (size[p], p)) if k in pm else -(k + 1) for k in range(len(pos))])
    rng = np.random.RandomState(seed); sizes = pd.Series(grp).value_counts()
    order = sizes.index.values.copy(); rng.shuffle(order); order = sorted(order, key=lambda c: -sizes[c])
    fold_of, load = {}, np.zeros(n_folds)
    for c in order:
        f = int(np.argmin(load + rng.rand(n_folds) * 0.5)); fold_of[c] = f; load[f] += sizes[c]
    pos_fold = np.array([fold_of[c] for c in grp])
    idx, y = _balanced_pairs(pos, neg, seed); neg_sel = idx[y == 0]; neg_fold = rng.randint(0, n_folds, len(neg_sel))
    print(f'{ds}: primary-study groups {len(sizes)}, largest {sizes.max()} pairs ({sizes.max()/len(pos):.2f}), fold sizes {np.bincount(pos_fold)}')
    return pos, pos_fold, neg_sel, neg_fold
import pandas as pd
pos, pf, negs, nf = study_folds(a.dataset, a.seed) if a.group == 'component' else primary_folds(a.dataset, a.seed)
ssmod.HERE = os.path.join(BASE, 'plmf_warm_20260922')
tagr = ('' if a.records == 'incrd' else 'F') + ('' if a.group == 'component' else 'P')
xr_, xd_ = load_nodes(a.dataset); bank = make_bank(xr_, xd_, 'kernel', 10)
import prov as provmod
if head in ('phlf', 'phlfs', 'provf'): provmod.HERE = os.path.join(HERE, 'data', f'full_{a.dataset}')
pv = Prov(a.dataset, head == 'phlfs', np.random.RandomState(a.seed), groups=PHL['prov'].split(',')) if head in ('phl', 'phlf', 'phlfs', 'provf') else None
provmod.HERE = os.path.join(BASE, 'plmf_warm_20260922')
mp = MPrior(a.dataset) if head in ('phl', 'phlf', 'phlfs') else None
xr = XRel(a.dataset, groups=groups, shuffle=groups if shuf else None, rng=np.random.RandomState(a.seed + 7)) if groups else None
res = []
for f in range(5):
    tr = np.concatenate([pos[pf != f], negs[nf != f]]); ytr = np.concatenate([np.ones((pf != f).sum()), np.zeros((nf != f).sum())])
    te = np.concatenate([pos[pf == f], negs[nf == f]]); yte = np.concatenate([np.ones((pf == f).sum()), np.zeros((nf == f).sum())])
    cf = os.path.join(HERE, 'cache', f'study{tagr}_{a.dataset}_s{a.seed}_f{f}_K{a.inner}.npz')     # ctx+diff (85) + phl (24 prov + 19 mprior)
    if os.path.exists(cf):
        z = np.load(cf); Xtr, Xte = z['Xtr'], z['Xte']
    else:
        Xtr, Xte = build_xy_k(bank, tr, ytr, te, {'ctx', 'diff'}, a.seed * 100 + f, a.inner)
        pv0 = Prov(a.dataset, False, None, groups=PHL['prov'].split(',')); mp0 = MPrior(a.dataset)
        Ptr, Pte = crossfit(pv0, tr, ytr, te, a.seed * 100 + f, lambda o, s, q: o.transform(s, q), K=a.inner)
        Etr, Ete, _ = build_extra(mp0, tr, ytr, te, set(PHL['mprior'].split(',')), a.seed * 100 + f, K=a.inner)
        Xtr = np.concatenate([Xtr, Ptr, Etr], 1); Xte = np.concatenate([Xte, Pte, Ete], 1)
        np.savez_compressed(cf, Xtr=Xtr, Xte=Xte)
    if head == 'base': Xtr, Xte = Xtr[:, :85], Xte[:, :85]
    elif head in ('phlf', 'phlfs', 'provf'):
        Ptr, Pte = crossfit(pv, tr, ytr, te, a.seed * 100 + f, lambda o, s, q: o.transform(s, q), K=a.inner)
        if head == 'provf': Xtr, Xte = np.concatenate([Xtr[:, :85], Ptr], 1), np.concatenate([Xte[:, :85], Pte], 1)
        else: Xtr, Xte = np.concatenate([Xtr[:, :85], Ptr, Xtr[:, 109:128]], 1), np.concatenate([Xte[:, :85], Pte, Xte[:, 109:128]], 1)
    if xr is not None:
        Rtr, Rte = crossfit(xr, tr, ytr, te, a.seed * 100 + f, lambda o, s, q: o.transform(s, q), K=a.inner)
        Xtr = np.concatenate([Xtr, Rtr], 1); Xte = np.concatenate([Xte, Rte], 1)
    m = LogisticRegression(max_iter=5000, C=0.3).fit(Xtr, ytr.astype(int)); s = m.predict_proba(Xte)[:, 1]
    res.append((roc_auc_score(yte, s), average_precision_score(yte, s)))
res = np.array(res)
line = f'## STUDY-GROUPED{tagr} {a.dataset} s{a.seed} K{a.inner} {a.variant} ncol={Xtr.shape[1]}: AUC {res[:,0].mean():.4f}±{res[:,0].std():.4f} AUPR {res[:,1].mean():.4f}±{res[:,1].std():.4f}'
print(line, flush=True); open(os.path.join(HERE, 'results', 'study_split.txt'), 'a').write(line + '\n')
