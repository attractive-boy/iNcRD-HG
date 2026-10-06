# -*- coding: utf-8 -*-
"""Supplement: study-grouped 5-fold split. Positives that share a PMID (connected components of the
pair-PMID bipartite graph) are kept in the same fold; positives without records are assigned at random;
negatives sampled/assigned exactly as the paper protocol. Runs base and PHL on this split so the
'same-study transitivity' contribution can be quantified."""
import os, sys, pickle, argparse, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'plmf_innov_20260921'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'mphgnn_plmf_20260921'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'dlstmda_strongest_20260920'))
from plmf_splits import load_dataset, _balanced_pairs
from run_probe import load_nodes, make_bank
from warm_scores import build_xy_k
from mprior import MPrior, build_extra
from prov import Prov, crossfit

def study_folds(ds, seed, n_folds=5):
    mir, dru, pos, neg = load_dataset(ds)
    names = mir['miRNA_Name'].astype(str).str.upper().values; dbs = dru['DrugBank_ID'].astype(str).values
    recs = pickle.load(open(os.path.join(HERE, 'data', 'pair_records.pkl'), 'rb'))
    pm = {}
    for k, (i, j) in enumerate(map(tuple, pos)):
        v = recs.get((names[i], dbs[j]))
        if v: pm[k] = v['pmids']
    pmids = sorted(set(p for v in pm.values() for p in v)); pid = {p: i for i, p in enumerate(pmids)}
    rows, cols = [], []
    for k, v in pm.items():
        for p in v: rows.append(k); cols.append(len(pos) + pid[p])
    n = len(pos) + len(pmids)
    G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, lab = connected_components(G + G.T, directed=False)
    comp = lab[:len(pos)]
    rng = np.random.RandomState(seed)
    # assign components to folds greedily by size (largest first) to balance
    sizes = pd.Series(comp).value_counts()
    order = sizes.index.values.copy(); rng.shuffle(order); order = sorted(order, key=lambda c: -sizes[c])
    fold_of_comp, load = {}, np.zeros(n_folds)
    for c in order:
        f = int(np.argmin(load + rng.rand(n_folds) * 0.5)); fold_of_comp[c] = f; load[f] += sizes[c]
    pos_fold = np.array([fold_of_comp[c] for c in comp])
    idx, y = _balanced_pairs(pos, neg, seed)          # same negatives as the paper protocol (seed)
    neg_sel = idx[y == 0]; neg_fold = rng.randint(0, n_folds, len(neg_sel))
    print(f'{ds}: {len(pmids)} PMIDs, largest component {sizes.max()} pairs ({sizes.max()/len(pos):.2f}), fold sizes {np.bincount(pos_fold)}')
    return pos, pos_fold, neg_sel, neg_fold

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='MDR'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--variant', default='final'); ap.add_argument('--inner', type=int, default=10); a = ap.parse_args()
    pos, pf, negs, nf = study_folds(a.dataset, a.seed)
    xr, xd = load_nodes(a.dataset); bank = make_bank(xr, xd, 'kernel', 10)
    V = {'base': ('', ''), 'final': ('pmid,costudy,ctx,rec', 'fam,seed,hmdd,xdeg'), 'final_shuf': ('pmid,costudy,ctx,rec', 'fam,seed,hmdd,xdeg'),
         'prov': ('pmid,costudy,ctx,rec', ''), 'mprior': ('', 'fam,seed,hmdd,xdeg')}[a.variant]
    shuf = a.variant.endswith('_shuf')
    pv = Prov(a.dataset, shuf, np.random.RandomState(a.seed), groups=V[0].split(',')) if V[0] else None
    mp = MPrior(a.dataset, rng=np.random.RandomState(a.seed), shuffle=set(V[1].split(',')) if shuf else None) if V[1] else None
    res = []
    for f in range(5):
        tr = np.concatenate([pos[pf != f], negs[nf != f]]); ytr = np.concatenate([np.ones((pf != f).sum()), np.zeros((nf != f).sum())])
        te = np.concatenate([pos[pf == f], negs[nf == f]]); yte = np.concatenate([np.ones((pf == f).sum()), np.zeros((nf == f).sum())])
        Xtr, Xte = build_xy_k(bank, tr, ytr, te, {'ctx', 'diff'}, a.seed * 100 + f, a.inner)
        if pv is not None:
            Ptr, Pte = crossfit(pv, tr, ytr, te, a.seed * 100 + f, lambda o, s, q: o.transform(s, q), K=a.inner)
            Xtr = np.concatenate([Xtr, Ptr], 1); Xte = np.concatenate([Xte, Pte], 1)
        if mp is not None:
            Etr, Ete, _ = build_extra(mp, tr, ytr, te, set(V[1].split(',')), a.seed * 100 + f, K=a.inner)
            Xtr = np.concatenate([Xtr, Etr], 1); Xte = np.concatenate([Xte, Ete], 1)
        m = LogisticRegression(max_iter=5000, C=0.3).fit(Xtr, ytr.astype(int)); s = m.predict_proba(Xte)[:, 1]
        res.append((roc_auc_score(yte, s), average_precision_score(yte, s)))
    res = np.array(res)
    line = f'## STUDY-GROUPED {a.dataset} s{a.seed} K{a.inner} {a.variant} ncol={Xtr.shape[1]}: AUC {res[:,0].mean():.4f}±{res[:,0].std():.4f} AUPR {res[:,1].mean():.4f}±{res[:,1].std():.4f}'
    print(line, flush=True); open(os.path.join(HERE, 'results', 'final', 'study_split.txt'), 'a').write(line + '\n')

if __name__ == '__main__':
    main()
