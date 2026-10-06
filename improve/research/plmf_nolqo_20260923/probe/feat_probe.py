# -*- coding: utf-8 -*-
"""Quick probe (no GNN): how much do structural pair features lose when the query edge is inside
the graph they are computed on, and how much of it comes back when every feature is defined over
simple paths (the query edge can never lie on a simple path of length >= 2 between its endpoints)?
LR on log1p/z-scored features, fitted on the training pairs of each PLMF-MDA 5cv fold (seed 42),
tested on the test fold.  Uses only the PLMF-MDA release (k-mer of sequences, ChemBERTa of SMILES,
names, training positives)."""
import sys, os, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import get_split, load_dataset

G = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921/cache/graph_{}.npz'


def rowminmax(x):
    mn = x.min(1, keepdims=True); mx = x.max(1, keepdims=True); return (x - mn) / (mx - mn + 1e-8)


def knn(x, k):
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12); s = xn @ xn.T; np.fill_diagonal(s, -1e9)
    idx = np.argsort(-s, 1)[:, :k]; R = np.zeros_like(s); np.put_along_axis(R, idx, 1.0, 1)
    return np.maximum(R, R.T)


def stem(n):
    return re.sub(r'-(3p|5p)$', '', n.lower().strip())


def same(lab):
    lab = np.asarray(lab); R = (lab[:, None] == lab[None, :]).astype(float); np.fill_diagonal(R, 0); return R


def feats(A, Sr, Sd, rels, r, d, exclusive):
    """pair features; exclusive=True removes every contribution of the query edge A[r,d]."""
    a = A[r, d] if exclusive else np.zeros(len(r))
    dr = A.sum(1)[r] - a; dd = A.sum(0)[d] - a
    Cm = A @ A.T; P3 = (Cm @ A)[r, d]
    if exclusive:
        # walks m-d'-m'-d that use the query edge: d'=d (deg(d) of them) or m'=m (deg(m)), minus the double count
        P3 = P3 - a * (A.sum(1)[r] + A.sum(0)[d] - 1)
    SrA = (Sr @ A)[r, d]; ASd = (A @ Sd)[r, d]; SrASd = (Sr @ A @ Sd)[r, d]   # zero-diagonal relations: already exclusive
    f = [dr, dd, P3, P3 / (dr * dd + 1), SrA, ASd, SrASd, SrA / (dd + 1), ASd / (dr + 1)]
    for R in rels:
        c = (R @ A)[r, d]; f += [c, c / (R.sum(1)[r] + 1)]
    return np.log1p(np.maximum(np.stack(f, 1), 0))


def main(ds):
    g = np.load(G.format(ds), allow_pickle=True)
    mir, dru, pos, neg = load_dataset(ds)
    names = mir['miRNA_Name'].astype(str).values; seqs = mir['miRNA_Sequence'].astype(str).str.upper().str.replace('T', 'U').tolist()
    xr = rowminmax(g['mirna_kmer'].astype(float)); xd = rowminmax(g['drug_x'].astype(float))
    Sr, Sd = knn(xr, 10), knn(xd, 10)
    rel = {'seed7': same([s[1:8] for s in seqs]), 'pre': same([stem(n) for n in names])}
    rows = []
    for f in range(5):
        tr, ytr, te, yte = get_split(ds, '5cv', f, 42)
        A = np.zeros((len(names), len(dru))); A[tr[ytr == 1, 0], tr[ytr == 1, 1]] = 1
        for name, rels in (('base', []), ('+seq', list(rel.values()))):
            for ex in (False, True):
                Xtr = feats(A, Sr, Sd, rels, tr[:, 0], tr[:, 1], ex); Xte = feats(A, Sr, Sd, rels, te[:, 0], te[:, 1], ex)
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                m = LogisticRegression(C=1.0, max_iter=3000).fit((Xtr - mu) / sd, ytr)
                s = m.predict_proba((Xte - mu) / sd)[:, 1]
                rows.append(dict(ds=ds, fold=f, feats=name, exclusive=ex, auc=roc_auc_score(yte, s), aupr=average_precision_score(yte, s)))
    d = pd.DataFrame(rows)
    print(d.groupby(['ds', 'feats', 'exclusive'])[['auc', 'aupr']].mean().round(4))


if __name__ == '__main__':
    for ds in ('MDR', 'MDS'):
        main(ds)
