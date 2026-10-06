# -*- coding: utf-8 -*-
"""Quick signal probe for internal-only (PLMF-MDA sequences / names / SMILES / train matrix)
miRNA relations: which label-free relation R makes  sum_m' R[m,m'] A[m',d]  informative on top of a
degree + path3 base, especially for sparse miRNAs?  Cross-fitted (K=5 groups inside the training
positives, like LQO) LR, 5cv seed 42."""
import os, re, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
from plmf_splits import get_split, load_dataset  # noqa

def stem(n): return re.sub(r'-(3p|5p)$', '', n.lower().strip())
def family(n):
    s = stem(n).replace('hsa-', ''); s = re.sub(r'-[12]$', '', s); return re.sub(r'([0-9])[a-z]$', r'\1', s)

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def rel_from_labels(lab):
    lab = np.asarray(lab); R = (lab[:, None] == lab[None, :]).astype(float); np.fill_diagonal(R, 0); return R

def knn_rel(S, k):
    S = S.copy(); np.fill_diagonal(S, -1e9); idx = np.argsort(-S, 1)[:, :k]
    R = np.zeros_like(S); np.put_along_axis(R, idx, 1.0, 1); return np.maximum(R, R.T)

def build_rels(ds):
    mir, dru, pos, neg = load_dataset(ds)
    names = mir['miRNA_Name'].astype(str).tolist(); seqs = mir['miRNA_Sequence'].astype(str).str.upper().str.replace('T', 'U').tolist()
    g = np.load(os.path.join(BASE, 'mphgnn_plmf_20260921', 'cache', f'graph_{ds}.npz'), allow_pickle=True)
    xr = g['mirna_kmer']; xn = xr / np.linalg.norm(xr, axis=1, keepdims=True); Sk = xn @ xn.T
    n = len(names); E = np.zeros((n, n))
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', f'lev_{ds}.npy')
    if os.path.exists(cache):
        E = np.load(cache)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                E[i, j] = E[j, i] = 1 - lev(seqs[i], seqs[j]) / max(len(seqs[i]), len(seqs[j]))
        np.save(cache, E)
    rels = {
        'knn10_kmer': knn_rel(Sk, 10),
        'dense_kmer': np.where(np.eye(n) > 0, 0, np.maximum(Sk, 0)),
        'knn10_lev': knn_rel(E, 10),
        'seed7': rel_from_labels([s[1:8] for s in seqs]),
        'seed6': rel_from_labels([s[1:7] for s in seqs]),
        'precursor': rel_from_labels([stem(x) for x in names]),
        'namefam': rel_from_labels([family(x) for x in names]),
        'mirstem': rel_from_labels([re.sub(r'-(\d+)$', '', stem(x)) for x in names]),
    }
    for k, R in rels.items():
        print(f'  {ds} rel {k:12s} coverage {(R.sum(1) > 0).mean():.2f} mean deg {R.sum(1).mean():.1f}', flush=True)
    return rels, pos

def feats(A, R_dict, r, d):
    dr, dd = A.sum(1), A.sum(0); P3 = (A @ A.T) @ A
    f = [np.log1p(dr[r]), np.log1p(dd[d]), np.log1p(P3[r, d]), np.log1p(P3[r, d] / (dr[r] * dd[d] + 1))]
    for k, R in R_dict.items():
        c = (R @ A)[r, d]; rs = R.sum(1)[r]
        f += [np.log1p(c), c / (rs + 1), np.log1p(c / (dd[d] + 1))]
    return np.stack(f, 1)

def run(ds, rel_keys, rels, seed=42):
    out = []
    for fold in range(5):
        tr, ytr, te, yte = get_split(ds, '5cv', fold, seed)
        nr = rels['seed7'].shape[0]; nd = int(max(tr[:, 1].max(), te[:, 1].max())) + 1
        rng = np.random.RandomState(fold); grp = rng.randint(5, size=len(tr))
        Xtr = np.zeros((len(tr), 4 + 3 * len(rel_keys)))
        pos_tr = tr[ytr == 1]; gpos = grp[ytr == 1]
        for j in range(5):
            A = np.zeros((nr, nd)); msg = pos_tr[gpos != j]; A[msg[:, 0], msg[:, 1]] = 1
            s = grp == j; Xtr[s] = feats(A, {k: rels[k] for k in rel_keys}, tr[s, 0], tr[s, 1])
        A = np.zeros((nr, nd)); A[pos_tr[:, 0], pos_tr[:, 1]] = 1
        Xte = feats(A, {k: rels[k] for k in rel_keys}, te[:, 0], te[:, 1])
        sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
        p = m.predict_proba(sc.transform(Xte))[:, 1]
        degte = A.sum(1)[te[:, 0]]; sparse = degte <= 3
        # sparse-subset AUC: sparse-miRNA positives vs all negatives
        msk = (yte == 0) | sparse
        out.append((roc_auc_score(yte, p), average_precision_score(yte, p), roc_auc_score(yte[msk], p[msk])))
    return np.array(out).mean(0)

if __name__ == '__main__':
    for ds in ('MDR', 'MDS'):
        rels, pos = build_rels(ds)
        base = run(ds, [], rels)
        print(f'{ds} base(deg+p3)      AUC {base[0]:.4f} AUPR {base[1]:.4f} sparseAUC {base[2]:.4f}', flush=True)
        for k in rels:
            r = run(ds, [k], rels)
            print(f'{ds} +{k:15s} AUC {r[0]:.4f} ({r[0]-base[0]:+.4f}) AUPR {r[1]:.4f} ({r[1]-base[1]:+.4f}) sparseAUC {r[2]:.4f} ({r[2]-base[2]:+.4f})', flush=True)
        r = run(ds, ['knn10_kmer', 'seed7', 'precursor', 'namefam'], rels)
        print(f'{ds} +kmer+seed7+pre+fam  AUC {r[0]:.4f} ({r[0]-base[0]:+.4f}) AUPR {r[1]:.4f} ({r[1]-base[1]:+.4f}) sparseAUC {r[2]:.4f} ({r[2]-base[2]:+.4f})', flush=True)
        r = run(ds, ['knn10_kmer'], rels)
        r2 = run(ds, ['knn10_kmer', 'seed7', 'precursor', 'namefam', 'knn10_lev'], rels)
        print(f'{ds} kmer-only {r[0]:.4f}/{r[1]:.4f} -> all-internal {r2[0]:.4f}/{r2[1]:.4f} ({r2[0]-r[0]:+.4f}/{r2[1]-r[1]:+.4f})', flush=True)
