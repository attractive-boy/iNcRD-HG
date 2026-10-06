"""Dump graph-derived pair features for GCFMCL D2R so TabICLv2 can score them.

Mirrors /tmp/opencode/gcfmcl_feat.py (balanced 5-fold) but writes npz dumps.
"""
import os

import numpy as np

DATA = "/tmp/opencode/comp/GCFMCL/D2R.inter"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabicl_gcfmcl")
os.makedirs(OUT, exist_ok=True)

pairs = []
with open(DATA) as f:
    next(f)
    for line in f:
        line = line.rstrip("\n").rstrip("\t")
        if not line:
            continue
        d, m = line.split("\t")[:2]
        pairs.append((d, m))
drugs = sorted({d for d, _ in pairs}); mirs = sorted({m for _, m in pairs})
d2i = {x: i for i, x in enumerate(drugs)}; m2i = {x: i for i, x in enumerate(mirs)}
nd, nm = len(drugs), len(mirs)
P = np.array([(m2i[m], d2i[d]) for d, m in pairs])
pos = set(map(tuple, P))
print("pairs", len(P), "drugs", nd, "mirs", nm)


def svd_feats(A, k=64):
    deg_r = A.sum(1); deg_c = A.sum(0)
    U, S, Vt = np.linalg.svd(A.astype(np.float32), full_matrices=False)
    k = min(k, len(S))
    Em = (U[:, :k] * S[:k]).astype(np.float32)
    Ed = (Vt[:k].T * S[:k]).astype(np.float32)
    return Em, Ed, deg_r, deg_c


rng = np.random.default_rng(0)
perm = rng.permutation(len(P))
folds = np.array_split(perm, 5)
for fi, te_i in enumerate(folds):
    tra = np.setdiff1d(perm, te_i)
    Atr = np.zeros((nm, nd), dtype=np.float32)
    for i in tra:
        Atr[P[i, 0], P[i, 1]] = 1.0
    Em, Ed, dm, dd = svd_feats(Atr)
    neg = []
    while len(neg) < 5 * len(tra):
        i = int(rng.integers(0, nm)); j = int(rng.integers(0, nd))
        if (i, j) not in pos:
            neg.append((i, j))
    neg = np.array(neg)

    def pairfeat(idx):
        i, j = idx[:, 0], idx[:, 1]
        return np.hstack([Em[i], Ed[j], Em[i] * Ed[j],
                          (dm[i] / max(dm.max(), 1))[:, None],
                          (dd[j] / max(dd.max(), 1))[:, None]])

    Xtr = pairfeat(np.vstack([P[tra], neg]))
    ytr = np.concatenate([np.ones(len(tra)), np.zeros(len(neg))])
    te = P[te_i]
    bneg = []
    while len(bneg) < len(te):
        i = int(rng.integers(0, nm)); j = int(rng.integers(0, nd))
        if (i, j) not in pos:
            bneg.append((i, j))
    bneg = np.array(bneg)
    Xte = np.vstack([pairfeat(te), pairfeat(bneg)])
    yte = np.concatenate([np.ones(len(te)), np.zeros(len(bneg))])
    np.savez_compressed(os.path.join(OUT, f"gcfmcl_fold{fi}.npz"),
                        X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
    print(f"fold {fi}: train {Xtr.shape} test {Xte.shape}")
print("done", OUT)
