import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = "/tmp/opencode/comp/GCFMCL/D2R.inter"
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
    Er = (U[:, :k] * S[:k]).astype(np.float32)
    Ec = (Vt[:k].T * S[:k]).astype(np.float32)
    return Er, Ec, deg_r, deg_c


rng = np.random.default_rng(0)
perm = rng.permutation(len(P))
folds = np.array_split(perm, 5)
aucs, auprs = [], []
all_pairs = np.array([(i, j) for i in range(nm) for j in range(nd)])
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
    idx_tr = np.vstack([P[tra], neg])

    def pairfeat(idx):
        i, j = idx[:, 0], idx[:, 1]
        return np.hstack([Em[i], Ed[j], Em[i] * Ed[j],
                          (dm[i] / max(dm.max(), 1))[:, None],
                          (dd[j] / max(dd.max(), 1))[:, None]])
    Xtr = pairfeat(idx_tr)
    ytr = np.concatenate([np.ones(len(tra)), np.zeros(len(neg))])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=fi)
    clf.fit(Xtr, ytr)
    te = P[te_i]
    # balanced test negatives (1:1), matching GCFMCL's balanced 5-fold protocol
    bneg = []
    while len(bneg) < len(te):
        i = int(rng.integers(0, nm)); j = int(rng.integers(0, nd))
        if (i, j) not in pos:
            bneg.append((i, j))
    bneg = np.array(bneg)
    Xte = pairfeat(te); Xneg = pairfeat(bneg)
    s_te = clf.predict_proba(Xte)[:, 1]; s_neg = clf.predict_proba(Xneg)[:, 1]
    pred = np.concatenate([s_te, s_neg]); yy = np.concatenate([np.ones(len(te)), np.zeros(len(bneg))])
    aucs.append(roc_auc_score(yy, pred)); auprs.append(average_precision_score(yy, pred))
    print(f"fold {fi}: AUC {aucs[-1]:.4f} AUPR {auprs[-1]:.4f}", flush=True)
print("HistGB graph-features on GCFMCL D2R: AUC %.4f+-%.4f AUPR %.4f+-%.4f" %
      (np.mean(aucs), np.std(aucs), np.mean(auprs), np.std(auprs)))
