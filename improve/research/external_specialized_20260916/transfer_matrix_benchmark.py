"""Transferable-core evaluation on external matrix-only lncRNA/ncRNA-drug datasets.

Our model form = iNcRD-HG graph-derived pair features (leak-free SVD + degree of
the training adjacency) scored by a tabular learner (HistGB). Supports:

  --mode fixed : explicit train/test positive-pair files (e.g. HGNNLDA RNAactDrug)
  --mode kfold : random k-fold over positives with balanced test negatives
                 (e.g. LRGCPND NoncoRNA+ncDR benchmark)

Both report AUC and AUPR. Balanced (1:1) test negatives are used so the numbers
line up with the balanced protocols those papers use; a global-ranking variant is
also printed for reference.
"""
import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def load_pairs(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", "\t").split()
            out.append((int(parts[0]), int(parts[1])))
    return np.array(out, dtype=int)


def feats(A, k=64):
    A = np.asarray(A, dtype=np.float32)
    deg_r = A.sum(1); deg_c = A.sum(0)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(S))
    Er = np.hstack([(U[:, :k] * S[:k]), (deg_r / max(deg_r.max(), 1))[:, None]])
    Ec = np.hstack([(Vt[:k].T * S[:k]), (deg_c / max(deg_c.max(), 1))[:, None]])
    return Er.astype(np.float32), Ec.astype(np.float32)


def pairfeat(Er, Ec, dm, dd, idx):
    i, j = idx[:, 0], idx[:, 1]
    return np.hstack([Er[i], Ec[j], Er[i] * Ec[j],
                      (dm[i] / max(dm.max(), 1))[:, None],
                      (dd[j] / max(dd.max(), 1))[:, None]])


def evaluate(pos_tr, pos_te, nr, nd, seed, neg_ratio=5, dump=None):
    rng = np.random.default_rng(seed)
    Atr = np.zeros((nr, nd), dtype=np.float32)
    for i, j in pos_tr:
        Atr[i, j] = 1.0
    Er, Ec = feats(Atr)
    dm = Atr.sum(1); dd = Atr.sum(0)
    posset = set(map(tuple, np.vstack([pos_tr, pos_te])))
    neg = []
    while len(neg) < neg_ratio * len(pos_tr):
        i = int(rng.integers(0, nr)); j = int(rng.integers(0, nd))
        if (i, j) not in posset:
            neg.append((i, j))
    neg = np.array(neg)
    Xtr = pairfeat(Er, Ec, dm, dd, np.vstack([pos_tr, neg]))
    ytr = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(neg))])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=seed)
    clf.fit(Xtr, ytr)
    # balanced test negatives (1:1) matching those papers' balanced protocol
    bneg = []
    while len(bneg) < len(pos_te):
        i = int(rng.integers(0, nr)); j = int(rng.integers(0, nd))
        if (i, j) not in posset:
            bneg.append((i, j))
    bneg = np.array(bneg)
    Xte = pairfeat(Er, Ec, dm, dd, pos_te); Xbn = pairfeat(Er, Ec, dm, dd, bneg)
    ste = clf.predict_proba(Xte)[:, 1]; sbn = clf.predict_proba(Xbn)[:, 1]
    yb = np.concatenate([np.ones(len(pos_te)), np.zeros(len(bneg))])
    sb = np.concatenate([ste, sbn])
    if dump:
        np.savez_compressed(dump, X_train=Xtr, y_train=ytr,
                            X_test=np.vstack([Xte, Xbn]), y_test=yb)
    auc = roc_auc_score(yb, sb); aupr = average_precision_score(yb, sb)
    # global ranking (subsample negatives if the pair space is huge)
    allp = np.array([(i, j) for i in range(nr) for j in range(nd) if (i, j) not in posset])
    if len(allp) > 100000:
        sel = rng.choice(len(allp), 100000, replace=False)
        allp = allp[sel]
    Xall = pairfeat(Er, Ec, dm, dd, allp)
    sall = clf.predict_proba(Xall)[:, 1]
    yg = np.concatenate([np.ones(len(pos_te)), np.zeros(len(allp))])
    sg = np.concatenate([ste, sall])
    return auc, aupr, roc_auc_score(yg, sg), average_precision_score(yg, sg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "kfold"], required=True)
    ap.add_argument("--pos", help="all positive pairs file (kfold mode)")
    ap.add_argument("--train", help="train positive pairs file (fixed mode)")
    ap.add_argument("--test", help="test positive pairs file (fixed mode)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--nr", type=int, default=None)
    ap.add_argument("--nd", type=int, default=None)
    ap.add_argument("--dump-prefix", default=None, help="dir/prefix to save pair features for a TabICLv2 stage")
    a = ap.parse_args()

    if a.mode == "fixed":
        ptr, pte = load_pairs(a.train), load_pairs(a.test)
        nr = a.nr or int(max(ptr[:, 0].max(), pte[:, 0].max())) + 1
        nd = a.nd or int(max(ptr[:, 1].max(), pte[:, 1].max())) + 1
        dump = f"{a.dump_prefix}.npz" if a.dump_prefix else None
        auc, aupr, gauc, gaupr = evaluate(ptr, pte, nr, nd, a.seed, dump=dump)
        print(f"fixed split: n_rna={nr} n_drug={nd} train={len(ptr)} test={len(pte)}")
        print(f"balanced  AUC {auc:.4f}  AUPR {aupr:.4f}")
        print(f"global    AUC {gauc:.4f}  AUPR {gaupr:.4f}")
    else:
        P = load_pairs(a.pos)
        nr = a.nr or int(P[:, 0].max()) + 1
        nd = a.nd or int(P[:, 1].max()) + 1
        rng = np.random.default_rng(a.seed)
        perm = rng.permutation(len(P))
        folds = np.array_split(perm, a.folds)
        A, R, GA, GR = [], [], [], []
        for fi, te in enumerate(folds):
            tr = np.setdiff1d(perm, te)
            dump = f"{a.dump_prefix}_fold{fi}.npz" if a.dump_prefix else None
            auc, aupr, gauc, gaupr = evaluate(P[tr], P[te], nr, nd, a.seed + fi, dump=dump)
            A.append(auc); R.append(aupr); GA.append(gauc); GR.append(gaupr)
            print(f"fold {fi}: balanced AUC/AUPR {auc:.4f}/{aupr:.4f}  global {gauc:.4f}/{gaupr:.4f}", flush=True)
        print(f"\nkfold({a.folds}) n_rna={nr} n_drug={nd} pos={len(P)}")
        print("balanced  AUC %.4f+-%.4f  AUPR %.4f+-%.4f" % (np.mean(A), np.std(A), np.mean(R), np.std(R)))
        print("global    AUC %.4f+-%.4f  AUPR %.4f+-%.4f" % (np.mean(GA), np.std(GA), np.mean(GR), np.std(GR)))


if __name__ == "__main__":
    main()
