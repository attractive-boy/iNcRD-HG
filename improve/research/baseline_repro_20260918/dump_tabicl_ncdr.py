"""Dump iNcRD-HG graph-derived pair features for the ncDR benchmark so the
frozen TabICLv2 stage (score_matrix_tabicl.py) can be applied.

Same split / negatives as baseline_repro common.py, so the resulting "our method
(TabICLv2)" number is directly comparable to the reproduced external baselines.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

OUTDIR = os.path.join(HERE, "tabicl_ncdr")
os.makedirs(OUTDIR, exist_ok=True)


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


def main():
    pairs, _, _ = common.load_ncdr()
    posset = common.ncdr_pos_set(pairs)
    for seed in common.SEEDS:
        msg, tpos = common.build_split(pairs, seed)
        trneg = common.sample_neg(posset, msg.shape[1], seed + 11)
        tneg = common.sample_neg(posset, tpos.shape[1], seed + 7)
        Atr = np.zeros((common.NM, common.ND), dtype=np.float32)
        Atr[msg[0], msg[1]] = 1.0
        Er, Ec = feats(Atr)
        dm = Atr.sum(1); dd = Atr.sum(0)
        ptr = np.vstack([msg.T, trneg.T]); ytr = np.r_[np.ones(msg.shape[1]), np.zeros(trneg.shape[1])]
        pte = np.vstack([tpos.T, tneg.T]); yte = np.r_[np.ones(tpos.shape[1]), np.zeros(tneg.shape[1])]
        Xtr = pairfeat(Er, Ec, dm, dd, ptr)
        Xte = pairfeat(Er, Ec, dm, dd, pte)
        np.savez_compressed(os.path.join(OUTDIR, f"ncdr_seed{seed}.npz"),
                            X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
        print(f"seed {seed}: train {Xtr.shape} test {Xte.shape} -> ncdr_seed{seed}.npz")
    print("done", OUTDIR)


if __name__ == "__main__":
    main()
