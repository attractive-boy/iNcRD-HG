"""Graph-feature + TabICLv2 dumps for the DeepLDA lncRNA-drug datasets.

Replicates DeepLDA's row-based protocol (transfer_deeplncrna.build_seed) but
replaces the GNN pair embeddings with compact Gaussian-similarity pair features
so the frozen TabICLv2 stage can score them, matching the "本文方法" recipe used
for the other external datasets.
"""
import os

import numpy as np

DATA_ROOT = "/tmp/opencode/deeplncrna/Datasets"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabicl_deeplncrna")
os.makedirs(OUT, exist_ok=True)
SEEDS = [666, 0, 1, 2, 3]


def gaussian_sim(M):
    n, m = M.shape
    sq_n = np.sum(M * M, axis=1)
    D = sq_n[:, None] + sq_n[None, :] - 2.0 * (M @ M.T)
    D = np.maximum(D, 0.0)
    L = np.exp(-(sq_n.sum() / n) * D)
    Mt = M.T
    sq_m = np.sum(Mt * Mt, axis=1)
    D2 = sq_m[:, None] + sq_m[None, :] - 2.0 * (Mt @ Mt.T)
    D2 = np.maximum(D2, 0.0)
    Dg = np.exp(-(sq_m.sum() / m) * D2)
    return L.astype(np.float32), Dg.astype(np.float32)


def lowrank(S, k):
    U, s, _ = np.linalg.svd(S, full_matrices=False)
    k = min(k, len(s))
    Z = U[:, :k] * s[:k]
    return Z.astype(np.float32), (s[:k] / max(s[0], 1e-9)).astype(np.float32)


def main():
    for ds in ("1", "2"):
        M = np.loadtxt(os.path.join(DATA_ROOT, f"Dataset{ds}", "lnc_drug_net.csv"),
                       delimiter=",", dtype=np.float32)
        n, m = M.shape
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            M2 = M[perm]
            L, Dg = gaussian_sim(M2)
            cut = int(round(0.9 * n))
            train_rows = np.arange(cut)
            test_rows = np.arange(cut, n)
            tri = np.argwhere(M2[train_rows] > 0.5)
            pos_set = set(map(tuple, np.argwhere(M2 > 0.5)))
            neg = []
            while len(neg) < len(tri):
                i = int(rng.integers(0, cut)); j = int(rng.integers(0, m))
                if (i, j) not in pos_set:
                    neg.append((i, j))
            neg = np.array(neg)
            Lr, _ = lowrank(L, min(32, n, m))
            Dr, _ = lowrank(Dg, min(32, n, m))

            def pf(i, j):
                return np.hstack([Lr[i], Dr[j], Lr[i] * Dr[j]])

            tr = np.vstack([tri, neg])
            Xtr = np.vstack([pf(tr[:, 0], tr[:, 1])])
            ytr = np.concatenate([np.ones(len(tri)), np.zeros(len(neg))])
            te = np.argwhere(np.ones((len(test_rows), m)) > 0.5)
            ti = test_rows[te[:, 0]]; tj = te[:, 1]
            Xte = pf(ti, tj)
            yte = M2[ti, tj].astype(np.int64)
            np.savez_compressed(os.path.join(OUT, f"deeplncrna{ds}_seed{seed}.npz"),
                                X_train=Xtr.astype(np.float32), y_train=ytr,
                                X_test=Xte.astype(np.float32), y_test=yte)
            print(f"ds{ds} seed{seed}: train {Xtr.shape} test {Xte.shape} pos {int(yte.sum())}",
                  flush=True)
    print("done", OUT)


if __name__ == "__main__":
    main()
