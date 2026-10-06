"""SubKNet-lite: per-link enclosing-subgraph signature on the MPHGNN benchmark.

Follows SubKNet's subgraph extraction idea (1-hop enclosing subgraph on the
bipartite association graph augmented with same-type similarity blocks, with
direction/hop node labels), but uses fixed graph-kernel features + GBDT rather
than the full trainable RW_layer. Leak-free: only benchmark edges enter the graph.
"""
import numpy as np
import torch
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

PTH = "/tmp/opencode/mphgnn_repo/MiDrug_data_end.pth"
EDGE = ("miRNA", "MiDrug", "drug")

D = torch.load(PTH, weights_only=False).cpu()
NM, ND = D['miRNA'].num_nodes, D['drug'].num_nodes
bench = D[EDGE].edge_index.cpu().numpy()
test_pairs = D[EDGE].edge_label_index.cpu().numpy()
test_y = D[EDGE].edge_label.numpy()
mimi = D['miRNA', 'MiMi', 'miRNA'].edge_index.cpu().numpy()
mimi = np.concatenate([mimi, mimi[::-1]], 1)

A = sparse.csr_matrix((np.ones(bench.shape[1]), (bench[0], bench[1])), shape=(NM, ND))
S = sparse.csr_matrix((np.ones(mimi.shape[1]), (mimi[0], mimi[1])), shape=(NM, NM))
S.setdiag(0); S.eliminate_zeros()


def feats(pairs):
    m, d = pairs[0], pairs[1]
    out = np.zeros((len(m), 12), dtype=np.float32)
    for i in range(len(m)):
        mi, di = int(m[i]), int(d[i])
        nm_pure = S[mi].indices                     # miRNA neighbours through similarity
        nm_mixed = A[:, di].indices                 # miRNAs associated with d (benchmark)
        nd_pure = A[mi].indices                     # drugs associated with m
        s_mi = set(nm_pure.tolist()); s_mx = set(nm_mixed.tolist()); s_d = set(nd_pure.tolist())
        shared = s_mi & s_mx
        # edges within the miRNA side (similarity) and cross association edges
        sub = S[np.ix_(list(s_mi | s_mx), list(s_mi | s_mx))] if (s_mi | s_mx) else None
        internal = int(sub.nnz) // 2 if sub is not None else 0
        cross = int(A[np.ix_(list(nd_pure), list(s_mx))].nnz) if (nd_pure.size and s_mx) else 0
        out[i] = [len(nm_pure), len(nm_mixed), len(nd_pure), len(shared), internal, cross,
                  A[mi].nnz, A[:, di].nnz, S[mi].nnz,
                  len(s_mi | s_mx), len(s_mi & s_mx), int(di in s_d)]
    return out


rng = np.random.default_rng(666)
pos = set(map(tuple, bench.T.tolist()))
negs = []
while len(negs) < bench.shape[1]:
    i, j = int(rng.integers(0, NM)), int(rng.integers(0, ND))
    if (i, j) not in pos:
        negs.append((i, j))
bp = np.concatenate([bench.T, np.asarray(negs)], 0)
by = np.concatenate([np.ones(bench.shape[1]), np.zeros(len(negs))])
Xtr, Xte = feats(bp.T), feats(test_pairs)

aucs, aprs = [], []
for seed in (666, 0, 1):
    order = np.random.default_rng(seed).permutation(len(by))
    X, y = Xtr[order], by[order]
    folds = np.array_split(np.arange(len(y)), 5)
    pred = np.zeros(len(test_y))
    for f, va in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(y)), va)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed + f)
        clf.fit(X[tr], y[tr])
        pred += clf.predict_proba(Xte)[:, 1] / len(folds)
    aucs.append(roc_auc_score(test_y, pred)); aprs.append(average_precision_score(test_y, pred))
    print(f"seed {seed}: test AUC/AUPR {aucs[-1]:.4f}/{aprs[-1]:.4f}", flush=True)
print(f"\n===== SubKNet-lite (enclosing-subgraph signature + HistGB, 5-fold) =====")
print("AUC %.4f  AUPR %.4f" % (np.mean(aucs), np.mean(aprs)))
print("SubKNet reported  AUC 0.9717  AUPR 0.9341")
print("MPHGNN reported   AUC 0.9780  AUPR 0.9370")
print("prior weak proxy  AUC 0.9514  AUPR 0.8922")
