"""Transfer the iNcRD-HG architecture onto DeepLDA's lncRNA-drug-resistance datasets.

DeepLDA (Gao & Shang, Front Microbiol 2023, doi:10.3389/fmicb.2023.1147778)
releases two binary lncRNA-drug association matrices:

  Dataset1 (NoncoRNA-derived)  3601 lncRNA x 71 drug, 3802 associations
  Dataset2 (ncDR-derived)       162 lncRNA x 31 drug,  184 associations

Its evaluation system:
  * shuffle lncRNA rows
  * Gaussian-similarity features from the full matrix
  * train on the first 90% of rows, test on the last 10% of rows (every entry
    of those rows is scored), reported over repeated 10-fold-style resampling
  * metrics: AUC, AUPR

We keep the data, the Gaussian-similarity inputs and the row-based split, and
swap the classifier for the accept-transferable core of iNcRD-HG: a
relation-aware heterogeneous GNN over the lncRNA/drug similarity graphs plus
the training associations, whose pair embeddings are scored (a) by the GNN
edge decoder itself and (b) by a tabular learner (HistGB / frozen TabICLv2).

Outputs per seed: pair embeddings (npz) + GNN-decoder AUC/AUPR.
"""
import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GeneralConv, to_hetero
from sklearn.metrics import roc_auc_score, average_precision_score


def gaussian_sim(M):
    n, m = M.shape
    sq_n = np.sum(M * M, axis=1)
    D = sq_n[:, None] + sq_n[None, :] - 2.0 * (M @ M.T)
    D = np.maximum(D, 0.0)
    gamma_n = sq_n.sum() / n
    L = np.exp(-gamma_n * D)

    Mt = M.T
    sq_m = np.sum(Mt * Mt, axis=1)
    D2 = sq_m[:, None] + sq_m[None, :] - 2.0 * (Mt @ Mt.T)
    D2 = np.maximum(D2, 0.0)
    gamma_m = sq_m.sum() / m
    Dg = np.exp(-gamma_m * D2)
    return L.astype(np.float32), Dg.astype(np.float32)


def knn_edges(S, k):
    n = S.shape[0]
    k = min(k, n - 1)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden, out):
        super().__init__()
        self.conv1 = GeneralConv((-1, -1), hidden)
        self.conv2 = GeneralConv((-1, -1), out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        return self.conv2(x, edge_index)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lin1 = Linear(2 * hidden, 2 * hidden)
        self.lin2 = Linear(2 * hidden, hidden)
        self.lin3 = Linear(hidden, 1)

    def forward(self, z, eli):
        r, c = eli
        h = torch.cat([z["lncRNA"][r], z["drug"][c]], dim=-1)
        h = F.leaky_relu(self.lin1(h), negative_slope=0.1)
        h = F.leaky_relu(self.lin2(h), negative_slope=0.1)
        return self.lin3(h).view(-1)


class Model(torch.nn.Module):
    def __init__(self, data, hidden, aggr):
        super().__init__()
        self.encoder = to_hetero(GNNEncoder(hidden, hidden), data.metadata(), aggr=aggr)
        self.decoder = EdgeDecoder(hidden)

    def forward(self, xd, ed, eli):
        return self.decoder(self.encoder(xd, ed), eli)


def pair_feats(z, eli):
    r, c = eli
    zl = z["lncRNA"][r]
    zd = z["drug"][c]
    return torch.cat([zl, zd, zl * zd, (zl - zd).abs()], dim=-1)


def build_seed(M, seed, knn, device):
    rng = np.random.default_rng(seed)
    n, m = M.shape
    perm = rng.permutation(n)
    M = M[perm]
    L, Dg = gaussian_sim(M)
    cut = int(round(0.9 * n))
    train_rows = np.arange(cut)
    test_rows = np.arange(cut, n)

    tri = np.argwhere(M[train_rows] > 0.5)
    tr_i, tr_j = tri[:, 0], tri[:, 1]
    pos_set = set(map(tuple, np.argwhere(M > 0.5)))
    n_neg = len(tr_i)
    neg = []
    while len(neg) < n_neg:
        i = int(rng.integers(0, cut))
        j = int(rng.integers(0, m))
        if (i, j) not in pos_set:
            neg.append((i, j))
    neg = np.array(neg)
    train_eli = torch.tensor(np.stack([np.concatenate([tr_i, neg[:, 0]]),
                                       np.concatenate([tr_j, neg[:, 1]])]), dtype=torch.long)
    train_y = torch.cat([torch.ones(len(tr_i)), torch.zeros(len(neg))]).float()

    te = np.argwhere(np.ones((len(test_rows), m)) > 0.5)
    te_i = test_rows[te[:, 0]]
    te_j = te[:, 1]
    test_eli = torch.tensor(np.stack([te_i, te_j]), dtype=torch.long)
    test_y = M[te_i, te_j].astype(np.float32)

    d = HeteroData()
    d["lncRNA"].x = torch.tensor(L)
    d["drug"].x = torch.tensor(Dg)
    assoc = torch.tensor(np.stack([tr_i, tr_j]), dtype=torch.long)
    d["lncRNA", "LD", "drug"].edge_index = assoc
    d["drug", "rev_LD", "lncRNA"].edge_index = assoc.flip(0)
    d["lncRNA", "LL", "lncRNA"].edge_index = knn_edges(L, knn)
    d["drug", "DD", "drug"].edge_index = knn_edges(Dg, knn)
    d["lncRNA", "LD", "drug"].edge_label_index = train_eli
    d["lncRNA", "LD", "drug"].edge_label = train_y
    return d, test_eli, test_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["1", "2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[666, 0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--data-root", default="/tmp/opencode/deeplncrna/Datasets")
    ap.add_argument("--out", default="/tmp/opencode/deeplncrna")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    path = os.path.join(a.data_root, f"Dataset{a.dataset}", "lnc_drug_net.csv")
    M = np.loadtxt(path, delimiter=",", dtype=np.float32)
    print(f"Dataset{a.dataset}: matrix {M.shape}, associations {int(M.sum())}", flush=True)

    results = []
    for s in a.seeds:
        set_seed(s)
        d, test_eli, test_y = build_seed(M, s, a.knn, device)
        d = d.to(device)
        model = Model(d, a.hidden, "mean").to(device)
        opt = torch.optim.RMSprop(model.parameters(), lr=a.lr, alpha=0.99)
        for ep in range(1, a.epochs + 1):
            model.train(); opt.zero_grad()
            p = model(d.x_dict, d.edge_index_dict, d["lncRNA", "LD", "drug"].edge_label_index)
            loss = F.binary_cross_entropy_with_logits(p, d["lncRNA", "LD", "drug"].edge_label)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(d.x_dict, d.edge_index_dict, test_eli).cpu().numpy()
            z = model.encoder(d.x_dict, d.edge_index_dict)
            X = pair_feats(z, d["lncRNA", "LD", "drug"].edge_label_index).cpu().numpy()
            Xte = pair_feats(z, test_eli).cpu().numpy()
            ytr = d["lncRNA", "LD", "drug"].edge_label.cpu().numpy()
        auc = roc_auc_score(test_y, p); aupr = average_precision_score(test_y, p)
        results.append((s, auc, aupr))
        print(f"seed {s}: GNN decoder AUC {auc:.4f} AUPR {aupr:.4f}", flush=True)
        np.savez_compressed(os.path.join(a.out, f"deeplncrna{ a.dataset }_seed{s}.npz"),
                            X_train=X.astype(np.float32), y_train=ytr,
                            X_test=Xte.astype(np.float32), y_test=test_y)

    arr = np.array([(r[1], r[2]) for r in results])
    print(f"\n===== iNcRD-HG GNN transferred to DeepLDA Dataset{a.dataset} =====")
    print("GNN decoder  AUC %.4f±%.4f  AUPR %.4f±%.4f" %
          (arr[:, 0].mean(), arr[:, 0].std(), arr[:, 1].mean(), arr[:, 1].std()))
    json.dump({"dataset": a.dataset, "seeds": a.seeds,
               "per_seed": results,
               "gnn_mean_auc": float(arr[:, 0].mean()), "gnn_std_auc": float(arr[:, 0].std()),
               "gnn_mean_aupr": float(arr[:, 1].mean()), "gnn_std_aupr": float(arr[:, 1].std())},
              open(os.path.join(a.out, f"deeplncrna{ a.dataset }_gnn.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
