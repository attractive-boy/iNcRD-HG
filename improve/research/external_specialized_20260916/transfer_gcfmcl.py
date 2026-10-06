"""Transfer the iNcRD-HG architecture onto GCFMCL's D2R dataset.

GCFMCL (Wei et al., Briefings in Bioinformatics 2023, doi:10.1093/bib/bbad247)
releases D2R: 2049 miRNA-drug sensitivity associations over 140 drugs x 431
miRNAs. Its evaluation system is a random interaction split (5-fold) with
global AUC / AUPR ranking.

We keep the data and the random 5-fold interaction split. GCFMCL provides no
node features, so we build leak-free node representations from the *training*
adjacency only (truncated SVD + degree) and put the iNcRD-HG heterogeneous GNN
on top. Positives = held-out interactions; negatives = every other pair.
"""
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GeneralConv, to_hetero
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = "/tmp/opencode/comp/GCFMCL/D2R.inter"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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
        h = torch.cat([z["miRNA"][r], z["drug"][c]], dim=-1)
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


def feats(A, dim=64):
    A = A.astype(np.float32)
    deg_r = A.sum(1); deg_c = A.sum(0)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(dim, len(S))
    Er = (U[:, :k] * S[:k]).astype(np.float32)
    Ec = (Vt[:k].T * S[:k]).astype(np.float32)
    Er = np.hstack([Er, deg_r[:, None] / max(deg_r.max(), 1)])
    Ec = np.hstack([Ec, deg_c[:, None] / max(deg_c.max(), 1)])
    return Er, Ec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

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
    P = np.array([(m2i[m], d2i[d]) for d, m in pairs])   # (miRNA, drug)
    print(f"GCFMCL D2R: {len(P)} interactions, {nd} drugs x {nm} miRNAs", flush=True)

    pos_all = set(map(tuple, P[:, ::-1]))                # (drug, miRNA)
    all_pairs = np.array([(i, j) for i in range(nd) for j in range(nm)])
    is_pos = np.zeros((nd, nm), dtype=bool)
    for d, m in pos_all:
        is_pos[d, m] = True

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(P))
    folds = np.array_split(perm, a.folds)
    aucs, auprs = [], []
    for fi, test_idx in enumerate(folds):
        tra = np.setdiff1d(perm, test_idx)
        Atr = np.zeros((nm, nd), dtype=np.float32)
        for i in tra:
            Atr[P[i, 0], P[i, 1]] = 1.0
        Erow, Ecol = feats(Atr.T)                        # rows = drugs (nd), cols = miRNAs (nm)
        set_seed(a.seed + fi)
        d = HeteroData()
        d["drug"].x = torch.tensor(Erow)
        d["miRNA"].x = torch.tensor(Ecol)
        tr = torch.tensor(P[tra].T, dtype=torch.long)
        d["miRNA", "MD", "drug"].edge_index = tr
        d["drug", "rev_MD", "miRNA"].edge_index = tr.flip(0)
        n_neg = a.neg_ratio * len(tra)
        neg = []
        while len(neg) < n_neg:
            i = int(rng.integers(0, nm)); j = int(rng.integers(0, nd))
            if not is_pos[j, i]:
                neg.append((i, j))
        neg = np.array(neg)
        eli = torch.tensor(np.stack([np.concatenate([P[tra, 0], neg[:, 0]]),
                                     np.concatenate([P[tra, 1], neg[:, 1]])]), dtype=torch.long)
        y = torch.cat([torch.ones(len(tra)), torch.zeros(len(neg))]).float()
        d = d.to(dev)
        eli = eli.to(dev); y = y.to(dev)
        d["miRNA", "MD", "drug"].edge_label_index = eli
        d["miRNA", "MD", "drug"].edge_label = y
        model = Model(d, a.hidden, "mean").to(dev)
        opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99)
        for ep in range(1, a.epochs + 1):
            model.train(); opt.zero_grad()
            p = model(d.x_dict, d.edge_index_dict, eli)
            loss = F.binary_cross_entropy_with_logits(p, y)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            te = P[test_idx]
            eli_te = torch.tensor(te.T, dtype=torch.long)
            p = model(d.x_dict, d.edge_index_dict, eli_te).cpu().numpy()
        neg_mask = np.array([(i, j) not in pos_all for i, j in all_pairs])
        neg_pairs = all_pairs[neg_mask]
        pred = np.concatenate([p, np.zeros(len(neg_pairs))])
        yy = np.concatenate([np.ones(len(p)), np.zeros(len(neg_pairs))])
        auc = roc_auc_score(yy, pred); aupr = average_precision_score(yy, pred)
        aucs.append(auc); auprs.append(aupr)
        print(f"fold {fi}: AUC {auc:.4f} AUPR {aupr:.4f}", flush=True)

    print("\n===== iNcRD-HG GNN on GCFMCL D2R (5-fold, global ranking) =====")
    print("AUC %.4f±%.4f  AUPR %.4f±%.4f" % (np.mean(aucs), np.std(aucs), np.mean(auprs), np.std(auprs)))
    import json
    json.dump({"folds": a.folds, "aucs": aucs, "auprs": auprs,
               "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
               "mean_aupr": float(np.mean(auprs)), "std_aupr": float(np.std(auprs))},
              open("/tmp/opencode/gcfmcl_transfer.json", "w"), indent=2)


if __name__ == "__main__":
    main()
