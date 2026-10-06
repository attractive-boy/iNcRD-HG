# -*- coding: utf-8 -*-
"""Reproduce DeepLDA (Gao & Shang, Front. Microbiol. 2023) with the official code.

Official repo: github.com/meihonggao/DeepLDA (Codes/ copied to ./official_codes).
Protocol (as coded in official utils.load_data):
  * read the lncRNA x drug association matrix (0/1),
  * shuffle the lncRNA (row) order,
  * build lncRNA / drug Gaussian-similarity node features from the *whole*
    association matrix, binarize them (>0.5) into GAT adjacencies,
  * single row split: first 9/10 rows = train, last 1/10 rows = val = test.

Variants:
  * deeplda : official GCN -> GAT model (Deep_LDA)
  * gcn     : GCN branch only (drop the GAT stack)
  * gat     : GAT stack only (drop the GCN feature extractor)

Usage:
  python repro_deeplda_official.py --dataset 1 --variant deeplda --seeds 72 666 0 1 2
"""
import argparse
import csv
import os
import statistics as st
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "official_codes"))

from utils import load_data, metrics  # noqa: E402
from layers import GATLayer, GCNLayer  # noqa: E402
from models import Deep_LDA  # noqa: E402

DEV = "cuda:0"


def build_args(dataset):
    args = argparse.Namespace()
    args.dataset_path = os.path.join(HERE, "Datasets", f"Dataset{dataset}") + "/"
    args.result_path = os.path.join(HERE, "Results", str(dataset)) + "/"
    args.no_cuda = False
    args.fastmode = False
    args.sparse = False
    args.seed = 72
    args.epochs = 1050
    args.lr = 1e-3
    args.weight_decay = 5e-4
    args.nhid = 50
    args.nclass = 200
    args.nheads = 8
    args.dropout = 0.0
    args.alpha = 0.2
    args.patience = 1000
    return args


class GCNOnly(nn.Module):
    """GCN branch only: A = sigmoid(gcn_lnc @ gcn_drug^T)."""

    def __init__(self, args):
        super().__init__()
        self.gcn = GCNLayer(args)

    def forward(self, dataset):
        l_feat, g_feat = self.gcn(dataset)
        return torch.sigmoid(torch.mm(l_feat, torch.t(g_feat)))


class GATOnly(nn.Module):
    """GAT stack only: raw similarity features -> GAT -> A = sigmoid(x @ y^T)."""

    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.l_in, self.g_in = args.l_f_nfeat, args.g_f_nfeat
        self.l_att = [GATLayer(self.l_in, args.nhid, dropout=args.dropout, alpha=args.alpha, concat=True) for _ in range(args.nheads)]
        for i, m in enumerate(self.l_att):
            self.add_module(f"l_att_{i}", m)
        self.l_out = GATLayer(args.nhid * args.nheads, args.nclass, dropout=args.dropout, alpha=args.alpha, concat=True)
        self.g_att = [GATLayer(self.g_in, args.nhid, dropout=args.dropout, alpha=args.alpha, concat=True) for _ in range(args.nheads)]
        for i, m in enumerate(self.g_att):
            self.add_module(f"g_att_{i}", m)
        self.g_out = GATLayer(args.nhid * args.nheads, args.nclass, dropout=args.dropout, alpha=args.alpha, concat=False)

    def forward(self, dataset):
        import torch.nn.functional as F
        x = F.dropout(dataset["Lnc_f_features"], self.dropout, training=self.training)
        x = torch.cat([att(x, dataset["Lnc_f_adj"]) for att in self.l_att], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.l_out(x, dataset["Lnc_f_adj"]))
        y = F.dropout(dataset["Gene_f_features"], self.dropout, training=self.training)
        y = torch.cat([att(y, dataset["Gene_f_adj"]) for att in self.g_att], dim=1)
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.elu(self.g_out(y, dataset["Gene_f_adj"]))
        return torch.sigmoid(torch.mm(x, torch.t(y)))


def make_model(variant, args):
    if variant == "deeplda":
        return Deep_LDA(args)
    if variant == "gcn":
        return GCNOnly(args)
    if variant == "gat":
        return GATOnly(args)
    raise ValueError(variant)


def select_pairs(lab, rows, balanced, gen):
    """Return (row, col) flat index tensors over `rows`.

    unbalanced: every entry of the selected rows (official behaviour).
    balanced:   all positives + an equal number of sampled negatives.
    """
    if not balanced:
        nd = lab.shape[1]
        r = rows.repeat_interleave(nd)
        c = torch.arange(nd, device=lab.device).repeat(rows.shape[0])
        return r, c
    sub = lab[rows]
    pos = (sub > 0.5).nonzero(as_tuple=False)
    neg = (sub <= 0.5).nonzero(as_tuple=False)
    n = pos.shape[0]
    perm = torch.randperm(neg.shape[0], generator=gen)[:n]
    idx = torch.cat([pos, neg[perm]], 0)
    return rows[idx[:, 0]], idx[:, 1]


def run_seed(dataset, variant, seed, epochs, balanced=False, verbose=False):
    args = build_args(dataset)
    args.seed = seed
    args.epochs = epochs
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_data(args)
    args.l_f_nfeat = data["Lnc_f_features"].shape[1]
    args.g_f_nfeat = data["Gene_f_features"].shape[1]
    model = make_model(variant, args).to(DEV)
    for k in ("Lnc_f_edge_index", "Lnc_f_adj", "Lnc_f_features",
              "Gene_f_edge_index", "Gene_f_adj", "Gene_f_features",
              "labels", "idx_train", "idx_val", "idx_test"):
        data[k] = data[k].to(DEV)
    gen = torch.Generator().manual_seed(seed + 1)
    lab = data["labels"]
    tr_r, tr_c = select_pairs(lab, data["idx_train"], balanced, gen)
    te_r, te_c = select_pairs(lab, data["idx_test"], balanced, gen)
    tr_y = lab[tr_r, tr_c]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.MSELoss(reduction="mean")
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(data)
        loss_train = loss(out[tr_r, tr_c], tr_y)
        loss_train.backward()
        opt.step()
        if verbose and (ep + 1) % 100 == 0:
            print(f"  ep{ep+1:04d} loss_train={loss_train.item():.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        out = model(data)
    auc, aupr, f1, mcc = metrics(out[te_r, te_c], lab[te_r, te_c])
    return {"dataset": dataset, "variant": variant, "balanced": int(balanced), "seed": seed,
            "auc": float(auc), "aupr": float(aupr), "f1": float(f1), "mcc": float(mcc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="1")
    ap.add_argument("--variant", default="deeplda", choices=["deeplda", "gcn", "gat"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[72, 666, 0, 1, 2])
    ap.add_argument("--epochs", type=int, default=1050)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results_deeplda_official.csv"))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        r = run_seed(a.dataset, a.variant, s, a.epochs, a.balanced, a.verbose)
        rows.append(r)
        tag = "bal" if a.balanced else "unbal"
        print(f"[{a.variant} ds{a.dataset} {tag}] seed {s}: AUC {r['auc']:.4f} AUPR {r['aupr']:.4f} F1 {r['f1']:.4f} MCC {r['mcc']:.4f}", flush=True)
    mode = "a" if os.path.exists(a.out) else "w"
    with open(a.out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant", "balanced", "seed", "auc", "aupr", "f1", "mcc"])
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    for k in ("auc", "aupr", "f1", "mcc"):
        v = [r[k] for r in rows]
        print(f"  {k.upper():5s} {st.mean(v):.4f} +- {st.pstdev(v):.4f}")


if __name__ == "__main__":
    main()
