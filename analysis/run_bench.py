"""
Run iNcRD-HG and the Table 3 GNN baselines across every NcRD-Bench split.

Two protocol changes relative to `predict_association.py`, both needed before the
numbers mean anything:

  * the epoch is chosen on the validation fold, not the test fold.  The released
    loop scores test every epoch and the paper's headline matches the best epoch
    seen, which is model selection on the test set.
  * results are averaged over seeds and reported with per-drug and full-ranking
    metrics alongside global AUC/AUPR.

Run from the directory holding ./9.bench:

    python analysis/run_bench.py --seeds 666 0 1
"""

import argparse
import collections
import csv
import os
import statistics as st
import sys

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

from bench_metrics import pairwise_metrics, ranking_metrics

TASKS = [("lncRNA", "LncDrug", "LncLnc"), ("miRNA", "MiDrug", "MiMi")]
SPLITS = [
    "random", "matched_drug", "matched_rna", "matched_both",
    "cold_rna", "cold_drug", "cold_pair", "matched",
]
DEFAULT_SPLITS = [
    "random", "matched_drug", "matched_rna", "matched_both",
    "cold_rna", "cold_drug", "cold_pair",
]
DEGREE_CONTROLS = ["degree", "rna_degree", "degree_product"]

# Encoders exactly as compare_base_GNN.py defines them; ASAP is LEConv there.
ENCODERS = {
    "GAT": lambda i, h, o: (pyg_nn.GATConv(i, h, heads=2), pyg_nn.GATConv(2 * h, o, heads=2)),
    "GraphSAGE": lambda i, h, o: (pyg_nn.SAGEConv(i, h), pyg_nn.SAGEConv(h, o)),
    "ASAP": lambda i, h, o: (pyg_nn.LEConv(i, h), pyg_nn.LEConv(h, o)),
    "GeneralGNN": lambda i, h, o: (pyg_nn.GeneralConv(i, h), pyg_nn.GeneralConv(h, o)),
    "GraphConv": lambda i, h, o: (pyg_nn.GraphConv(i, h), pyg_nn.GraphConv(h, o)),
    "GCN": lambda i, h, o: (pyg_nn.GCNConv(i, h), pyg_nn.GCNConv(h, o)),
}

HETERO_HP = {  # the paper's tuned settings
    "lncRNA": dict(lr=5e-4, epochs=101, channels=64, aggr="min"),
    "miRNA": dict(lr=1e-4, epochs=301, channels=256, aggr="mean"),
}


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_split(bench, split, edge_type, device):
    return [
        torch.load(f"{bench}/{split}/{edge_type}_{name}_data.pth", map_location=device)
        for name in ("train", "val", "test")
    ]


# --------------------------------------------------------------------------
# iNcRD-HG
# --------------------------------------------------------------------------

def run_hetero(bench, split, rna_type, edge_type, seed, device):
    import predict_association as pa
    pa.RNA_type, pa.edge_type = rna_type, edge_type

    seed_everything(seed)
    train_data, val_data, test_data = load_split(bench, split, edge_type, device)
    hp = HETERO_HP[rna_type]

    model = pa.Model(train_data, hp["channels"], hp["aggr"]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=hp["lr"], alpha=0.99)
    key = (rna_type, edge_type, "drug")

    def forward(data, index):
        out = model(data.x_dict, data.edge_index_dict, index)
        return out[0] if isinstance(out, tuple) else out

    best_state, best_auc = None, -1.0
    for _ in range(1, hp["epochs"]):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            forward(train_data, train_data[key].edge_label_index), train_data[key].edge_label
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = forward(val_data, val_data[key].edge_label_index).cpu().numpy()
        auc = _safe_auc(val_data[key].edge_label.cpu().numpy(), scores)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    def score_fn(pairs):
        with torch.no_grad():
            return forward(test_data, torch.tensor(pairs, dtype=torch.long, device=device)).cpu().numpy()

    return score_fn, test_data


# --------------------------------------------------------------------------
# flat GNN baselines
# --------------------------------------------------------------------------

class FlatGNN(nn.Module):
    def __init__(self, name, in_c, hid_c, out_c, num_rna):
        super().__init__()
        self.gcn1, self.gcn2 = ENCODERS[name](in_c, hid_c, out_c)
        self.num_rna = num_rna

    def forward(self, x, edge_index):
        x = torch.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x[:self.num_rna], x[self.num_rna:]


def _flatten(data, rna_type, edge_type, sim_type, device):
    """Collapse the heterogeneous graph the way compare_base_GNN.py does."""
    num_rna = data[rna_type].x.shape[0]
    rna_f, drug_f = data[rna_type].x, data["drug"].x

    x = torch.zeros((num_rna + data["drug"].x.shape[0], rna_f.shape[1] + drug_f.shape[1]))
    x[:num_rna, :rna_f.shape[1]] = rna_f.cpu()
    x[num_rna:, rna_f.shape[1]:] = drug_f.cpu()

    rna_sim = data[rna_type, sim_type, rna_type].edge_index
    drug_sim = data["drug", "DrugDrug", "drug"].edge_index.clone() + num_rna
    assoc = data[rna_type, edge_type, "drug"].edge_index.clone()
    assoc[1] += num_rna

    edges = torch.cat([assoc, assoc.flip(0), rna_sim, rna_sim.flip(0),
                       drug_sim, drug_sim.flip(0)], dim=1)
    return Data(x=x.to(device), edge_index=edges.to(device)), num_rna


def run_flat(bench, split, rna_type, edge_type, sim_type, name, seed, device,
             lr=5e-4, epochs=200, hidden=128, out=128):
    seed_everything(seed)
    train_data, val_data, test_data = load_split(bench, split, edge_type, device)
    key = (rna_type, edge_type, "drug")

    graphs = {n: _flatten(d, rna_type, edge_type, sim_type, device)[0]
              for n, d in (("train", train_data), ("val", val_data), ("test", test_data))}
    num_rna = train_data[rna_type].x.shape[0]

    model = FlatGNN(name, graphs["train"].x.shape[1], hidden, out, num_rna).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)

    def score(graph, index):
        rna_emb, drug_emb = model(graph.x, graph.edge_index)
        return (rna_emb[index[0]] * drug_emb[index[1]]).sum(-1)

    best_state, best_auc = None, -1.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = nn.BCEWithLogitsLoss()(
            score(graphs["train"], train_data[key].edge_label_index), train_data[key].edge_label
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            s = score(graphs["val"], val_data[key].edge_label_index).cpu().numpy()
        auc = _safe_auc(val_data[key].edge_label.cpu().numpy(), s)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    def score_fn(pairs):
        with torch.no_grad():
            return score(graphs["test"], torch.tensor(pairs, dtype=torch.long, device=device)).cpu().numpy()

    return score_fn, test_data


# --------------------------------------------------------------------------
# degree control
# --------------------------------------------------------------------------

def run_degree(bench, split, rna_type, edge_type, device, method):
    """Endpoint-popularity controls: no biology, no ncRNA--drug features."""
    train_data, _, test_data = load_split(bench, split, edge_type, device)
    key = (rna_type, edge_type, "drug")
    rna_degree = collections.Counter(train_data[key].edge_index[0].tolist())
    drug_degree = collections.Counter(train_data[key].edge_index[1].tolist())

    if method == "degree":
        def score(pairs):
            return np.array([drug_degree[int(d)] for d in pairs[1]], dtype=float)
    elif method == "rna_degree":
        def score(pairs):
            return np.array([rna_degree[int(r)] for r in pairs[0]], dtype=float)
    elif method == "degree_product":
        def score(pairs):
            return np.array([
                np.log1p(rna_degree[int(r)]) * np.log1p(drug_degree[int(d)])
                for r, d in pairs.T.tolist()
            ], dtype=float)
    else:
        raise ValueError(f"unknown degree control: {method}")
    return score, test_data


def _safe_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(score_fn, test_data, meta, rna_type, edge_type):
    key = (rna_type, edge_type, "drug")
    index = test_data[key].edge_label_index.cpu().numpy()
    labels = test_data[key].edge_label.cpu().numpy()

    result = pairwise_metrics(labels, score_fn(index), index[1])

    test_pos = np.array(meta["test_pos"])
    folds = meta.get("rna_folds")
    candidates = folds[2] if folds and meta["kind"] in ("cold_rna", "cold_pair") else range(meta["num_rna"])
    exclude = set(map(tuple, np.array(meta["train_pos"]).T.tolist()))
    exclude |= set(map(tuple, np.array(meta["val_pos"]).T.tolist()))
    result.update(ranking_metrics(score_fn, test_pos, candidates, exclude))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--bench", default="./9.bench")
    parser.add_argument("--out", default="./9.bench/results.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[666, 0, 1])
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, choices=SPLITS)
    parser.add_argument("--methods", nargs="+",
                        default=DEGREE_CONTROLS + ["iNcRD-HG"] + list(ENCODERS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.root))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    device = torch.device(args.device)
    rows = []

    for split in args.splits:
        for rna_type, edge_type, sim_type in TASKS:
            meta = torch.load(f"{args.bench}/{split}/{edge_type}_meta.pth")
            print(f"\n=== {split} / {edge_type} ===")
            for method in args.methods:
                per_seed = []
                seeds = [0] if method in DEGREE_CONTROLS else args.seeds
                for seed in seeds:
                    if method in DEGREE_CONTROLS:
                        fn, td = run_degree(
                            args.bench, split, rna_type, edge_type, device, method
                        )
                    elif method == "iNcRD-HG":
                        fn, td = run_hetero(args.bench, split, rna_type, edge_type, seed, device)
                    else:
                        fn, td = run_flat(args.bench, split, rna_type, edge_type, sim_type,
                                          method, seed, device)
                    per_seed.append(evaluate(fn, td, meta, rna_type, edge_type))

                agg = {}
                for k in per_seed[0]:
                    vals = [r[k] for r in per_seed if not np.isnan(r[k])]
                    agg[k] = st.mean(vals) if vals else float("nan")
                    agg[k + "_std"] = st.stdev(vals) if len(vals) > 1 else 0.0
                rows.append(dict(split=split, task=edge_type, method=method, **agg))
                print(f"  {method:12s} AUC={agg['auc']:.4f}  AUPR={agg['aupr']:.4f}  "
                      f"perDrugAUC={agg['auc_per_drug']:.4f}  rankAUC={agg['rank_auc']:.4f}  "
                      f"Hits@30={agg['hits@30']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
