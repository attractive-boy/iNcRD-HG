"""
NcRD-Bench: evaluation splits for ncRNA-drug resistance prediction that do not
reward drug popularity.

`analysis/popularity_bias_diagnosis.py` shows the released split is solved to
AUC 0.978 by drug degree alone, because `T.RandomLinkSplit` draws negatives
uniformly over (ncRNA, drug) while positives pile onto a few studied drugs.
This module builds five splits that close that gap:

  random     the original protocol, uniform negatives -- kept as a reference
             point so the drop on the other splits is directly readable
  matched    same positives, but every negative sits on the SAME drug as the
             positive it is matched to, so the drug marginal is identical in
             both classes and popularity carries no information
  cold_rna   ncRNAs are partitioned by node; test ncRNAs have no drug edge in
             the training graph at all, only expression similarity and ceRNA
             links, so the model must generalise from what the ncRNA *is*
  cold_drug  the same for drugs
  cold_pair  test pairs where both endpoints are held out

Every split except `random` uses same-drug negatives.

Files are written in exactly the schema `3.heter_data` uses, so existing
training code reads them without modification:

    ./9.bench/<split>/<EdgeType>_{train,val,test}_data.pth

A companion `<EdgeType>_meta.pth` records the node partition and the full set of
known positives, which `bench_metrics.py` needs for full-ranking evaluation.

Run from the directory holding ./3.heter_data:

    python analysis/ncrd_bench.py --out ./9.bench
"""

import argparse
import os

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

TASKS = [("lncRNA", "LncDrug"), ("miRNA", "MiDrug")]
SPLITS = ["random", "matched", "cold_rna", "cold_drug", "cold_pair"]
FRACTIONS = (0.6, 0.2, 0.2)  # train / val / test
NEG_RATIO = 2.0  # matches the released protocol


# --------------------------------------------------------------------------
# base graph
# --------------------------------------------------------------------------

def load_base_graph(root, sim_method="ken"):
    """Rebuild the full heterogeneous graph, before any edge is held out."""
    lnc = torch.load(f"{root}/3.heter_data/lncRNA_info.pth")
    mi = torch.load(f"{root}/3.heter_data/miRNA_info.pth")
    drug = torch.load(f"{root}/3.heter_data/drug_info.pth")
    lncmi = torch.load(f"{root}/3.heter_data/LncMi_info.pth")

    def norm(frame):
        x = torch.tensor(frame.values, dtype=torch.float)
        lo = x.min(dim=1, keepdim=True).values
        hi = x.max(dim=1, keepdim=True).values
        return (x - lo) / (hi - lo + 1e-8)

    data = HeteroData()
    data["lncRNA"].node_id = torch.arange(len(lnc["lncRNA_names"]))
    data["miRNA"].node_id = torch.arange(len(mi["miRNA_names"]))
    data["drug"].node_id = torch.arange(len(drug["drug_names"]))
    data["lncRNA"].x = norm(lnc["lncRNA_feature"])
    data["miRNA"].x = norm(mi["miRNA_feature"])
    data["drug"].x = norm(drug["drug_feature"])

    def edge(value):
        arr = value.values if hasattr(value, "values") else value
        return torch.tensor(np.asarray(arr), dtype=torch.long)

    data["lncRNA", "LncLnc", "lncRNA"].edge_index = edge(lnc[f"lncRNA_edge_{sim_method}"])
    data["miRNA", "MiMi", "miRNA"].edge_index = edge(mi[f"miRNA_edge_{sim_method}"])
    data["drug", "DrugDrug", "drug"].edge_index = edge(drug["drug_edge"])
    data["lncRNA", "LncDrug", "drug"].edge_index = edge(lnc["lncRNADrug_edge"])
    data["miRNA", "MiDrug", "drug"].edge_index = edge(mi["miRNADrug_edge"])
    data["lncRNA", "LncMi", "miRNA"].edge_index = edge(lncmi["LncMi_edge"])
    return data


def deduplicate(edge_index):
    """Drop repeated (ncRNA, drug) pairs; the source tables contain a few."""
    seen = sorted({(int(r), int(c)) for r, c in edge_index.t().tolist()})
    return torch.tensor(seen, dtype=torch.long).t()


# --------------------------------------------------------------------------
# partitioning
# --------------------------------------------------------------------------

def split_nodes_by_degree(positives, axis, num_nodes, rng):
    """Partition nodes into train/val/test, balancing degree across the folds.

    A uniform random partition would hand one fold most of the heavily-studied
    drugs and leave another with almost no positives to test on.  Walking the
    degree-sorted order and dealing nodes out in proportion to the fold sizes
    keeps the popularity profile comparable across folds.
    """
    degree = np.bincount(positives[axis].numpy(), minlength=num_nodes)
    order = np.argsort(-degree, kind="stable")
    folds = ([], [], [])
    quota = np.array(FRACTIONS, dtype=float)
    filled = np.zeros(3)
    for node in order:
        target = int(np.argmax(quota - filled / max(filled.sum(), 1)))
        folds[target].append(int(node))
        filled[target] += 1
    return [set(rng.permutation(f).tolist()) for f in folds]


def partition_positives(positives, kind, num_rna, num_drug, rng):
    """Return (train, val, test) positive pair arrays plus the node partition."""
    pairs = positives.t().numpy()
    meta = {}

    if kind in ("random", "matched"):
        order = rng.permutation(len(pairs))
        n_train = int(FRACTIONS[0] * len(pairs))
        n_val = int(FRACTIONS[1] * len(pairs))
        chunks = [order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]]
        parts = [pairs[c] for c in chunks]
    else:
        rna_folds = split_nodes_by_degree(positives, 0, num_rna, rng)
        drug_folds = split_nodes_by_degree(positives, 1, num_drug, rng)
        meta["rna_folds"] = [sorted(f) for f in rna_folds]
        meta["drug_folds"] = [sorted(f) for f in drug_folds]

        parts = []
        for fold in range(3):
            if kind == "cold_rna":
                keep = [i for i, (r, _) in enumerate(pairs) if r in rna_folds[fold]]
            elif kind == "cold_drug":
                keep = [i for i, (_, d) in enumerate(pairs) if d in drug_folds[fold]]
            else:  # cold_pair: both endpoints must belong to the fold
                keep = [
                    i for i, (r, d) in enumerate(pairs)
                    if r in rna_folds[fold] and d in drug_folds[fold]
                ]
            parts.append(pairs[keep])

    return [p.T for p in parts], meta


# --------------------------------------------------------------------------
# negative sampling
# --------------------------------------------------------------------------

def sample_uniform_negatives(positives, known, num_rna, num_drug, rng):
    """The released protocol: any (ncRNA, drug) pair that is not a known positive."""
    wanted = int(NEG_RATIO * positives.shape[1])
    out = set()
    while len(out) < wanted:
        r = int(rng.integers(0, num_rna))
        d = int(rng.integers(0, num_drug))
        if (r, d) not in known and (r, d) not in out:
            out.add((r, d))
    return np.array(sorted(out)).T


def sample_matched_negatives(positives, known, allowed_rna, rng, strict):
    """Negatives on the same drug as the positive they are matched to.

    Because each drug then contributes the same share of both classes, a scorer
    that only knows drug identity sits exactly at AUC 0.5.

    A heavily studied drug can want more negatives than the candidate pool holds
    -- in the cold splits the pool is a single node fold, and even transductively
    the most connected drug is associated with over a third of all lncRNAs.
    Simply capping the negatives there would quietly restore the popularity
    signal, so on the evaluation folds (`strict`) positives are dropped instead
    until the ratio holds exactly.  Training keeps every positive and takes
    whatever negatives the pool allows, since the bias only has to be neutral in
    the sets we score.  Returns the surviving positives alongside.
    """
    by_drug = {}
    for r, d in positives.T.tolist():
        by_drug.setdefault(d, []).append(r)

    kept_pos, negatives, dropped = [], set(), 0
    for drug, rows in by_drug.items():
        pool = [r for r in allowed_rna if (r, drug) not in known]
        capacity = int(len(pool) // NEG_RATIO)
        if strict and capacity < len(rows):
            dropped += len(rows) - capacity
            rows = list(rng.choice(rows, size=capacity, replace=False)) if capacity else []
        if not rows:
            continue
        kept_pos.extend((int(r), int(drug)) for r in rows)
        take = min(int(NEG_RATIO * len(rows)), len(pool))
        for r in rng.choice(pool, size=take, replace=False):
            negatives.add((int(r), int(drug)))

    pos_arr = np.array(sorted(kept_pos)).T if kept_pos else np.empty((2, 0), dtype=int)
    neg_arr = np.array(sorted(negatives)).T if negatives else np.empty((2, 0), dtype=int)
    return pos_arr, neg_arr, dropped


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build_split(base, rna_type, edge_type, kind, seed):
    rng = np.random.default_rng(seed)
    rev_type = "rev_" + edge_type

    positives = deduplicate(base[rna_type, edge_type, "drug"].edge_index)
    num_rna = base[rna_type].node_id.numel()
    num_drug = base["drug"].node_id.numel()
    known = {(int(r), int(d)) for r, d in positives.t().tolist()}

    parts, meta = partition_positives(positives, kind, num_rna, num_drug, rng)
    train_pos, val_pos, test_pos = parts

    # Message passing sees train positives when scoring val, and train+val when
    # scoring test -- the convention RandomLinkSplit uses.
    message = {
        "train": train_pos,
        "val": train_pos,
        "test": np.concatenate([train_pos, val_pos], axis=1) if val_pos.size else train_pos,
    }

    rna_folds = meta.get("rna_folds")
    outputs = {}
    for name, pos in (("train", train_pos), ("val", val_pos), ("test", test_pos)):
        if pos.size == 0:
            outputs[name] = None
            continue

        if kind == "random":
            neg = sample_uniform_negatives(pos, known, num_rna, num_drug, rng)
            sup_pos, dropped = pos, 0
        else:
            # Confine negatives to the same ncRNA fold as the positives, so a
            # cold split stays cold: every pair in the test set involves a
            # held-out ncRNA, positive or not.
            if kind in ("cold_rna", "cold_pair") and rna_folds is not None:
                allowed = rna_folds[("train", "val", "test").index(name)]
            else:
                allowed = list(range(num_rna))
            sup_pos, neg, dropped = sample_matched_negatives(
                pos, known, list(allowed), rng, strict=(name != "train")
            )

        if sup_pos.size == 0:
            outputs[name] = None
            continue
        meta.setdefault("dropped_positives", {})[name] = dropped

        data = base.clone()
        mp = torch.tensor(message[name], dtype=torch.long)

        if kind in ("cold_drug", "cold_pair"):
            # A held-out drug must be invisible to BOTH ncRNA types.  Removing it
            # only from the target relation would let the model read the drug's
            # profile off the other relation -- the leak that
            # predict_simulated_drug.py leaves open.
            held = set().union(*[meta["drug_folds"][i] for i in range(3)
                                 if i != ("train", "val", "test").index(name)])
            other_rna, other_edge = ("miRNA", "MiDrug") if rna_type == "lncRNA" else ("lncRNA", "LncDrug")
            oi = data[other_rna, other_edge, "drug"].edge_index
            data[other_rna, other_edge, "drug"].edge_index = oi[:, ~torch.isin(oi[1], torch.tensor(sorted(held)))]

        data[rna_type, edge_type, "drug"].edge_index = mp
        data[rna_type, edge_type, "drug"].edge_label_index = torch.tensor(
            np.concatenate([sup_pos, neg], axis=1), dtype=torch.long
        )
        data[rna_type, edge_type, "drug"].edge_label = torch.cat(
            [torch.ones(sup_pos.shape[1]), torch.zeros(neg.shape[1])]
        )

        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        # ToUndirected mirrors edge_label onto the reverse relation; supervision
        # belongs to the forward relation only.
        for attr in ("edge_label", "edge_label_index"):
            if attr in data["drug", rev_type, rna_type]:
                del data["drug", rev_type, rna_type][attr]
        outputs[name] = data

    meta.update(
        rna_type=rna_type, edge_type=edge_type, kind=kind, seed=seed,
        num_rna=num_rna, num_drug=num_drug,
        known_pairs=sorted(known),
        train_pos=train_pos.tolist(), val_pos=val_pos.tolist(), test_pos=test_pos.tolist(),
    )
    return outputs, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="./9.bench")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    args = parser.parse_args()

    base = load_base_graph(args.root)

    for kind in args.splits:
        os.makedirs(f"{args.out}/{kind}", exist_ok=True)
        for rna_type, edge_type in TASKS:
            outputs, meta = build_split(base, rna_type, edge_type, kind, args.seed)
            for name, data in outputs.items():
                if data is None:
                    print(f"  {kind:10s} {edge_type:8s} {name:5s}  EMPTY -- no positives in this fold")
                    continue
                torch.save(data, f"{args.out}/{kind}/{edge_type}_{name}_data.pth")
                label = data[rna_type, edge_type, "drug"].edge_label
                dropped = meta.get("dropped_positives", {}).get(name, 0)
                note = f"  ({dropped} positives dropped to hold the ratio)" if dropped else ""
                print(
                    f"  {kind:10s} {edge_type:8s} {name:5s}  "
                    f"{int(label.sum()):5d} pos / {int((label == 0).sum()):5d} neg{note}"
                )
            torch.save(meta, f"{args.out}/{kind}/{edge_type}_meta.pth")
        print()


if __name__ == "__main__":
    main()
