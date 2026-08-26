"""
NcRD-Bench: evaluation splits for ncRNA-drug resistance prediction that do not
reward drug popularity.

`analysis/popularity_bias_diagnosis.py` shows the released split is solved to
AUC 0.978 by drug degree alone, because `T.RandomLinkSplit` draws negatives
uniformly over (ncRNA, drug) while positives pile onto a few studied drugs.
This module builds popularity-controlled splits:

  random     the original protocol, uniform negatives -- kept as a reference
             point so the drop on the other splits is directly readable
  matched_drug
             same positives, but every negative sits on the SAME drug as the
             positive it is matched to, so the drug marginal is identical in
             both classes and popularity carries no information
  matched_rna
             negatives share the ncRNA with a positive, neutralising the
             ncRNA-popularity marginal
  matched_both
             negatives share the drug and the ncRNA degree bin, neutralising
             both endpoints' coarse popularity signal
  cold_rna   ncRNAs are partitioned by node; test ncRNAs have no drug edge in
             the training graph at all, only expression similarity and ceRNA
             links, so the model must generalise from what the ncRNA *is*
  cold_drug  the same for drugs
  cold_pair  test pairs where both endpoints are held out

Every split except `random` uses popularity-controlled negatives.  `matched`
is retained as a backward-compatible alias for `matched_drug`.

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
SPLITS = [
    "random", "matched_drug", "matched_rna", "matched_both",
    "cold_rna", "cold_drug", "cold_pair", "matched",
]
DEFAULT_SPLITS = [
    "random", "matched_drug", "matched_rna", "matched_both",
    "cold_rna", "cold_drug", "cold_pair",
]
MATCHING_BY_SPLIT = {
    "random": "none",
    "matched": "drug",
    "matched_drug": "drug",
    "matched_rna": "rna",
    "matched_both": "both",
    "cold_rna": "drug",
    "cold_drug": "drug",
    "cold_pair": "drug",
}
FRACTIONS = (0.6, 0.2, 0.2)  # train / val / test
NEG_RATIO = 2.0  # matches the released protocol
DEGREE_BINS = 4


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

    if kind in ("random", "matched", "matched_drug", "matched_rna", "matched_both"):
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

def sample_uniform_negatives(positives, known, num_rna, num_drug, rng, neg_ratio):
    """The released protocol: any (ncRNA, drug) pair that is not a known positive."""
    wanted = int(neg_ratio * positives.shape[1])
    out = set()
    while len(out) < wanted:
        r = int(rng.integers(0, num_rna))
        d = int(rng.integers(0, num_drug))
        if (r, d) not in known and (r, d) not in out:
            out.add((r, d))
    return np.array(sorted(out)).T


def degree_bins(rna_degrees, num_bins=DEGREE_BINS):
    """Map ncRNA training degree to a deterministic popularity bin."""
    degrees = np.asarray(rna_degrees, dtype=float)
    positive = degrees[degrees > 0]
    if positive.size == 0 or num_bins <= 1:
        return np.zeros_like(degrees, dtype=int)
    edges = np.unique(np.quantile(positive, np.linspace(0, 1, num_bins + 1)[1:-1]))
    if edges.size == 0:
        return np.zeros_like(degrees, dtype=int)
    return np.searchsorted(edges, degrees, side="right")


def sample_matched_negatives(
    positives, known, allowed_rna, rng, strict, matching="drug",
    neg_ratio=NEG_RATIO, num_drug=None, rna_degree_bins=None,
):
    """Draw negatives that preserve a chosen endpoint-popularity marginal.

    ``drug`` fixes the drug marginal, ``rna`` fixes the ncRNA marginal, and
    ``both`` fixes the drug marginal while also matching the ncRNA's coarse
    training-degree bin.  A scorer that only knows the fixed endpoint therefore
    sits at AUC 0.5.

    A heavily studied group can want more negatives than the candidate pool holds
    -- in the cold splits the pool is a single node fold, and even transductively
    the most connected drug is associated with over a third of all lncRNAs.
    Simply capping the negatives there would quietly restore the popularity
    signal, so on the evaluation folds (`strict`) positives are dropped instead
    until the ratio holds exactly.  Training keeps every positive and takes
    whatever negatives the pool allows, since the bias only has to be neutral in
    the sets we score.  Returns the surviving positives alongside.
    """
    if matching == "rna" and num_drug is None:
        raise ValueError("num_drug is required for RNA-matched negatives")
    if matching == "both" and rna_degree_bins is None:
        raise ValueError("rna_degree_bins is required for degree-matched negatives")

    groups = {}
    for r, d in positives.T.tolist():
        if matching == "drug":
            key = int(d)
        elif matching == "rna":
            key = int(r)
        elif matching == "both":
            key = (int(rna_degree_bins[int(r)]), int(d))
        else:
            raise ValueError(f"unknown matching mode: {matching}")
        groups.setdefault(key, []).append((int(r), int(d)))

    kept_pos, negatives, dropped = [], set(), 0
    for key, rows in groups.items():
        if matching == "drug":
            drug = key
            pool = [
                (int(r), drug) for r in allowed_rna
                if (int(r), drug) not in known and (int(r), drug) not in negatives
            ]
        elif matching == "rna":
            rna = key
            pool = [
                (rna, int(d)) for d in range(num_drug)
                if (rna, int(d)) not in known and (rna, int(d)) not in negatives
            ]
        else:
            _, drug = key
            pool = [
                (int(r), drug) for r in allowed_rna
                if int(rna_degree_bins[int(r)]) == key[0]
                and (int(r), drug) not in known and (int(r), drug) not in negatives
            ]
        capacity = int(len(pool) // neg_ratio)
        if strict and capacity < len(rows):
            dropped += len(rows) - capacity
            rows = [
                (int(pair[0]), int(pair[1]))
                for pair in rng.choice(rows, size=capacity, replace=False)
            ] if capacity else []
        if not rows:
            continue
        kept_pos.extend(rows)
        take = min(int(neg_ratio * len(rows)), len(pool))
        chosen = rng.choice(len(pool), size=take, replace=False)
        negatives.update(pool[int(i)] for i in chosen)

    pos_arr = np.array(sorted(kept_pos)).T if len(kept_pos) else np.empty((2, 0), dtype=int)
    neg_arr = np.array(sorted(negatives)).T if len(negatives) else np.empty((2, 0), dtype=int)
    return pos_arr, neg_arr, dropped


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build_split(
    base, rna_type, edge_type, kind, seed,
    neg_ratio=NEG_RATIO, degree_bin_count=DEGREE_BINS,
):
    rng = np.random.default_rng(seed)
    rev_type = "rev_" + edge_type

    positives = deduplicate(base[rna_type, edge_type, "drug"].edge_index)
    num_rna = base[rna_type].node_id.numel()
    num_drug = base["drug"].node_id.numel()
    known = {(int(r), int(d)) for r, d in positives.t().tolist()}
    matching = MATCHING_BY_SPLIT[kind]
    rna_degrees = np.bincount(positives[0].numpy(), minlength=num_rna)
    rna_degree_bins = degree_bins(rna_degrees, degree_bin_count)

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
            neg = sample_uniform_negatives(pos, known, num_rna, num_drug, rng, neg_ratio)
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
                pos, known, list(allowed), rng, strict=(name != "train"),
                matching=matching, neg_ratio=neg_ratio, num_drug=num_drug,
                rna_degree_bins=rna_degree_bins,
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
            # Train and val graphs expose only training drugs; the test graph
            # may expose train+val drugs because its message graph contains
            # train+val positives.  Test drugs remain fully invisible.
            fold = ("train", "val", "test").index(name)
            visible = {0} if fold in (0, 1) else {0, 1}
            held = set().union(*[meta["drug_folds"][i] for i in range(3) if i not in visible])
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
        matching=matching, neg_ratio=neg_ratio, degree_bin_count=degree_bin_count,
        num_rna=num_rna, num_drug=num_drug,
        rna_degree_bins=rna_degree_bins.tolist(),
        known_pairs=sorted(known),
        train_pos=train_pos.tolist(), val_pos=val_pos.tolist(), test_pos=test_pos.tolist(),
    )
    return outputs, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="./9.bench")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, choices=SPLITS)
    parser.add_argument("--neg-ratio", type=float, default=NEG_RATIO)
    parser.add_argument("--degree-bins", type=int, default=DEGREE_BINS)
    args = parser.parse_args()

    base = load_base_graph(args.root)

    for kind in args.splits:
        os.makedirs(f"{args.out}/{kind}", exist_ok=True)
        for rna_type, edge_type in TASKS:
            outputs, meta = build_split(
                base, rna_type, edge_type, kind, args.seed,
                neg_ratio=args.neg_ratio, degree_bin_count=args.degree_bins,
            )
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
