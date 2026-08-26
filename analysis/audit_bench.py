"""
Audit NcRD-Bench splits for leakage and popularity bias.

The checks are intentionally mechanical.  A benchmark can only be trusted if
the files themselves show that:

  * held-out positives are absent from both forward and reverse message edges;
  * no sampled negative is a known positive;
  * evaluation folds keep the requested positive:negative ratio;
  * drug- or RNA-matched negatives really preserve the intended marginal;
  * cold-drug graphs contain no association edge to an invisible drug, in
    either ncRNA task.

Run after generating the benchmark:

    python analysis/audit_bench.py --bench ./9.bench

The command exits non-zero when a hard check fails, so it can be used directly
in a reproduction pipeline.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

TASKS = [("lncRNA", "LncDrug"), ("miRNA", "MiDrug")]
FOLD_NAMES = ("train", "val", "test")
ASSOCIATION_EDGES = [
    ("lncRNA", "LncDrug", "drug", "rev_LncDrug"),
    ("miRNA", "MiDrug", "drug", "rev_MiDrug"),
]


def pair_set(edge_index):
    if edge_index is None or edge_index.numel() == 0:
        return set()
    return {(int(r), int(d)) for r, d in edge_index.cpu().t().tolist()}


def as_pairs(values):
    return {(int(r), int(d)) for r, d in values}


def as_fold_pairs(values):
    """Read the column-oriented pair lists stored in split metadata."""
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[0] == 2:
        array = array.T
    return {(int(r), int(d)) for r, d in array.tolist()}


def expected_message_pairs(meta, name):
    if name in ("train", "val"):
        return as_fold_pairs(meta["train_pos"])
    return as_fold_pairs(meta["train_pos"]) | as_fold_pairs(meta["val_pos"])


def association_pairs(data, rna_type, edge_type, rev_type):
    """Target ncRNA-drug association pairs, forward and reverse, in a graph."""
    out = set()
    forward = (rna_type, edge_type, "drug")
    reverse = ("drug", rev_type, rna_type)
    if forward in data.edge_types:
        out |= pair_set(data[forward].edge_index)
    if reverse in data.edge_types:
        out |= {(int(d), int(r)) for r, d in data[reverse].edge_index.cpu().t().tolist()}
    return out


def association_drugs(data):
    """Drug IDs appearing in either ncRNA-drug association relation."""
    out = set()
    for rna_type, edge_type, drug_type, rev_type in ASSOCIATION_EDGES:
        forward = (rna_type, edge_type, drug_type)
        reverse = (drug_type, rev_type, rna_type)
        if forward in data.edge_types and data[forward].edge_index is not None:
            out.update(int(d) for d in data[forward].edge_index[1].cpu().tolist())
        if reverse in data.edge_types and data[reverse].edge_index is not None:
            out.update(int(d) for d in data[reverse].edge_index[0].cpu().tolist())
    return out


def visible_drugs(meta, name):
    fold = FOLD_NAMES.index(name)
    visible_folds = {0} if fold in (0, 1) else {0, 1}
    return set().union(*[
        set(meta["drug_folds"][i]) for i in visible_folds
    ]) if meta.get("drug_folds") else None


def check_ratio(data, key, neg_ratio, strict):
    labels = data[key].edge_label.cpu().numpy()
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    wanted = int(neg_ratio * positives)
    passed = (negatives == wanted) if strict else (negatives <= wanted)
    return passed, {
        "positives": positives,
        "negatives": negatives,
        "expected_negatives": wanted,
    }


def check_marginal(data, key, meta):
    labels = data[key].edge_label.cpu().numpy()
    index = data[key].edge_label_index.cpu().numpy()
    pos = index[:, labels == 1]
    neg = index[:, labels == 0]
    matching = meta.get("matching", "drug")

    if matching == "drug":
        pos_keys = pos[1].tolist()
        neg_keys = neg[1].tolist()
    elif matching == "rna":
        pos_keys = pos[0].tolist()
        neg_keys = neg[0].tolist()
    elif matching == "both":
        bins = np.asarray(meta.get("rna_degree_bins", []), dtype=int)
        if bins.size != meta["num_rna"]:
            return False, {"error": "degree bins missing or have the wrong size"}
        pos_keys = [(int(bins[int(r)]), int(d)) for r, d in pos.T.tolist()]
        neg_keys = [(int(bins[int(r)]), int(d)) for r, d in neg.T.tolist()]
    else:
        return True, {"matching": matching, "checked": False}

    positive_counts = collections.Counter(pos_keys)
    negative_counts = collections.Counter(neg_keys)
    positive_total = sum(positive_counts.values())
    negative_total = sum(negative_counts.values())
    keys_match = set(positive_counts) == set(negative_counts)
    proportions_match = keys_match and all(
        positive_counts[group] * negative_total
        == negative_counts[group] * positive_total
        for group in positive_counts
    )
    passed = proportions_match
    return passed, {
        "matching": matching,
        "positive_groups": len(positive_counts),
        "negative_groups": len(negative_counts),
        "keys_match": keys_match,
        "proportions_match": proportions_match,
    }


def degree_control_auc(train_data, test_data, key):
    degrees = collections.Counter(
        train_data[key].edge_index[1].cpu().tolist()
    )
    labels = test_data[key].edge_label.cpu().numpy()
    index = test_data[key].edge_label_index.cpu().numpy()
    scores = np.array([degrees[int(d)] for d in index[1]], dtype=float)
    if np.unique(labels).size < 2 or np.unique(scores).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def audit_split(bench, split, rna_type, edge_type):
    key = (rna_type, edge_type, "drug")
    meta = torch.load(f"{bench}/{split}/{edge_type}_meta.pth", map_location="cpu")
    graphs = {
        name: torch.load(
            f"{bench}/{split}/{edge_type}_{name}_data.pth", map_location="cpu"
        )
        for name in FOLD_NAMES
    }

    checks = {}
    known = as_pairs(meta["known_pairs"])
    for name, data in graphs.items():
        heldout = set()
        if name == "train":
            heldout |= as_fold_pairs(meta["val_pos"]) | as_fold_pairs(meta["test_pos"])
        elif name == "val":
            heldout |= as_fold_pairs(meta["val_pos"]) | as_fold_pairs(meta["test_pos"])
        else:
            heldout |= as_fold_pairs(meta["test_pos"])

        message = association_pairs(data, rna_type, edge_type, "rev_" + edge_type)
        leaked = sorted(heldout & message)
        checks[f"{name}_no_heldout_edge"] = {
            "passed": not leaked,
            "leaked_pairs": leaked[:20],
            "leak_count": len(leaked),
        }

        labels = data[key].edge_label.cpu().numpy()
        index = data[key].edge_label_index.cpu().numpy()
        negatives = as_pairs(index[:, labels == 0].T.tolist())
        known_negatives = sorted(negatives & known)
        checks[f"{name}_negatives_unknown"] = {
            "passed": not known_negatives,
            "known_negative_count": len(known_negatives),
            "examples": known_negatives[:20],
        }

        ratio_pass, ratio_details = check_ratio(
            data, key, meta.get("neg_ratio", 2.0), strict=(name != "train")
        )
        checks[f"{name}_negative_ratio"] = {
            "passed": ratio_pass,
            **ratio_details,
        }

        if name != "train":
            marginal_pass, marginal_details = check_marginal(data, key, meta)
            checks[f"{name}_matching_marginal"] = {
                "passed": marginal_pass,
                **marginal_details,
            }

        if meta["kind"] in ("cold_drug", "cold_pair"):
            visible = visible_drugs(meta, name)
            observed = association_drugs(data)
            leaked_drugs = sorted(observed - visible)
            checks[f"{name}_cold_drug_isolated"] = {
                "passed": not leaked_drugs,
                "visible_drugs": len(visible),
                "observed_association_drugs": len(observed),
                "leaked_drugs": leaked_drugs,
            }

    auc = degree_control_auc(graphs["train"], graphs["test"], key)
    matching = meta.get("matching", "drug")
    drug_neutral = matching in ("drug", "both")
    passed = np.isnan(auc) or abs(auc - 0.5) <= 1e-12 if drug_neutral else True
    checks["degree_control_neutral"] = {
        "passed": bool(passed),
        "auc": auc,
        "applicable": drug_neutral,
    }

    return {
        "split": split,
        "task": edge_type,
        "matching": matching,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default="./9.bench")
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    results = []
    failed = []
    for split in args.splits:
        for rna_type, edge_type in TASKS:
            try:
                result = audit_split(args.bench, split, rna_type, edge_type)
            except FileNotFoundError as exc:
                result = {
                    "split": split,
                    "task": edge_type,
                    "checks": {
                        "files_exist": {"passed": False, "error": str(exc)}
                    },
                }
            results.append(result)
            bad = [name for name, check in result["checks"].items() if not check["passed"]]
            status = "PASS" if not bad else "FAIL"
            print(f"[{status}] {split:12s} {edge_type:8s}", end="")
            if bad:
                print(f"  failed: {', '.join(bad)}")
                failed.append((split, edge_type, bad))
            else:
                degree = result["checks"].get("degree_control_neutral", {}).get("auc")
                print(f"  degree-AUC={degree:.4f}" if degree is not None else "")

    if args.json_out:
        directory = os.path.dirname(args.json_out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        print(f"wrote {args.json_out}")

    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
