"""
Popularity-bias diagnosis for the iNcRD-HG benchmark.

The benchmark negatives are drawn uniformly over (ncRNA, drug) pairs by
`T.RandomLinkSplit(neg_sampling_ratio=2.0)`, while the positives concentrate on
a handful of heavily-studied drugs.  A scorer that looks *only* at how many
training associations a drug has -- ignoring the ncRNA entirely -- therefore
separates the classes almost perfectly.

This script quantifies three things:

  1. how well trivial degree-only scorers do on the released test split;
  2. how strongly a trained iNcRD-HG model's own scores track drug degree;
  3. what happens to that model when negatives are matched to the same drug as
     the positive, which removes the popularity shortcut by construction.

Run from the directory holding ./3.heter_data and ./4.results:

    python analysis/popularity_bias_diagnosis.py

Note on (3): the model was *trained* against uniform negatives, so evaluating it
against same-drug negatives is a distribution shift as well as a debiasing.  The
number is a lower bound on what retraining with hard negatives would give, not a
final verdict on the architecture.
"""

import argparse
import collections
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

TASKS = [("lncRNA", "LncDrug"), ("miRNA", "MiDrug")]
NEG_DRAW_ATTEMPTS = 200


def degree_baselines(rna_type, edge_type, root):
    """AUC/AUPR of scorers that use nothing but node degree in the training graph."""
    train = torch.load(f"{root}/3.heter_data/{edge_type}_train_data.pth", map_location="cpu")
    test = torch.load(f"{root}/3.heter_data/{edge_type}_test_data.pth", map_location="cpu")

    edge_index = train[rna_type, edge_type, "drug"].edge_index
    rna_deg = collections.Counter(edge_index[0].tolist())
    drug_deg = collections.Counter(edge_index[1].tolist())

    label = test[rna_type, edge_type, "drug"].edge_label.numpy()
    index = test[rna_type, edge_type, "drug"].edge_label_index.numpy()

    scorers = {
        "drug degree only": np.array([drug_deg[d] for d in index[1]], dtype=float),
        "ncRNA degree only": np.array([rna_deg[r] for r in index[0]], dtype=float),
        "log-degree product": np.array(
            [np.log1p(rna_deg[r]) * np.log1p(drug_deg[d]) for r, d in zip(index[0], index[1])]
        ),
    }

    print(f"  test pairs: {len(label)} ({int(label.sum())} positive)")
    for name, score in scorers.items():
        print(
            f"  {name:20s} AUC={roc_auc_score(label, score):.4f} "
            f"AUPR={average_precision_score(label, score):.4f}"
        )

    pos_per_drug = collections.Counter(index[1][label == 1])
    total = sum(pos_per_drug.values())
    top20 = sum(count for _, count in pos_per_drug.most_common(20))
    print(
        f"  positives span {len(pos_per_drug)} drugs; the top 20 carry "
        f"{top20}/{total} = {top20 / total:.1%} of them"
    )


def sample_same_drug_negatives(positives, known_pairs, num_rna, rng):
    """For each positive (r, d) draw a negative (r', d) on the *same* drug.

    Matching the drug marginal between classes makes any drug-popularity signal
    worthless by construction, isolating whatever the model knows about the ncRNA.
    """
    negatives = []
    for _, drug in positives.T.tolist():
        for _ in range(NEG_DRAW_ATTEMPTS):
            candidate = int(rng.integers(0, num_rna))
            if (candidate, drug) not in known_pairs:
                negatives.append((candidate, drug))
                break
    return np.array(negatives).T


def debiased_evaluation(rna_type, edge_type, root, device, seed):
    """Re-score a trained checkpoint against drug-matched negatives."""
    import predict_association as pa

    # The saved decoder closes over these module-level globals.
    pa.RNA_type, pa.edge_type = rna_type, edge_type

    train = torch.load(f"{root}/3.heter_data/{edge_type}_train_data.pth", map_location=device)
    test = torch.load(f"{root}/3.heter_data/{edge_type}_test_data.pth", map_location=device)
    model = torch.load(f"{root}/4.results/model_our_{edge_type}.pth", map_location=device)
    model.eval()

    label = test[rna_type, edge_type, "drug"].edge_label.cpu().numpy()
    index = test[rna_type, edge_type, "drug"].edge_label_index.cpu().numpy()
    positives = index[:, label == 1]

    # Every association we know about, so a "negative" is never a masked positive.
    known = set(map(tuple, train[rna_type, edge_type, "drug"].edge_index.t().cpu().numpy().tolist()))
    known |= set(map(tuple, positives.T.tolist()))

    rng = np.random.default_rng(seed)
    negatives = sample_same_drug_negatives(
        positives, known, train[rna_type].node_id.numel(), rng
    )

    def score(pairs):
        edge = torch.tensor(pairs, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(test.x_dict, test.edge_index_dict, edge)
        return (out[0] if isinstance(out, tuple) else out).cpu().numpy()

    original = score(index)
    drug_deg = collections.Counter(train[rna_type, edge_type, "drug"].edge_index[1].tolist())
    rho = spearmanr(original, [drug_deg[d] for d in index[1]])
    print(
        f"  Spearman(model score, drug training degree) = {rho.statistic:.3f} "
        f"(p={rho.pvalue:.1e})"
    )
    print(
        f"  original split (uniform negatives) AUC={roc_auc_score(label, original):.4f} "
        f"AUPR={average_precision_score(label, original):.4f}"
    )

    y = np.r_[np.ones(positives.shape[1]), np.zeros(negatives.shape[1])]
    s = np.r_[score(positives), score(negatives)]
    print(
        f"  drug-matched negatives             AUC={roc_auc_score(y, s):.4f} "
        f"AUPR={average_precision_score(y, s):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="directory holding 3.heter_data and 4.results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="only run the degree baselines, which need no trained checkpoint",
    )
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.root))
    device = torch.device(args.device)

    for rna_type, edge_type in TASKS:
        print(f"=== {edge_type} ===")
        degree_baselines(rna_type, edge_type, args.root)
        if not args.skip_model:
            debiased_evaluation(rna_type, edge_type, args.root, device, args.seed)
        print()


if __name__ == "__main__":
    main()
