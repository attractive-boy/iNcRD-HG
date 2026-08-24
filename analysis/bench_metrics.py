"""
Metrics for NcRD-Bench.

Global AUC/AUPR over a pool of sampled pairs is what the literature reports, so
it is kept for comparability, but it is only meaningful once the negatives are
drug-matched -- otherwise it measures drug popularity (see
`popularity_bias_diagnosis.py`).

Two further readouts are reported because they cannot be gamed by popularity at
all:

  per-drug macro  AUC/AP computed within each drug, then averaged over drugs with
                  both classes present.  Every drug counts once, so a model that
                  nails the ten famous drugs and fails the rest cannot hide.

  full ranking    for each test drug, score *every* candidate ncRNA and ask where
                  the true partners land (Hits@k, MRR, NDCG@k).  No negative
                  sampling at all.  This is the shape of the actual screening
                  task: given a drug, hand a biologist a shortlist of ncRNAs.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def pairwise_metrics(labels, scores, drugs):
    """Global and per-drug-macro AUC/AUPR over a set of scored pairs."""
    out = {
        "auc": roc_auc_score(labels, scores),
        "aupr": average_precision_score(labels, scores),
    }

    per_auc, per_ap = [], []
    for drug in np.unique(drugs):
        mask = drugs == drug
        y = labels[mask]
        if y.min() == y.max():  # only one class for this drug -- AUC undefined
            continue
        per_auc.append(roc_auc_score(y, scores[mask]))
        per_ap.append(average_precision_score(y, scores[mask]))

    out["auc_per_drug"] = float(np.mean(per_auc)) if per_auc else float("nan")
    out["ap_per_drug"] = float(np.mean(per_ap)) if per_ap else float("nan")
    out["n_drugs_scored"] = len(per_auc)
    return out


def ranking_metrics(score_fn, test_pos, candidate_rna, exclude, ks=(10, 30, 50), seed=0):
    """Rank every candidate ncRNA for each test drug against its true partners.

    score_fn      maps a 2 x N array of (ncRNA, drug) pairs to N scores
    test_pos      2 x P array of held-out positives
    candidate_rna ncRNAs eligible to be ranked (the test fold in a cold split)
    exclude       pairs already visible in training, dropped from the ranking so
                  a model is not penalised for putting known partners on top

    Ties are broken at random rather than by node index.  A constant scorer -- the
    drug-degree control, which cannot distinguish ncRNAs under a fixed drug --
    otherwise inherits whatever ordering the index happens to impose and lands
    well off 0.5, making a chance-level baseline look actively harmful.
    """
    rng = np.random.default_rng(seed)
    by_drug = {}
    for r, d in np.asarray(test_pos).T.tolist():
        by_drug.setdefault(int(d), set()).add(int(r))

    candidates = np.asarray(sorted(candidate_rna))
    hits = {k: [] for k in ks}
    ndcg = {k: [] for k in ks}
    reciprocal, drug_auc = [], []

    for drug, truth in by_drug.items():
        pool = np.array([r for r in candidates if (r, drug) not in exclude or r in truth])
        if len(pool) < 2:
            continue
        pairs = np.vstack([pool, np.full(len(pool), drug)])
        scores = score_fn(pairs)
        shuffle = rng.permutation(len(pool))
        order = pool[shuffle][np.argsort(-scores[shuffle], kind="stable")]
        relevant = np.isin(order, list(truth)).astype(float)

        if relevant.sum() == 0 or relevant.sum() == len(relevant):
            continue

        n_true = int(relevant.sum())
        for k in ks:
            hits[k].append(relevant[:k].sum() / min(k, n_true))
            gain = relevant[:k] / np.log2(np.arange(2, min(k, len(order)) + 2))
            ideal = np.ones(min(k, n_true)) / np.log2(np.arange(2, min(k, n_true) + 2))
            ndcg[k].append(gain.sum() / ideal.sum())

        first = np.argmax(relevant) + 1
        reciprocal.append(1.0 / first)
        # AUC of this drug's ranking: probability a true partner outranks a non-partner
        ranks = np.arange(1, len(order) + 1)
        pos_ranks = ranks[relevant == 1]
        n_neg = len(order) - n_true
        drug_auc.append(1.0 - (pos_ranks.sum() - n_true * (n_true + 1) / 2) / (n_true * n_neg))

    out = {"mrr": float(np.mean(reciprocal)) if reciprocal else float("nan"),
           "rank_auc": float(np.mean(drug_auc)) if drug_auc else float("nan"),
           "n_drugs_ranked": len(reciprocal)}
    for k in ks:
        out[f"hits@{k}"] = float(np.mean(hits[k])) if hits[k] else float("nan")
        out[f"ndcg@{k}"] = float(np.mean(ndcg[k])) if ndcg[k] else float("nan")
    return out
