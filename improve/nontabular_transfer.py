"""Non-foundation-model (non-tabular-FM) scorers for the accepted iNcRD-HG reference.

Three named, non-tabular techniques are evaluated on the frozen inner feature
banks and then re-checked on the three independent development resplits that
`validate_transfer_splits.py` created:

  1. `retrieval_knn`  - distance-weighted k-nearest-neighbour retrieval over the
                        evidence pair features (non-parametric, no fitted
                        parameters).
  2. `label_prop`     - transductive label propagation (Zhou et al. 2004) over a
                        kNN graph built on the pair features.  Uses the
                        unlabelled evaluation manifold, which the inductive
                        reference ignores.
  3. `pu_gbdt`        - prior-reweighted histogram gradient boosting.  The `0`
                        labels are sampled-unobserved, so positives are upweighted
                        by an estimated inverse propensity before fitting.

Every technique is combined with the frozen reference only through a fixed small
logit weight; no coefficient is chosen after seeing metrics.  The acceptance gate
is the project's own rule.  This module is a development screen, not a test-set
evaluation.

Run:
    python -m improve.nontabular_transfer report
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import sparse

from improve import tabular_foundation as base

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "results/foundation_transfer_20260914/inputs"
REFERENCE = ROOT / "results/tabicl_pathway_context_20260913"
SPLITS = ROOT / "results/transfer_splits_20260914"
OUT = ROOT / "results/nontabular_transfer_20260914"
SEEDS = (666, 0, 1)
RESPLITS = (42, 2026, 31415)
TASKS = ("MiDrug", "LncDrug")
MEMBERS = ("retrieval_knn", "label_prop", "pu_gbdt")
LOGIT_EPS = 1e-12


def logit(p):
    p = np.clip(np.asarray(p, float), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p) - np.log1p(-p)


def ref_inner(task, seed):
    path = REFERENCE / ("predictions/inner/%s/s%d/scores.npz" % (task, seed))
    with np.load(path, allow_pickle=False) as z:
        return z["tiered_0p5000"].copy(), z["y_val"].copy(), z["pairs_val"].copy()


def lp_transductive(Xtr, ytr, Xq, k=20, alpha=0.85, iters=30):
    from sklearn.neighbors import NearestNeighbors
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zq = (Xtr - mu) / sd, (Xq - mu) / sd
    X = np.vstack([Ztr, Zq])
    n, ntr = len(X), len(Ztr)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(X)
    dist, idx = nn.kneighbors(X)
    sigma = np.median(dist[:, 1]) + 1e-8
    w = np.exp(-(dist[:, 1:] ** 2) / (sigma ** 2))
    rows = np.repeat(np.arange(n), w.shape[1])
    cols = idx[:, 1:].ravel()
    W = sparse.csr_matrix((w.ravel(), (rows, cols)), shape=(n, n))
    W = (W + W.T) * 0.5
    deg = np.asarray(W.sum(1)).ravel()
    S = sparse.diags(np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)) @ W
    Y = np.zeros((n, 2))
    Y[np.arange(ntr), ytr.astype(int)] = 1.0
    F = Y.copy()
    for _ in range(iters):
        F = alpha * (S @ F) + (1 - alpha) * Y
    return F[ntr:, 1]


def knn_retrieval(Xtr, ytr, Xq, k=40):
    from sklearn.neighbors import KNeighborsClassifier
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    m = KNeighborsClassifier(n_neighbors=min(k, len(Xtr)), weights="distance", n_jobs=8)
    m.fit((Xtr - mu) / sd, ytr)
    return m.predict_proba((Xq - mu) / sd)[:, 1]


def pu_gbdt(Xtr, ytr, Xq, target_positive_rate=0.5):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pi = float(np.clip(ytr.mean(), 0.02, 0.98))
    # inverse-propensity weights: positives carry 1/pi, negatives 1/(1-pi),
    # then normalised so the effective positive rate approaches the target.
    w = np.where(ytr > 0, (1.0 / pi), (1.0 / (1.0 - pi)))
    w = w / w.mean()
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=20260914)
    m.fit(Xtr, ytr, sample_weight=w)
    return m.predict_proba(Xq)[:, 1]


def member_scores(name, Xtr, ytr, Xq):
    if name == "retrieval_knn":
        return knn_retrieval(Xtr, ytr, Xq)
    if name == "label_prop":
        return lp_transductive(Xtr, ytr, Xq)
    if name == "pu_gbdt":
        return pu_gbdt(Xtr, ytr, Xq)
    raise ValueError("Unknown member " + name)


def assess_inner(config_id, deltas):
    """Project gate: LncDrug every seed >= 0; MiDrug mean up and >=2/3 seeds up."""
    detail = {}
    for task in TASKS:
        da, dp = deltas[task]
        detail[task] = dict(
            delta_auc=da, delta_ap=dp,
            mean_auc=float(np.mean(da)), mean_ap=float(np.mean(dp)))
    eligible = (all(v >= 0 for v in detail["LncDrug"]["delta_auc"]) and
                all(v >= 0 for v in detail["LncDrug"]["delta_ap"]) and
                detail["MiDrug"]["mean_auc"] > 0 and detail["MiDrug"]["mean_ap"] > 0 and
                int((np.array(detail["MiDrug"]["delta_auc"]) > 0).sum()) >= 2 and
                int((np.array(detail["MiDrug"]["delta_ap"]) > 0).sum()) >= 2)
    return dict(recipe=config_id, eligible=bool(eligible), metrics=detail)


def run(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = {}
    # ---- inner partition: frozen banks + frozen reference -------------------
    for task in TASKS:
        for seed in SEEDS:
            with np.load(INPUTS / ("tiered62_%s.npz" % task), allow_pickle=False) as z:
                xtr, ytr, xq = z["x_train"], z["y_train"], z["x_val"]
            for name in MEMBERS:
                scores[("inner", task, seed, name)] = member_scores(name, xtr, ytr, xq).astype(np.float64)
    # ---- resplits: 36 graph columns, reference = accepted `tiered` arm ------
    for task in TASKS:
        for split in RESPLITS:
            with np.load(SPLITS / ("inputs/%s/%d/arrays.npz" % (task, split)), allow_pickle=False) as z:
                xtr, ytr, xq = z["x_train"], z["y_train"], z["x_val"]
            for name in MEMBERS:
                scores[("split%d" % split, task, split, name)] = member_scores(name, xtr, ytr, xq).astype(np.float64)

    rows = []
    for name in MEMBERS:
        for w in (0.1, 0.2, 0.3, 0.5):
            deltas = {"MiDrug": ([], []), "LncDrug": ([], [])}
            for task in TASKS:
                for seed in SEEDS:
                    ref, y, pairs = ref_inner(task, seed)
                    s = scores[("inner", task, seed, name)]
                    v = expit((1 - w) * logit(ref) + w * logit(s))
                    m = base.metrics(y, v, pairs)
                    r = base.metrics(y, ref, pairs)
                    deltas[task][0].append(m["auc"] - r["auc"])
                    deltas[task][1].append(m["ap"] - r["ap"])
            verdict = assess_inner("ref_%g_%s" % (w, name), deltas)
            rows.append(dict(scope="inner", member=name, w=w, verdict=verdict))

    # resplit cross-check for the fixed w=0.2 recipe
    for name in MEMBERS:
        for w in (0.1, 0.2, 0.3):
            rec = []
            for task in TASKS:
                for split in RESPLITS:
                    with np.load(SPLITS / ("inputs/%s/%d/arrays.npz" % (task, split)), allow_pickle=False) as z:
                        y, pairs = z["y_val"], z["pairs_val"]
                    with np.load(SPLITS / ("runs/%s/%d/tiered/scores.npz" % (task, split)), allow_pickle=False) as zz:
                        ref = zz["pred"].copy()
                    s = scores[("split%d" % split, task, split, name)]
                    v = expit((1 - w) * logit(ref) + w * logit(s))
                    m = base.metrics(y, v, pairs)
                    r = base.metrics(y, ref, pairs)
                    rec.append(dict(task=task, split=split, d_auc=m["auc"] - r["auc"], d_ap=m["ap"] - r["ap"]))
            rows.append(dict(scope="resplit", member=name, w=w, deltas=rec))

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inner_recipe_weights": [0.1, 0.2, 0.3, 0.5],
        "resplit_weights": [0.1, 0.2, 0.3],
        "members": list(MEMBERS),
        "gate": "LncDrug AUC/AP unchanged in every seed; MiDrug mean AUC/AP up with >=2/3 seeds positive.",
        "results": rows,
        "limitations": [
            "Historical inner partition and resplits are development data.",
            "Inference seeds are randomness, not independent biological splits.",
            "No outer, original test or source-holdout array is evaluated.",
        ],
    }
    (out / "RESULTS.json").write_text(json.dumps(summary, indent=2))
    inner = pd.DataFrame([dict(member=r["member"], w=r["w"], eligible=r["verdict"]["eligible"],
                               mi_auc=r["verdict"]["metrics"]["MiDrug"]["mean_auc"],
                               mi_ap=r["verdict"]["metrics"]["MiDrug"]["mean_ap"],
                               ln_auc=r["verdict"]["metrics"]["LncDrug"]["mean_auc"],
                               ln_ap=r["verdict"]["metrics"]["LncDrug"]["mean_ap"])
                          for r in rows if r["scope"] == "inner"])
    inner.to_csv(out / "inner_summary.csv", index=False)
    split_rows = []
    for r in rows:
        if r["scope"] != "resplit":
            continue
        for d in r["deltas"]:
            split_rows.append(dict(member=r["member"], w=r["w"], **d))
    pd.DataFrame(split_rows).to_csv(out / "resplit_deltas.csv", index=False)
    print(inner.to_string(index=False))
    print("WROTE", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report",))
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    {"report": run}[args.command](args.out_dir)


if __name__ == "__main__":
    main()
