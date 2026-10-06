"""Validation-only diagnostic of unused information in frozen MiDrug inputs.

Every training pair has its own cell removed from label-derived features.
No test features or scores are constructed. This probe is not the final M123
model: its purpose is to decide which internal channels merit integration.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[2] / "release/iNcRD-HG-M1M2M3"
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(HERE.parent / "midrug_ranking_20261006"))
from incrd_hg_m123.data import load_fold, load_graph, structure_data, sha256
from incrd_hg_m123.config import ModelConfig
from incrd_hg_m123.pair_channels import PairStruct, ease_masked
from run_ranking import make_graph


def score(y, p):
    return dict(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)))


def content_kernels(x):
    x = x.double()
    cosine = F.normalize(x, dim=1) @ F.normalize(x, dim=1).T
    center = F.normalize(x - x.mean(1, keepdim=True), dim=1)
    corr = center @ center.T
    out = {}
    for name, sim in (("cos", cosine), ("corr", corr.clamp(min=0)), ("abscorr", corr.abs())):
        sim = sim.clone()
        sim.fill_diagonal_(0)
        for k in (5, 15, 40, 100):
            v, ix = sim.topk(min(k, len(x)-1), dim=1)
            mat = torch.zeros_like(sim).scatter(1, ix, v)
            out[f"{name}{k}"] = mat / mat.sum(1, keepdim=True).clamp(min=1e-12)
    return out


def features(train_pairs, train_y, query_pairs, graph, device="cpu"):
    """Features invariant to deleting a query's positive OR negative label."""
    graph = graph.to(device)
    D = structure_data(graph)
    ps = PairStruct(D, torch.as_tensor(train_pairs[train_y == 1], device=device), ModelConfig())
    r, d = torch.as_tensor(query_pairs.T, device=device)
    cols = [ps.qis(r, d), ps.ease(r, d)]
    names = [f"m123_qis_{i}" for i in range(cols[0].shape[1])] + [f"m123_ease_{i}" for i in range(cols[1].shape[1])]
    A = ps.A.double()
    N = torch.zeros_like(A)
    nr, nd = torch.as_tensor(train_pairs[train_y == 0].T, device=device)
    N[nr, nd] = 1
    qpos, qneg = A[r, d], N[r, d]

    def add(name, value):
        cols.append(value[:, None].float())
        names.append(name)

    for label, mat, q in (("pos", A, qpos), ("neg", N, qneg)):
        add(label + "_r_degree", torch.log1p(mat.sum(1)[r] - q))
        add(label + "_d_degree", torch.log1p(mat.sum(0)[d] - q))
    kr, kd = content_kernels(D.xr), content_kernels(D.xd)
    # Zero-diagonal content operators give exact single-cell exclusion.
    for side, kernels in (("rna", kr), ("drug", kd)):
        for name, K in kernels.items():
            p = K @ A if side == "rna" else A @ K.T
            n = K @ N if side == "rna" else N @ K.T
            add(f"{side}_{name}_pos", p[r, d])
            add(f"{side}_{name}_neg", n[r, d])
            add(f"{side}_{name}_ratio", (p / (p+n+0.01))[r, d])
    for name in ("cos15", "corr15", "abscorr15", "corr40", "abscorr40"):
        for label, mat in (("pos", A), ("neg", N)):
            add(f"bridge_{name}_{label}", (kr[name] @ mat @ kd[name].T)[r, d])

    # Spectral content priors in MC-EASE: Gram + lambda K^{-1}.
    for prior in ("identity", "cos", "corr"):
        priors = []
        for x in (D.xd, D.xr):
            x = x.double()
            if prior == "corr":
                x = x - x.mean(1, keepdim=True)
            x = F.normalize(x, dim=1)
            I = torch.eye(len(x), dtype=x.dtype, device=device)
            Kinv = I if prior == "identity" else torch.linalg.inv(x @ x.T + 0.1 * I)
            priors.append(Kinv)
        for lam in (0.1, 1., 10., 100.):
            for side, X, Kinv, u, i in (("d", A, priors[0], r, d), ("r", A.T, priors[1], d, r)):
                P = torch.linalg.inv(X.T @ X + lam * Kinv)
                v = ease_masked(X, P, u, i, qpos)
                add(f"ease_{prior}_{lam}_{side}", torch.sign(v)*torch.log1p(v.abs()*10))

    # Endpoint PCA uses only released node covariates, no association labels.
    for name, x, ix in (("rna", D.xr, r), ("drug", D.xd, d)):
        pc = PCA(n_components=32, random_state=42).fit_transform(x.cpu().numpy())
        for j in range(pc.shape[1]):
            add(f"pc_{name}_{j}", torch.as_tensor(pc[:, j], device=device)[ix])
    out = torch.cat(cols, dim=1).cpu().numpy().astype(np.float32)
    assert np.isfinite(out).all()
    return out, names


def main():
    torch.set_num_threads(2)
    output = HERE / "probe"
    output.mkdir(exist_ok=True)
    data = load_fold("MiDrug", 0)
    # Test fields never enter the featurizer, model, or diagnostics.
    train = {k: data[k] for k in ("train_pairs", "train_y", "val_pairs")}
    graph = make_graph(train)
    plan = dict(data="Frozen MiDrug fold0; miRNA/drug only", external_data=False,
                test_evaluated=False, depths=[2, 3, 5, 7], feature_sets=["structure", "content", "all"],
                min_child_weight=10, iterations=1600, learning_rate=.03, early_stopping_rounds=150,
                source_sha256=sha256(__file__), selection="Validation AUC; diagnostic only")
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    t0 = time.time()
    q = np.concatenate([train["train_pairs"], train["val_pairs"]])
    X, names = features(train["train_pairs"], train["train_y"], q, graph)
    n = len(train["train_y"])
    np.savez_compressed(output / "features.npz", train=X[:n], val=X[n:], names=names)
    # Direct deletion audit includes positive and negative training labels.
    inds = np.r_[np.flatnonzero(train["train_y"] == 1)[:3], np.flatnonzero(train["train_y"] == 0)[:3]]
    diffs = []
    for idx in inds:
        keep = np.arange(n) != idx
        rebuild, names2 = features(train["train_pairs"][keep], train["train_y"][keep], q[idx:idx+1], graph)
        assert names == names2
        np.testing.assert_allclose(X[idx:idx+1], rebuild, atol=2e-4, rtol=2e-4)
        diffs.append(float(np.max(np.abs(X[idx:idx+1] - rebuild))))
    (output / "checks.json").write_text(json.dumps(dict(status="passed", deletion_max_errors=diffs), indent=2))
    print("FEATURES", X.shape, "seconds", time.time()-t0, flush=True)
    ranking = HERE.parent / "midrug_ranking_20261006/results/validation"
    base = np.mean([np.load(ranking / f"bpr010_mixed_s{s}_predictions.npz")["probability"] for s in (42,43,44)], axis=0)
    records = []
    for fs in plan["feature_sets"]:
        mask = np.array([not k.startswith("pc_") and (fs != "structure" or k.startswith("m123_")) for k in names])
        if fs == "all":
            mask[:] = True
        for depth in plan["depths"]:
            tag = f"{fs}_d{depth}"
            model = XGBClassifier(n_estimators=1600, learning_rate=.03, max_depth=depth,
                min_child_weight=10, reg_lambda=10, subsample=.85, colsample_bytree=.85,
                tree_method="hist", device="cpu", n_jobs=4, random_state=42,
                eval_metric="auc", early_stopping_rounds=150)
            model.fit(X[:n, mask], train["train_y"], eval_set=[(X[n:, mask], data["val_y"])], verbose=False)
            p = model.predict_proba(X[n:, mask])[:, 1]
            model.save_model(output / f"{tag}.json")
            blends = []
            for w in (0, .25, .5, .75, 1.):
                pp = expit((1-w)*logit(np.clip(base, 1e-6, 1-1e-6)) + w*logit(np.clip(p, 1e-6, 1-1e-6)))
                blends.append(dict(weight=w, **score(data["val_y"], pp)))
            rec = dict(tag=tag, n_features=int(mask.sum()), best_iteration=model.best_iteration,
                       metrics=score(data["val_y"], p), m123_blends=blends, test_evaluated=False)
            records.append(rec)
            np.savez_compressed(output / f"{tag}_val.npz", probability=p, pairs=train["val_pairs"], y=data["val_y"])
            print(json.dumps(rec), flush=True)
            (output / "results.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
