"""Small, validation-only screen on the ORIGINAL three-type iNcRD-HG graph.

Each comparison is frozen before training. No test file is opened. The only
common data correction removes a training copy of a validation pair. This is
not the circRNA-expanded graph or a TabICL experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from improve.model import Model
from improve.network_feature_ablation import ExperimentModel
from improve.run import TASKS, set_seed

HERE = Path(__file__).resolve().parent
DATA = REPO.parent / "incrd-orig"
ARMS = ("base", "pca64", "assoc030", "half_control", "half_target_mask")
SEEDS = (666, 0, 1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"scope": "original three-node-type GeneralConv GNN; validation-only exploratory screen",
        "arms": list(ARMS), "seeds": list(SEEDS),
        "comparisons": {"pca64": "base", "assoc030": "base", "half_target_mask": "half_control"},
        "task_settings": {t: dict(c, train_epochs=c["epochs"] - 1) for t, c in TASKS.items()},
        "optimizer": "RMSprop, alpha=0.99, weight_decay=0, task-specific original learning rates",
        "selection": "first strict maximum of mean validation AUC and AP; one checkpoint for both metrics",
        "screen_rule": "all four mean AUC/AP deltas across both tasks positive; each metric positive in at least 2/3 seeds; descriptive only",
        "mask_rule": "two fixed stratified halves of the training pairs alternate by epoch; mask the current half's positive target edges and reverses; other relations retained",
        "mask_control": "identical alternating supervised halves, initialization and optimizer-update budget, with target edges retained",
        "mask_limit": "block masking inspired by target-link exclusion literature, not an exact SpotTarget reproduction; evaluation uses the full training graph",
        "association_rule": "all six directed cross-type association relations multiplied by 0.3; topology and same-type weights unchanged",
        "pca_rule": "raw RNA expression; training-node StandardScaler, 64 PCs, PC StandardScaler; no added statistics; drug features unchanged",
        "validation_policy": "fixed original validation pairs/labels; common removal of any matching training copy; no test access or test claim",
        "source_files": {str(p.relative_to(REPO)): digest(p) for p in [Path(__file__).resolve(), REPO / "improve/model.py", REPO / "improve/network_feature_ablation.py", REPO / "improve/run.py"]},
        "inputs": {}, "audit": {}}
    records = {}
    for task, config in TASKS.items():
        nt = config["rna"]
        key, rev = (nt, task, "drug"), ("drug", "rev_" + task, nt)
        paths = {f: DATA / "3.heter_data" / f"{task}_{f}_data.pth" for f in ("train", "val")}
        tr, va = [torch.load(paths[f], map_location="cpu") for f in ("train", "val")]
        assert set(tr.node_types) == {"lncRNA", "miRNA", "drug"}
        tp, ty = tr[key].edge_label_index.numpy().copy(), tr[key].edge_label.numpy().copy()
        vp, vy = va[key].edge_label_index.numpy().copy(), va[key].edge_label.numpy().copy()
        assert len(set(map(tuple, vp.T))) == len(vy)
        val_set, seen, keep, removed = set(map(tuple, vp.T)), {}, [], []
        for i, (pair, y) in enumerate(zip(tp.T, ty)):
            pair = tuple(pair)
            if pair in seen and seen[pair] != y:
                raise ValueError("Conflicting training labels")
            reason = "overlaps_validation" if pair in val_set else "duplicate_training" if pair in seen else None
            if reason:
                removed.append({"row": i, "pair": [int(x) for x in pair], "label": int(y), "reason": reason})
            else:
                keep.append(i)
            seen[pair] = y
        tp, ty = tp[:, keep], ty[keep]
        positive = set(map(tuple, tp[:, ty == 1].T))
        tr = tr.clone()
        old_edges = tr[key].edge_index.numpy()
        retain = np.array([tuple(p) in positive for p in old_edges.T])
        tr[key].edge_index = torch.from_numpy(old_edges[:, retain].copy())
        tr[rev].edge_index = tr[key].edge_index.flip(0).clone()
        tr[key].edge_label_index, tr[key].edge_label = torch.from_numpy(tp), torch.from_numpy(ty)
        assert set(map(tuple, tr[key].edge_index.numpy().T)) == positive
        assert not positive.intersection(val_set)
        assert torch.equal(tr[key].edge_index.flip(0), tr[rev].edge_index)
        for nt0 in tr.node_types:
            assert torch.isfinite(tr[nt0].x).all()
        target_dir = out / task
        target_dir.mkdir()
        torch.save(tr, target_dir / "train_graph.pth")
        records[task] = {"graph": tr, "pairs_train": tp, "y_train": ty, "pairs_val": vp, "y_val": vy}
        audit = {"node_features": {n: list(tr[n].x.shape) for n in tr.node_types},
            "edge_counts": {"|".join(k): int(v.shape[1]) for k, v in tr.edge_index_dict.items()},
            "train_positive": int(ty.sum()), "train_unlabeled": int((ty == 0).sum()),
            "val_positive": int(vy.sum()), "val_unlabeled": int((vy == 0).sum()),
            "removed_training_rows": removed, "train_positive_target_presence": 1.0,
            "val_target_presence": 0, "reverse_edges_consistent": True}
        manifest["audit"][task] = audit
        manifest["inputs"].update({str(p): digest(p) for p in paths.values()})
    pca_values, transform_arrays = {}, {}
    for task, config in TASKS.items():
        nt = config["rna"]
        fname = "LncExp.csv" if nt == "lncRNA" else "MiExp.csv"
        path = DATA / "2.construct_network" / fname
        frame = pd.read_csv(path, index_col=0).T
        if nt == "miRNA":
            frame = frame[sorted(frame.columns, key=lambda c: int(c.split("-")[-1]))]
        raw = frame.values.astype(np.float32)
        fit_ids = np.unique(records[task]["pairs_train"][0])
        tensor = torch.from_numpy(raw)
        expected = (tensor - tensor.min(1, keepdim=True).values) / (
            tensor.max(1, keepdim=True).values - tensor.min(1, keepdim=True).values + (1e-8 if nt == "lncRNA" else 0))
        np.testing.assert_allclose(expected.numpy(), records[task]["graph"][nt].x.numpy(), atol=2e-6, rtol=2e-6)
        scaler = StandardScaler().fit(raw[fit_ids])
        scaled = scaler.transform(raw)
        pca = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(scaled[fit_ids])
        z = pca.transform(scaled)
        post = StandardScaler().fit(z[fit_ids])
        pca_values[nt] = post.transform(z).astype(np.float32)
        for k, v in {"fit_ids": fit_ids, "input_mean": scaler.mean_, "input_scale": scaler.scale_,
            "pca_mean": pca.mean_, "components": pca.components_, "explained_variance_ratio": pca.explained_variance_ratio_,
            "post_mean": post.mean_, "post_scale": post.scale_}.items():
            transform_arrays[nt + "_" + k] = v
        manifest["inputs"][str(path)] = digest(path)
        manifest["audit"][task]["pca_fitted_rna_nodes"] = len(fit_ids)
        manifest["audit"][task]["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
    np.savez_compressed(out / "pca_transform.npz", **transform_arrays)
    for task, record in records.items():
        arrays = {k: v for k, v in record.items() if k != "graph"}
        arrays.update({"pca_" + nt: v for nt, v in pca_values.items()})
        np.savez_compressed(out / task / "train_val_inputs.npz", **arrays)
    manifest["cached_inputs"] = {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}
    write_json(out / "protocol.json", manifest)
    print("PREPARED", json.dumps(manifest["audit"], ensure_ascii=False), flush=True)


def metrics(y, pred, pairs):
    macro = []
    for drug in np.unique(pairs[1]):
        use = pairs[1] == drug
        if np.unique(y[use]).size == 2:
            macro.append(average_precision_score(y[use], pred[use]))
    return {"val_auc": float(roc_auc_score(y, pred)), "val_ap": float(average_precision_score(y, pred)),
            "macro_drug_ap": float(np.mean(macro)), "macro_drug_count": len(macro)}


def partitions(y, seed):
    rng = np.random.default_rng(seed + 31991)
    groups = np.empty(len(y), dtype=np.int64)
    for label in (0, 1):
        ids = rng.permutation(np.flatnonzero(y == label))
        groups[ids] = np.arange(len(ids)) % 2
    return groups


def fit_one(task, arm, seed, graph, arrays, device, epochs):
    start = time.perf_counter()
    config = TASKS[task]
    nt, hidden = config["rna"], config["channels"]
    key, rev = (nt, task, "drug"), ("drug", "rev_" + task, nt)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    g = graph.clone()
    if arm == "pca64":
        for rna in ("lncRNA", "miRNA"):
            g[rna].x = torch.from_numpy(arrays["pca_" + rna])
    g = g.to(device)
    indices = {f: torch.from_numpy(arrays["pairs_" + f]).to(device) for f in ("train", "val")}
    target = torch.from_numpy(arrays["y_train"]).to(device)
    weights = {k: torch.ones(v.shape[1], device=device) for k, v in g.edge_index_dict.items()}
    model = ExperimentModel(g, task).to(device)
    with torch.no_grad():
        first = model(g.x_dict, g.edge_index_dict, indices["train"], weights)
    initial_hash = state_digest(model)
    if arm == "base":
        # Prove unit-weight propagation is the original baseline implementation.
        ordinary = Model(g, hidden, config["aggr"], nt).to(device)
        with torch.no_grad():
            ordinary(g.x_dict, g.edge_index_dict, indices["train"])
        ordinary.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            check = ordinary(g.x_dict, g.edge_index_dict, indices["train"])
        torch.testing.assert_close(first, check, atol=2e-5, rtol=2e-5)
        del ordinary
    if arm == "assoc030":
        weights = {k: w * (0.3 if k[0] != k[2] else 1.0) for k, w in weights.items()}
    groups = partitions(arrays["y_train"], seed)
    query_rows = [torch.from_numpy(np.flatnonzero(groups == i)).to(device) for i in (0, 1)]
    pos_group = {tuple(pair): int(group) for pair, y, group in zip(arrays["pairs_train"].T, arrays["y_train"], groups) if y == 1}
    message_group = torch.tensor([pos_group[tuple(p)] for p in graph[key].edge_index.numpy().T], device=device)
    masked = []
    for half in (0, 1):
        keep = message_group != half
        edges, w = dict(g.edge_index_dict), dict(weights)
        edges[key] = g[key].edge_index[:, keep]
        edges[rev] = edges[key].flip(0)
        w[key], w[rev] = weights[key][keep], weights[rev][keep]
        assert torch.equal(edges[key].flip(0), edges[rev])
        omitted = {p for p, h in pos_group.items() if h == half}
        assert not omitted.intersection(map(tuple, edges[key].cpu().numpy().T))
        masked.append((edges, w))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"], alpha=0.99)
    trace, best, state, best_pred, best_epoch = [], -np.inf, None, None, None
    for epoch in range(1, epochs + 1):
        half = (epoch - 1) % 2
        subset = query_rows[half] if arm in ("half_control", "half_target_mask") else slice(None)
        edges, w = masked[half] if arm == "half_target_mask" else (g.edge_index_dict, weights)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(g.x_dict, edges, indices["train"][:, subset], w)
        loss = F.binary_cross_entropy_with_logits(logits, target[subset])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(g.x_dict, g.edge_index_dict, indices["val"], weights).cpu().numpy()
        auc, ap = roc_auc_score(arrays["y_val"], pred), average_precision_score(arrays["y_val"], pred)
        criterion = (auc + ap) / 2
        trace.append({"epoch": epoch, "loss": float(loss.detach()), "val_auc": auc, "val_ap": ap, "criterion": criterion})
        if criterion > best:
            best, best_epoch, best_pred = criterion, epoch, pred.copy()
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        replay = model(g.x_dict, g.edge_index_dict, indices["val"], weights).cpu().numpy()
    np.testing.assert_allclose(replay, best_pred, atol=3e-5, rtol=3e-5)
    row = dict(task=task, arm=arm, seed=seed, selected_epoch=best_epoch, epochs=epochs,
        initial_state_sha256=initial_hash, parameters=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=time.perf_counter() - start, replay_max_error=float(np.max(np.abs(replay-best_pred))),
        **metrics(arrays["y_val"], best_pred, arrays["pairs_val"]))
    return row, best_pred, trace, state


def run(args):
    out = Path(args.out)
    protocol = json.loads((out / "protocol.json").read_text())
    for path, sha in protocol["source_files"].items():
        assert digest(REPO / path) == sha, "Source changed after protocol freeze"
    for path, sha in protocol["cached_inputs"].items():
        assert digest(out / path) == sha, "Frozen inputs changed"
    g = torch.load(out / args.task / "train_graph.pth", map_location="cpu")
    with np.load(out / args.task / "train_val_inputs.npz", allow_pickle=False) as z:
        assert not any("test" in k for k in z.files)
        arrays = {k: z[k] for k in z.files}
    seeds = (666,) if args.smoke else SEEDS
    epochs = 2 if args.smoke else protocol["task_settings"][args.task]["train_epochs"]
    for seed in seeds:
        raw_hash = None
        for arm in ARMS:
            dest = out / ("smoke" if args.smoke else "runs") / f"{args.task}_{arm}_s{seed}"
            dest.mkdir(parents=True, exist_ok=False)
            result, pred, trace, state = fit_one(args.task, arm, seed, g, arrays, args.device, epochs)
            if arm == "base":
                raw_hash = result["initial_state_sha256"]
            if arm != "pca64":
                assert result["initial_state_sha256"] == raw_hash, "Raw-input arms must start identically"
            pd.DataFrame(trace).to_csv(dest / "trace.csv", index=False)
            np.savez_compressed(dest / "val_scores.npz", pred=pred, y=arrays["y_val"], pairs=arrays["pairs_val"])
            torch.save(state, dest / "checkpoint.pt")
            result["artifacts_sha256"] = {p.name: digest(p) for p in dest.iterdir() if p.is_file()}
            write_json(dest / "result.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)


def report(out):
    rows = []
    for task in TASKS:
        for seed in SEEDS:
            for arm in ARMS:
                dest = out / "runs" / f"{task}_{arm}_s{seed}"
                row = json.loads((dest / "result.json").read_text())
                for p, sha in row["artifacts_sha256"].items():
                    assert digest(dest / p) == sha
                with np.load(dest / "val_scores.npz") as a:
                    actual = metrics(a["y"], a["pred"], a["pairs"])
                for k, v in actual.items():
                    assert abs(row[k] - v) < 1e-12
                trace = pd.read_csv(dest / "trace.csv")
                assert int(trace.loc[trace.criterion.idxmax(), "epoch"]) == row["selected_epoch"]
                rows.append({k: v for k, v in row.items() if k != "artifacts_sha256"})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_run.csv", index=False)
    summary = frame.groupby(["task", "arm"])[["val_auc", "val_ap", "macro_drug_ap", "elapsed_seconds", "parameters"]].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    contrasts = []
    for candidate, control in (("pca64", "base"), ("assoc030", "base"), ("half_control", "base"), ("half_target_mask", "half_control"), ("half_target_mask", "base")):
        for task in TASKS:
            a = frame[(frame.task == task) & (frame.arm == candidate)].set_index("seed")
            b = frame[(frame.task == task) & (frame.arm == control)].set_index("seed")
            row = {"task": task, "candidate": candidate, "control": control}
            for metric in ("val_auc", "val_ap", "macro_drug_ap"):
                d = a[metric] - b[metric]
                row[metric + "_delta"] = float(d.mean())
                row[metric + "_positive_seeds"] = int((d > 0).sum())
                row[metric + "_paired_sd"] = float(d.std(ddof=1))
            contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(out / "contrasts.csv", index=False)
    assessment = {}
    for candidate, control in (("pca64", "base"), ("assoc030", "base"), ("half_target_mask", "half_control")):
        relevant = [x for x in contrasts if x["candidate"] == candidate and x["control"] == control]
        passed = all(x[m + "_delta"] > 0 and x[m + "_positive_seeds"] >= 2 for x in relevant for m in ("val_auc", "val_ap"))
        assessment[candidate] = {"control": control, "passed_joint_descriptive_screen": passed}
    write_json(out / "assessment.json", assessment)
    write_json(out / "completion.json", {"runs": len(rows), "gpu_training_seconds_sum": float(frame.elapsed_seconds.sum()),
        "max_checkpoint_replay_error": float(frame.replay_max_error.max()), "checks": ["30 expected results", "artifact hashes", "metric recomputation", "single checkpoint selection", "unit weights match original base", "raw-arm matched initialization", "forward/reverse target masking", "train/val pair disjointness"],
        "test_evaluated": False, "independent_validation_claimed": False, "baseline_promoted": False})
    print(summary.to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))
    print(json.dumps(assessment, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "run", "report"))
    p.add_argument("--out", required=True)
    p.add_argument("--task", choices=tuple(TASKS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        prepare(Path(args.out))
    elif args.command == "run":
        run(args)
    else:
        report(Path(args.out))


if __name__ == "__main__":
    main()
