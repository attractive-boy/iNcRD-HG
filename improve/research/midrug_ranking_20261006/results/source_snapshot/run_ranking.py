"""MiDrug, miRNA/drug only: train-only same-drug BCE + BPR experiment.

Training writes validation results only. Test evaluation requires a frozen
selection.json created by experiment.py after all validation runs finish.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[2] / "release" / "iNcRD-HG-M1M2M3"
sys.path.insert(0, str(RELEASE))
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.data import load_fold, load_graph, sha256, split_path
from incrd_hg_m123.engine import metrics, seed_all

SEEDS = (42, 43, 44)
CONFIGS = {
    "bce": {"rank_weight": 0.0, "sampler": "uniform"},
    "bpr005_mixed": {"rank_weight": 0.05, "sampler": "mixed_middle"},
    "bpr010_mixed": {"rank_weight": 0.1, "sampler": "mixed_middle"},
    "bpr020_mixed": {"rank_weight": 0.2, "sampler": "mixed_middle"},
    "bpr010_uniform": {"rank_weight": 0.1, "sampler": "uniform"},
}
RECIPE = dict(epochs=300, lr=1e-4, weight_decay=0.0, optimizer="RMSprop", alpha=0.99,
              warmup_epochs=30, ramp_epochs=30, candidate_count=8, negatives_per_positive=2)


def source_hashes():
    files = [Path(__file__), *sorted((RELEASE / "incrd_hg_m123").glob("*.py"))]
    return {str(p.relative_to(HERE.parents[2])): sha256(p) for p in files}


def signature():
    obj = dict(source=source_hashes(), configs=CONFIGS, recipe=RECIPE, seeds=SEEDS,
               split_sha256=sha256(split_path("MiDrug", 0)),
               graph_sha256=sha256(RELEASE / "data/frozen/MiDrug_graph.pt"))
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def make_graph(data):
    """Fresh graph: no lncRNA stores, attributes, edges, or supervision."""
    original = load_graph("MiDrug")
    graph = HeteroData()
    for node in ("miRNA", "drug"):
        graph[node].x = original[node].x.clone()
    for edge in (("miRNA", "MiMi", "miRNA"), ("drug", "DrugDrug", "drug")):
        graph[edge].edge_index = original[edge].edge_index.clone()
    pos = data["train_pairs"][data["train_y"] == 1]
    edges = torch.as_tensor(pos.T.copy(), dtype=torch.long)
    graph["miRNA", "MiDrug", "drug"].edge_index = edges
    graph["drug", "rev_MiDrug", "miRNA"].edge_index = edges.flip(0)
    assert set(graph.node_types) == {"miRNA", "drug"}
    assert all("lnc" not in str(e).lower() for e in graph.edge_types)
    return graph


def make_adapter(data, device, include_test=False):
    # Test identities and labels are excluded from training feature preparation.
    inputs = {k: data[k] for k in ("train_pairs", "train_y", "val_pairs")}
    inputs["test_pairs"] = data["test_pairs"] if include_test else np.empty((0, 2), dtype=np.int64)
    return Adapter(inputs, make_graph(inputs), device)


class SameDrugRanking:
    """Use indices into the given training pairs, never an external negative pool.

    Each anchor draws eight same-drug training negatives with replacement.
    Uniform: use the first two. Mixed-middle: one uniform plus one sampled
    from the middle four ranks by current detached negative score. This is a
    score-quantile heuristic, not a claim that candidates are true negatives.
    """
    def __init__(self, pairs, labels, device, seed, mode):
        if mode not in ("uniform", "mixed_middle"):
            raise ValueError(mode)
        self.mode = mode
        negative = np.flatnonzero(labels == 0)
        n_drug = int(pairs[:, 1].max()) + 1
        groups = [negative[pairs[negative, 1] == d] for d in range(n_drug)]
        counts = np.array([len(g) for g in groups])
        positive = np.flatnonzero(labels == 1)
        eligible = positive[counts[pairs[positive, 1]] > 0]
        if len(eligible) == 0:
            raise ValueError("No positive anchors with same-drug training negatives")
        table = np.zeros((n_drug, int(counts.max())), dtype=np.int64)
        for d, group in enumerate(groups):
            table[d, :len(group)] = group
        self.anchors = torch.as_tensor(eligible, dtype=torch.long, device=device)
        self.drugs = torch.as_tensor(pairs[eligible, 1], dtype=torch.long, device=device)
        self.table = torch.as_tensor(table, dtype=torch.long, device=device)
        self.counts = torch.as_tensor(counts, dtype=torch.long, device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed + 104729)
        self.coverage = dict(positive_count=len(positive), eligible_count=len(eligible),
                             negative_count=len(negative), mode=mode)

    def sample(self, logits):
        n = len(self.anchors)
        u = torch.rand((n, 8), device=logits.device, generator=self.generator)
        slots = (u * self.counts[self.drugs, None]).long()
        candidates = self.table[self.drugs[:, None], slots]
        if self.mode == "uniform":
            return candidates[:, :2]
        order = logits.detach()[candidates].argsort(dim=1, descending=True)
        middle_rank = torch.randint(2, 6, (n, 1), device=logits.device, generator=self.generator)
        middle = candidates.gather(1, order.gather(1, middle_rank))
        return torch.cat((candidates[:, :1], middle), dim=1)

    def loss(self, logits):
        negative = self.sample(logits)
        return F.softplus(logits[negative] - logits[self.anchors, None]).mean()


def group_metrics(pairs, labels, probability):
    rows = []
    for drug in np.unique(pairs[:, 1]):
        mask = pairs[:, 1] == drug
        y, p = labels[mask], probability[mask]
        positive = int(y.sum())
        negative = len(y) - positive
        if positive == 0 or negative == 0:
            continue
        rows.append(dict(drug=int(drug), n=len(y), positive=positive, negative=negative,
                         auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p))))
    if not rows:
        return dict(eligible_drugs=0, per_drug=[])
    return dict(eligible_drugs=len(rows), macro_drug_auc=float(np.mean([r["auc"] for r in rows])),
                macro_drug_ap=float(np.mean([r["ap"] for r in rows])),
                within_drug_auc=float(np.average([r["auc"] for r in rows],
                                      weights=[r["positive"] * r["negative"] for r in rows])),
                per_drug=rows)


def rank_scale(epoch, weight):
    return weight * min(1.0, max(0.0, (epoch - RECIPE["warmup_epochs"]) / RECIPE["ramp_epochs"]))


def train(config, seed, device, output):
    seed_all(seed)
    data = load_fold("MiDrug", 0)
    adapter = make_adapter(data, device)
    cfg = CONFIGS[config]
    ranking = SameDrugRanking(data["train_pairs"], data["train_y"], device, seed, cfg["sampler"])
    optimizer = torch.optim.RMSprop(adapter.model.parameters(), lr=RECIPE["lr"], alpha=0.99)
    tag = f"{config}_s{seed}"
    directory = output / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / f"{tag}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    curves, best_auc, best_state, best_epoch = [], -np.inf, None, None
    started = time.time()
    for epoch in range(1, RECIPE["epochs"] + 1):
        adapter.model.train()
        optimizer.zero_grad()
        logits = adapter.logits("train")
        bce = F.binary_cross_entropy_with_logits(logits, adapter.y)
        scale = rank_scale(epoch, cfg["rank_weight"])
        bpr = ranking.loss(logits) if scale > 0 else logits.new_zeros(())
        loss = bce + scale * bpr
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at {tag} epoch {epoch}")
        loss.backward()
        optimizer.step()
        adapter.model.eval()
        with torch.no_grad():
            p = adapter.predict("val")
        auc = float(roc_auc_score(data["val_y"], p))
        ap = float(average_precision_score(data["val_y"], p))
        curves.append(dict(epoch=epoch, loss=float(loss.detach()), bce=float(bce.detach()),
                           bpr=float(bpr.detach()), rank_weight=scale, val_auc=auc, val_ap=ap))
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0:
            print(tag, curves[-1], flush=True)
    adapter.model.load_state_dict(best_state)
    adapter.model.eval()
    with torch.no_grad():
        p = adapter.predict("val")
    checkpoint = directory / f"{tag}.pt"
    torch.save(best_state, checkpoint)
    predictions = directory / f"{tag}_predictions.npz"
    np.savez_compressed(predictions, pairs=data["val_pairs"], y=data["val_y"], probability=p)
    (directory / f"{tag}_curves.json").write_text(json.dumps(curves, indent=2))
    result = dict(config=config, seed=seed, signature=signature(), recipe=RECIPE, ranking=cfg,
                  sampler_coverage=ranking.coverage, best_epoch=best_epoch, epochs=RECIPE["epochs"],
                  selection="validation_auc", metrics=metrics(data["val_y"], p),
                  group_metrics=group_metrics(data["val_pairs"], data["val_y"], p),
                  graph_nodes=list(adapter.graph.node_types),
                  graph_edges={"|".join(e): adapter.graph[e].edge_index.shape[1] for e in adapter.graph.edge_types},
                  checkpoint_sha256=sha256(checkpoint), predictions_sha256=sha256(predictions),
                  elapsed_seconds=time.time() - started, test_evaluated=False)
    result_path.write_text(json.dumps(result, indent=2))
    print("DONE", tag, result["metrics"], flush=True)


def evaluate(config, seed, device, output):
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["signature"] == signature(), "Source/config changed after selection"
    assert config in ("bce", selection["best_bpr"]), "Unselected config cannot evaluate test"
    tag = f"{config}_s{seed}"
    val_file = output / "validation" / f"{tag}.json"
    assert sha256(val_file) == selection["validation_result_hashes"][tag]
    result = json.loads(val_file.read_text())
    checkpoint = val_file.with_suffix(".pt")
    assert sha256(checkpoint) == result["checkpoint_sha256"]
    test_dir = output / "test"
    test_dir.mkdir(exist_ok=True)
    if (test_dir / f"{tag}.json").exists():
        raise FileExistsError(tag)
    seed_all(seed)
    data = load_fold("MiDrug", 0)
    adapter = make_adapter(data, device, include_test=True)
    adapter.model.load_state_dict(torch.load(checkpoint, map_location=device))
    adapter.model.eval()
    with torch.no_grad():
        val_probability = adapter.predict("val")
        p = adapter.predict("test")
    recorded_val = np.load(val_file.with_name(f"{tag}_predictions.npz"))["probability"]
    np.testing.assert_allclose(val_probability, recorded_val, rtol=2e-5, atol=2e-5)
    np.savez_compressed(test_dir / f"{tag}_predictions.npz", pairs=data["test_pairs"],
                        y=data["test_y"], probability=p)
    test = dict(config=config, seed=seed, signature=signature(), best_epoch=result["best_epoch"],
                checkpoint_sha256=result["checkpoint_sha256"], selection_sha256=sha256(selection_path),
                metrics=metrics(data["test_y"], p),
                group_metrics=group_metrics(data["test_pairs"], data["test_y"], p))
    (test_dir / f"{tag}.json").write_text(json.dumps(test, indent=2))
    print("TEST", tag, test["metrics"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "test"), default="train")
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    (train if args.phase == "train" else evaluate)(args.config, args.seed, args.device, args.output)


if __name__ == "__main__":
    main()
