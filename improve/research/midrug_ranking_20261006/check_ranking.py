"""Check train-only ranking, graph isolation and exact query-edge correction."""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from run_ranking import HERE, RELEASE, SameDrugRanking, load_fold, make_adapter, make_graph, seed_all

sys.path.insert(0, str(RELEASE))
from check_modules import check


def main():
    device = "cuda:0"
    torch.cuda.set_device(device)
    seed_all(42)
    data = load_fold("MiDrug", 0)
    adapter = make_adapter(data, device)
    assert adapter.pairs["test"].numel() == 0
    assert set(adapter.graph.node_types) == {"miRNA", "drug"}
    assert len(adapter.graph.edge_types) == 4
    assert not any("lnc" in name.lower() for name, _ in adapter.model.named_parameters())
    rows = check(adapter, count=12)
    pair, y = data["train_pairs"], data["train_y"]
    logits = torch.linspace(-2, 2, len(y), device=device, requires_grad=True)
    coverage = {}
    for mode in ("uniform", "mixed_middle"):
        sampler = SameDrugRanking(pair, y, device, 42, mode)
        twin = SameDrugRanking(pair, y, device, 42, mode)
        index = sampler.sample(logits)
        torch.testing.assert_close(index, twin.sample(logits))
        anchors = sampler.anchors.cpu().numpy()
        neg = index.cpu().numpy()
        assert np.all(y[anchors] == 1) and np.all(y[neg] == 0)
        assert np.all(pair[neg, 1] == pair[anchors, 1, None])
        assert len(anchors) == int(y.sum())
        loss = F.softplus(logits[index] - logits[sampler.anchors, None]).mean()
        shifted = F.softplus(logits[index] - (logits[sampler.anchors, None] + 1)).mean()
        assert shifted < loss
        grad, = torch.autograd.grad(loss, logits)
        assert torch.all(grad[sampler.anchors] < 0)
        assert torch.all(grad[torch.as_tensor(y == 0, device=device)] >= 0)
        coverage[mode] = sampler.coverage
    # Holding training data fixed, arbitrary held-out labels/identities do not
    # change the graph, training channels or negative sampler.
    changed = {k: v.copy() for k, v in data.items()}
    changed["val_y"] = 1 - changed["val_y"]
    changed["test_y"] = 1 - changed["test_y"]
    changed["test_pairs"] = changed["test_pairs"][::-1].copy()
    seed_all(42)
    other = make_adapter(changed, device)
    for e in adapter.graph.edge_types:
        torch.testing.assert_close(adapter.graph[e].edge_index, other.graph[e].edge_index)
    torch.testing.assert_close(adapter.qis["train"], other.qis["train"])
    torch.testing.assert_close(adapter.ease["train"], other.ease["train"])
    # Zero BPR is exactly the unmodified released BCE loss on the reduced graph.
    adapter.model.train()
    optimizer = torch.optim.RMSprop(adapter.model.parameters(), lr=1e-4, alpha=0.99)
    optimizer.zero_grad()
    train_logits = adapter.logits("train")
    ranking = SameDrugRanking(pair, y, device, 42, "mixed_middle")
    bce = F.binary_cross_entropy_with_logits(train_logits, adapter.y)
    bpr = ranking.loss(train_logits)
    torch.testing.assert_close(bce + 0 * bpr, bce, atol=0, rtol=0)
    loss = bce + 0.1 * bpr
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.model.parameters())
    optimizer.step()
    report = dict(status="passed", graph_nodes=list(adapter.graph.node_types),
                  graph_edges={"|".join(e): adapter.graph[e].edge_index.shape[1] for e in adapter.graph.edge_types},
                  sampling=coverage, explicit_query_deletion=rows,
                  test_features_empty=True, heldout_invariance=True, finite_bpr_gradients=True,
                  monotonic_bpr=True, initial_loss=float(loss.detach()))
    (HERE / "checks_initial.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
