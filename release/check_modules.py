#!/usr/bin/env python
"""Explicit edge-deletion equivalence checks for M1 / M2 / M3.

For sampled training positives the query-hollow computation is compared with an
explicit edge deletion on a cloned graph:

* endpoint states from ``forward_dense`` equal the PyG encoder on the deleted
  graph (M1 QMP),
* pooled intermediate states (M2), pair statistics and EASE scores (M3) match,
* the fused logit matches.

The held-out labels are then permuted; the message graph and all pair channels
must stay unchanged.  The checks are run before any training step.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.data import build_graph, load_fold, structure_data
from incrd_hg_m123.engine import seed_all
from incrd_hg_m123.pair_channels import PairStruct


def check(adapter, count=6):
    model = adapter.model.eval()
    enc = model.encoder
    graph = adapter.graph
    rows = []
    with torch.no_grad():
        original = enc(graph.x_dict, graph.edge_index_dict)
        zr, zd = enc.forward_dense(*adapter.dense)
        for node, actual in (("miRNA", zr), ("drug", zd)):
            torch.testing.assert_close(actual, original[node], atol=3e-4, rtol=2e-5)
        pos = graph["miRNA", "MiDrug", "drug"].edge_index.t()
        indices = np.linspace(0, len(pos) - 1, count, dtype=int)
        for index in indices:
            pair = pos[index:index + 1].t()
            r, d = map(int, pair[:, 0])
            q = adapter.ps.A[r, d].reshape(1)
            z = enc.forward_dense(*adapter.dense, *pair, q)
            f1 = model(pair, adapter.ps, (adapter.ps.qis(*pair) - adapter.mu) / adapter.sd,
                       adapter.ps.ease(*pair), adapter.dense)
            g2 = graph.clone()
            for edge, first, second in ((("miRNA", "MiDrug", "drug"), r, d),
                                        (("drug", "rev_MiDrug", "miRNA"), d, r)):
                idx = g2[edge].edge_index
                g2[edge].edge_index = idx[:, ~((idx[0] == first) & (idx[1] == second))]
            d2 = structure_data(g2)
            ps2 = PairStruct(d2, g2["miRNA", "MiDrug", "drug"].edge_index.t(), adapter.cfg)
            z2 = enc(g2.x_dict, g2.edge_index_dict)
            torch.testing.assert_close(z[2][0], z2["miRNA"][r], atol=3e-4, rtol=2e-5)
            torch.testing.assert_close(z[3][0], z2["drug"][d], atol=3e-4, rtol=2e-5)
            pools1 = adapter.ps.qncn(*pair, {"miRNA": z[0], "drug": z[1]}, z[4])
            pools2 = ps2.qncn(*pair, z2)
            for x, y in zip(pools1, pools2):
                torch.testing.assert_close(x, y, atol=3e-4, rtol=2e-5)
            torch.testing.assert_close(adapter.ps.qis(*pair), ps2.qis(*pair), atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(adapter.ps.ease(*pair), ps2.ease(*pair), atol=3e-4, rtol=3e-4)
            old_graph, old_adj = enc.graph, enc.adj
            enc.graph = g2
            enc.adj = dict(old_adj)
            for edge in (("miRNA", "MiDrug", "drug"), ("drug", "rev_MiDrug", "miRNA")):
                idx = g2[edge].edge_index
                enc.adj[edge] = torch.sparse_coo_tensor(idx.flip(0), torch.ones(idx.shape[1], device=idx.device),
                                                        old_adj[edge].shape).coalesce()
            try:
                f2 = model(pair, ps2, (ps2.qis(*pair) - adapter.mu) / adapter.sd, ps2.ease(*pair),
                           (d2.xr, d2.xd, d2.Sr1, d2.Sd1, ps2.A))
            finally:
                enc.graph, enc.adj = old_graph, old_adj
            torch.testing.assert_close(f1, f2, atol=3e-4, rtol=3e-4)
            rows.append({"m": r, "d": d,
                         "logit_abs_error": float((f1 - f2).abs().max()),
                         "encoder_abs_error": max(float((z[2][0] - z2["miRNA"][r]).abs().max()),
                                                  float((z[3][0] - z2["drug"][d]).abs().max()))})
    return rows


def label_invariance(adapter, data, dataset, device):
    changed = dict(data)
    for part in ("val", "test"):
        changed[part + "_y"] = 1 - data[part + "_y"]
    seed_all(42)
    other = Adapter(changed, build_graph(changed, dataset), device)
    for e in adapter.graph.edge_types:
        torch.testing.assert_close(adapter.graph[e].edge_index, other.graph[e].edge_index)
    for part in ("train", "val", "test"):
        torch.testing.assert_close(adapter.qis[part], other.qis[part])
        torch.testing.assert_close(adapter.ease[part], other.ease[part])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="MiDrug,MDR,MDS")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--output", default="")
    args = p.parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    reports = {}
    for ds in [d for d in args.datasets.split(",") if d]:
        seed_all(42)
        data = load_fold(ds, 0)
        adapter = Adapter(data, build_graph(data, ds), args.device)
        rows = check(adapter, args.count)
        label_invariance(adapter, data, ds, args.device)
        optimizer = torch.optim.RMSprop(adapter.model.parameters(), lr=1e-4)
        adapter.model.train()
        loss = adapter.train_epoch(optimizer, 1)
        assert np.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.model.parameters())
        reports[ds] = {"relations": len(adapter.graph.edge_types), "n_queries": len(rows), "checks": rows}
        print(f"PASS {ds}: dense encoder == PyG, explicit deletion == query-hollow "
              f"for encoder/M2/M3/fused logits, label invariance, finite gradients", flush=True)
        del adapter
    if args.output:
        Path(args.output).write_text(json.dumps(reports, indent=2))
        print("wrote " + args.output, flush=True)


if __name__ == "__main__":
    main()
