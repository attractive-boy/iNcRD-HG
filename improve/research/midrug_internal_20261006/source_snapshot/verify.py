"""Recompute validation predictions and audit trained query deletion."""
import copy
import json
import numpy as np
import torch
from fusion import HERE, CONFIGS, InteractionModel, load_fold, make_adapter, seed_all, sha256
from graphdrop import context
from fullgraph_drop import make_adapter as make_full_adapter
from check_modules import check
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    rows = []
    for family in ("fusion", "graphdrop", "fullgraph_drop"):
        for path in sorted((HERE / family).glob("*_s42.json")):
            record = json.loads(path.read_text())
            assert sha256(path.with_suffix(".pt")) == record["checkpoint_sha256"]
            assert sha256(HERE / f"{family}.py") == record["source_sha256"]
            seed_all(42)
            data = load_fold("MiDrug", 0)
            adapter = (make_full_adapter if family == "fullgraph_drop" else make_adapter)(data, "cuda:1")
            if family == "fusion":
                config = CONFIGS[path.stem.rsplit("_s", 1)[0]]
                adapter.model = InteractionModel(adapter.graph, adapter.ps, adapter.qis["train"].shape[1],
                    adapter.ease["train"].shape[1], adapter.cfg, config).to("cuda:1")
            adapter.model.load_state_dict(torch.load(path.with_suffix(".pt"), map_location="cuda:1"))
            adapter.model.eval()
            with torch.no_grad():
                prediction = adapter.predict("val")
            saved = np.load(path.with_name(path.stem + "_val.npz"))
            np.testing.assert_array_equal(data["val_pairs"], saved["pairs"])
            np.testing.assert_array_equal(data["val_y"], saved["y"])
            np.testing.assert_allclose(prediction, saved["probability"], atol=2e-5, rtol=2e-5)
            auc = roc_auc_score(saved["y"], saved["probability"])
            ap = average_precision_score(saved["y"], saved["probability"])
            assert abs(auc-record["val_auc"]) < 1e-12 and abs(ap-record["val_ap"]) < 1e-12
            curves = json.loads(path.with_name(path.stem + "_curves.json").read_text())
            best = max(curves, key=lambda r:r["val_auc"])
            assert best["epoch"] == record["best_epoch"] and best["val_auc"] == record["val_auc"]
            checks = check(adapter, count=6)
            if family in ("graphdrop", "fullgraph_drop"):
                # Audit a sparse training context as well as the evaluation graph.
                edges = adapter.graph["miRNA", "MiDrug", "drug"].edge_index[:, ::2]
                c = context(adapter, edges)
                other = copy.copy(adapter)
                other.graph, other.ps, other.dense = c["graph"], c["ps"], c["dense"]
                other.model.encoder.graph, other.model.encoder.adj = c["graph"], c["adj"]
                checks += check(other, count=6)
            row = dict(family=family, tag=path.stem, status="passed", explicit_deletion=checks,
                       validation_prediction_sha256=sha256(path.with_name(path.stem + "_val.npz")))
            rows.append(row)
            print("PASS", family, path.stem, flush=True)
            del adapter
            torch.cuda.empty_cache()
    assert len(rows) == 14, "Expected 8 fusion, 4 graphdrop, 2 original-graph runs"
    (HERE / "verification.json").write_text(json.dumps(dict(status="passed", test_evaluated=False, records=rows), indent=2))


if __name__ == "__main__":
    main()
