"""Recompute all ten PLMF-MDA reference folds with the unchanged release."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[2] / "release/iNcRD-HG-M1M2M3"
sys.path.insert(0, str(RELEASE))
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.data import build_graph, load_fold, sha256, split_path
from incrd_hg_m123.engine import seed_all, metrics


def main():
    refs = HERE.parent / "baselines_m123_20261004/results"
    records = []
    for dataset in ("MDR", "MDS"):
        for fold in range(5):
            seed_all(42)
            tag = f"incrd_m123_{dataset}_f{fold}_s42"
            reference = json.loads((refs / f"{tag}.json").read_text())
            checkpoint = refs / f"{tag}.pt"
            assert sha256(checkpoint) == reference["checkpoint_sha256"]
            assert sha256(split_path(dataset, fold)) == reference["split_sha256"]
            data = load_fold(dataset, fold)
            adapter = Adapter(data, build_graph(data, dataset), "cuda:1")
            adapter.model.load_state_dict(torch.load(checkpoint, map_location="cuda:1"))
            adapter.model.eval()
            with torch.no_grad():
                p = adapter.predict("test")
            saved = np.load(refs / f"{tag}_predictions.npz")
            np.testing.assert_array_equal(data["test_pairs"], saved["test_pairs"])
            np.testing.assert_array_equal(data["test_y"], saved["test_y"])
            np.testing.assert_allclose(p, saved["test_probability"], rtol=3e-5, atol=3e-5)
            result = metrics(data["test_y"], p)
            # Tiny numerical score perturbations must not change reported AUC/AP.
            assert abs(result["auc"] - reference["auc"]) < 1e-6
            assert abs(result["aupr"] - reference["aupr"]) < 1e-6
            row = dict(dataset=dataset, fold=fold, auc=result["auc"], ap=result["aupr"],
                       max_prediction_delta=float(np.max(np.abs(p-saved["test_probability"]))),
                       checkpoint_sha256=sha256(checkpoint), split_sha256=sha256(split_path(dataset, fold)))
            records.append(row)
            print(json.dumps(row), flush=True)
            del adapter
            torch.cuda.empty_cache()
    result = dict(status="passed", scope="Ten reference checkpoint evaluations; unchanged release route",
                  records=records, source_hashes={str(p.relative_to(RELEASE)): sha256(p)
                  for p in sorted((RELEASE / "incrd_hg_m123").glob("*.py"))})
    (HERE / "plmf_protection.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
