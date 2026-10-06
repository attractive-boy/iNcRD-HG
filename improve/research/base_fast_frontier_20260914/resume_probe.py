"""Resume only missing registered runs without modifying frozen probe code."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fast_probe as probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", choices=tuple(probe.TASKS), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = Path(args.out)
    protocol = json.loads((out / "protocol.json").read_text())
    for path, sha in protocol["source_files"].items():
        assert probe.digest(probe.REPO / path) == sha, "Frozen source changed"
    for path, sha in protocol["cached_inputs"].items():
        assert probe.digest(out / path) == sha, "Frozen input changed"
    graph = torch.load(out / args.task / "train_graph.pth", map_location="cpu")
    with np.load(out / args.task / "train_val_inputs.npz", allow_pickle=False) as z:
        assert not any("test" in key for key in z.files)
        arrays = {key: z[key] for key in z.files}
    completed, resumed = [], []
    for seed in protocol["seeds"]:
        raw_hash = None
        for arm in protocol["arms"]:
            dest = out / "runs" / f"{args.task}_{arm}_s{seed}"
            if (dest / "result.json").exists():
                row = json.loads((dest / "result.json").read_text())
                for name, sha in row["artifacts_sha256"].items():
                    assert probe.digest(dest / name) == sha
                completed.append(dest.name)
            else:
                if dest.exists():
                    archive = out / "interrupted_runs"
                    archive.mkdir(exist_ok=True)
                    assert not (archive / dest.name).exists()
                    dest.rename(archive / dest.name)
                dest.mkdir()
                row, pred, trace, state = probe.fit_one(
                    args.task, arm, seed, graph, arrays, args.device,
                    protocol["task_settings"][args.task]["train_epochs"])
                pd.DataFrame(trace).to_csv(dest / "trace.csv", index=False)
                np.savez_compressed(dest / "val_scores.npz", pred=pred,
                                    y=arrays["y_val"], pairs=arrays["pairs_val"])
                torch.save(state, dest / "checkpoint.pt")
                row["artifacts_sha256"] = {
                    p.name: probe.digest(p) for p in dest.iterdir() if p.is_file()}
                probe.write_json(dest / "result.json", row)
                resumed.append(dest.name)
                print(json.dumps({k: row[k] for k in
                    ("task", "arm", "seed", "val_auc", "val_ap", "elapsed_seconds")}), flush=True)
            if arm == "base":
                raw_hash = row["initial_state_sha256"]
            if arm != "pca64":
                assert row["initial_state_sha256"] == raw_hash
    probe.write_json(out / f"RESUMED_{args.task}.json", dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        wrapper_sha256=probe.digest(__file__), completed_before=completed,
        newly_completed=resumed, original_protocol_sha256=probe.digest(out / "protocol.json")))


if __name__ == "__main__":
    main()
