"""Predeclare, run validation grid, freeze selection, then evaluate two configs."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import numpy as np

from run_ranking import CONFIGS, HERE, RECIPE, SEEDS, sha256, signature, source_hashes


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    plan = dict(signature=signature(), configs=CONFIGS, seeds=SEEDS, recipe=RECIPE,
                source_hashes=source_hashes(), data="frozen MiDrug fold0, unchanged pairs and labels",
                graph="miRNA/drug only; original features and same-type edges; train-positive target edges",
                selection="Best BPR by mean validation AUC across three seeds; tie: mean validation AP",
                test="Evaluate baseline and best BPR only after selection is frozen; report failure too",
                adoption="Require test mean AUC and AP to improve; test does not select another candidate",
                negative_pool="Existing training negatives only; same drug as the positive anchor",
                constraints=["No lncRNA", "No external data", "No test metrics during search"])
    if path.exists():
        assert json.loads(path.read_text())["signature"] == plan["signature"]
    else:
        path.write_text(json.dumps(plan, indent=2))
        for relative in plan["source_hashes"]:
            src = HERE.parents[2] / relative
            dst = output / "source_snapshot" / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for src in HERE.glob("*.py"):
            shutil.copy2(src, output / "source_snapshot" / src.name)


def run_jobs(output, configs, phase):
    jobs = Queue()
    for config in configs:
        for seed in SEEDS:
            jobs.put((config, seed))
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")

    def worker(device):
        while True:
            try:
                config, seed = jobs.get_nowait()
            except Empty:
                return
            tag = f"{config}_s{seed}"
            result = output / ("validation" if phase == "train" else "test") / f"{tag}.json"
            if result.exists():
                old = json.loads(result.read_text())
                assert old["signature"] == signature()
                if phase == "train":
                    assert sha256(result.with_suffix(".pt")) == old["checkpoint_sha256"]
                print("RESUME", phase, tag, flush=True)
                continue
            cmd = [sys.executable, str(HERE / "run_ranking.py"), "--phase", phase,
                   "--config", config, "--seed", str(seed), "--device", device, "--output", str(output)]
            print("START", phase, tag, device, flush=True)
            with (logs / f"{phase}_{tag}.log").open("w") as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads(result.read_text())
            print("DONE", phase, tag, summary["metrics"], flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, device) for device in ("cuda:0", "cuda:1")]
        for future in futures:
            future.result()


def select(output):
    path = output / "selection.json"
    if path.exists():
        selection = json.loads(path.read_text())
        assert selection["signature"] == signature()
        return selection
    summary, hashes = {}, {}
    for config in CONFIGS:
        records = []
        for seed in SEEDS:
            tag = f"{config}_s{seed}"
            file = output / "validation" / f"{tag}.json"
            record = json.loads(file.read_text())
            assert record["signature"] == signature() and not record["test_evaluated"]
            assert sha256(file.with_suffix(".pt")) == record["checkpoint_sha256"]
            assert sha256(file.with_name(f"{tag}_predictions.npz")) == record["predictions_sha256"]
            hashes[tag] = sha256(file)
            records.append(record)
        summary[config] = {k: float(np.mean([r["metrics"][k] for r in records])) for k in ("auc", "aupr")}
        summary[config]["best_epochs"] = [r["best_epoch"] for r in records]
    best_bpr = max((k for k in CONFIGS if k != "bce"),
                   key=lambda k: (summary[k]["auc"], summary[k]["aupr"]))
    selection = dict(signature=signature(), best_bpr=best_bpr, validation_summary=summary,
                     validation_result_hashes=hashes, test_results_used=False)
    path.write_text(json.dumps(selection, indent=2))
    print("FROZEN SELECTION", json.dumps(selection["validation_summary"]), "BEST", best_bpr, flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "train", "select", "test"), required=True)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    prepare(args.output)
    if args.phase == "train":
        run_jobs(args.output, CONFIGS, "train")
    elif args.phase == "select":
        select(args.output)
    elif args.phase == "test":
        selection = json.loads((args.output / "selection.json").read_text())
        assert selection["signature"] == signature()
        run_jobs(args.output, ("bce", selection["best_bpr"]), "test")


if __name__ == "__main__":
    main()
