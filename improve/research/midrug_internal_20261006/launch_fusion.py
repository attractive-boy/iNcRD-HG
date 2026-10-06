"""Run a fixed validation-only grid, resumable by verified completed results."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue, Empty
from fusion import CONFIGS, HERE, sha256


def main():
    queue = Queue()
    out = HERE / "fusion"
    out.mkdir(exist_ok=True)
    source_hash = sha256(HERE / "fusion.py")
    plan = dict(configs=CONFIGS, seeds=[42], selection="validation AUC only",
                test_evaluated=False, source_sha256=source_hash)
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    for config in CONFIGS:
        # Two pilot tasks were launched before this batch. Their completion
        # is checked by the caller before the batch is launched.
        result = out / f"{config}_s42.json"
        if result.exists():
            rec = json.loads(result.read_text())
            assert rec["source_sha256"] == source_hash
            assert rec["checkpoint_sha256"] == sha256(result.with_suffix(".pt"))
            continue
        queue.put(config)
    def worker(device):
        while True:
            try:
                config = queue.get_nowait()
            except Empty:
                return
            cmd = [sys.executable, str(HERE / "fusion.py"), "--config", config, "--device", device]
            with (HERE / f"fusion_{config}.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            print("DONE", config, device, flush=True)
    with ThreadPoolExecutor(2) as pool:
        for f in [pool.submit(worker, d) for d in ("cuda:0", "cuda:1")]:
            f.result()


if __name__ == "__main__":
    main()
