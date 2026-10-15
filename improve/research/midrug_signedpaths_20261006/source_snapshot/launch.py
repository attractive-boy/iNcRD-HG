"""Frozen, validation-only M2/M3 search, one process per GPU."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from train import HERE,CONFIGS,sha256


def main():
    out=HERE/"results"
    out.mkdir(exist_ok=True)
    plan=dict(configs=CONFIGS,seeds=[42],epochs=600,minimum_epochs=300,patience=200,
              graph="Original full lncRNA/miRNA/drug; train-positive target edges",
              selection="validation AUC",test_evaluated=False,
              train_sha256=sha256(HERE/"train.py"),modules_sha256=sha256(HERE/"modules.py"))
    plan_path=out/"plan.json"
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan
    else:plan_path.write_text(json.dumps(plan,indent=2))
    queue=Queue()
    for config in CONFIGS:
        result=out/f"{config}_s42.json"
        if result.exists():
            r=json.loads(result.read_text())
            assert r["source_sha256"]==plan["train_sha256"]
            assert r["checkpoint_sha256"]==sha256(result.with_suffix(".pt"))
        else:queue.put(config)
    def worker(device):
        while True:
            try:config=queue.get_nowait()
            except Empty:return
            with (out/f"{config}.log").open("w") as log:
                subprocess.run([sys.executable,str(HERE/"train.py"),"--config",config,"--device",device],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            print("DONE",config,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(worker,d) for d in ("cuda:0","cuda:1")]:future.result()


if __name__=="__main__":main()
