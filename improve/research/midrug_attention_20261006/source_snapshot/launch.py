"""Run the eight declared attention recipes with validation-only selection."""
import json
import subprocess
import sys
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from model import HERE,CONFIGS,sha256


def main():
    assert json.loads((HERE/"initial_checks.json").read_text())["status"]=="passed"
    out=HERE/"results";out.mkdir(exist_ok=True)
    plan=dict(configs=CONFIGS,seeds=[42],max_epochs=700,patience=200,minimum_epochs=300,
              source_sha256=sha256(HERE/"train.py"),model_sha256=sha256(HERE/"model.py"),
              graph="Frozen original lncRNA/miRNA/drug graph; target train positives only",
              selection="min(validation AUC gain, validation AP gain)",test_evaluated=False)
    p=out/"plan.json"
    if p.exists():assert json.loads(p.read_text())==plan
    else:p.write_text(json.dumps(plan,indent=2))
    queue=Queue()
    for key in CONFIGS:
        path=out/f"{key}_s42.json"
        if path.exists():
            r=json.loads(path.read_text())
            assert r["source_sha256"]==plan["source_sha256"] and r["model_sha256"]==plan["model_sha256"]
            assert r["checkpoint_sha256"]==sha256(path.with_suffix(".pt"))
        else:queue.put(key)
    def worker(device):
        while True:
            try:key=queue.get_nowait()
            except Empty:return
            with (out/f"{key}.log").open("w") as log:
                subprocess.run([sys.executable,str(HERE/"train.py"),"--config",key,"--device",device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print("DONE",key,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for f in [pool.submit(worker,d) for d in ("cuda:0","cuda:1")]:f.result()


if __name__=="__main__":main()
