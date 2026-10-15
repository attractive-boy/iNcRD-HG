"""Run the six declared dual-metric objective recipes."""
import json
import subprocess
import sys
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from dual_objective import HERE,OBJECTIVES,sha256


def main():
    out=HERE/"dual_objective"
    out.mkdir(exist_ok=True)
    plan=dict(objectives=OBJECTIVES,seeds=[42],source_sha256=sha256(HERE/"dual_objective.py"),
              selection="min(validation AUC gain, validation AP gain)",test_evaluated=False)
    path=out/"plan.json"
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    queue=Queue()
    for key in OBJECTIVES:
        f=out/f"{key}_s42.json"
        if f.exists():
            r=json.loads(f.read_text())
            assert r["source_sha256"]==plan["source_sha256"] and r["checkpoint_sha256"]==sha256(f.with_suffix(".pt"))
        else:queue.put(key)
    def worker(device):
        while True:
            try:key=queue.get_nowait()
            except Empty:return
            with (out/f"{key}.log").open("w") as log:
                subprocess.run([sys.executable,str(HERE/"dual_objective.py"),"--config",key,"--device",device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print("DONE",key,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for f in [pool.submit(worker,d) for d in ("cuda:0","cuda:1")]:f.result()


if __name__=="__main__":main()
