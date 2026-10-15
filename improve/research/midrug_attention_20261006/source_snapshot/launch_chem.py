"""Fixed validation-only graph-construction grid."""
import json
import subprocess
import sys
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from model import HERE,sha256


def main():
    out=HERE/"chemical_results";out.mkdir(exist_ok=True)
    configs=[("original",3)]+[(name,k) for name in ("morgan2","morgan3","maccs","pharm2") for k in (3,10)]
    plan=dict(configs=configs,source_sha256=sha256(HERE/"train_chem.py"),chemical_sha256=sha256(HERE/"chemical_kernels.npz"),
        selection="min(validation AUC gain, validation AP gain)",test_evaluated=False)
    p=out/"plan.json"
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(plan))
    else:p.write_text(json.dumps(plan,indent=2))
    queue=Queue()
    for name,k in configs:
        f=out/f"{name}_k{k}.json"
        if f.exists():
            r=json.loads(f.read_text());assert r["source_sha256"]==plan["source_sha256"] and r["checkpoint_sha256"]==sha256(f.with_suffix(".pt"))
        else:queue.put((name,k))
    def worker(device):
        while True:
            try:name,k=queue.get_nowait()
            except Empty:return
            with (out/f"{name}_k{k}.log").open("w") as log:
                subprocess.run([sys.executable,str(HERE/"train_chem.py"),"--kernel",name,"--k",str(k),"--device",device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print("DONE",name,k,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for f in [pool.submit(worker,d) for d in ("cuda:0","cuda:1")]:f.result()


if __name__=="__main__":main()
