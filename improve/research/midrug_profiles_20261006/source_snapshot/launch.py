"""Frozen eight-configuration interaction-profile grid."""
import json
import subprocess
import sys
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from model import HERE,CONFIGS,sha256


def main():
    out=HERE/'results';out.mkdir(exist_ok=True)
    plan=dict(configs=CONFIGS,seed=42,source_sha256=sha256(HERE/'train.py'),model_sha256=sha256(HERE/'model.py'),
        selection='min(validation AUC gain, validation AP gain)',test_evaluated=False)
    p=out/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:p.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for name in CONFIGS:
        r=out/f'{name}_s42.json'
        if r.exists():
            rec=json.loads(r.read_text())
            assert rec['source_sha256']==plan['source_sha256'] and rec['model_sha256']==plan['model_sha256']
            assert rec['checkpoint_sha256']==sha256(r.with_suffix('.pt'))
        else:q.put(name)
    def worker(device):
        while True:
            try:name=q.get_nowait()
            except Empty:return
            with (out/f'{name}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,device,flush=True)
    with ThreadPoolExecutor(2) as pool:
        for f in [pool.submit(worker,d) for d in ('cuda:0','cuda:1')]:f.result()

if __name__=='__main__':main()
