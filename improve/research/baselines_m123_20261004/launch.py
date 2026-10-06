"""Run the fixed, predeclared 13 training jobs on two GPUs; resume per job."""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

HERE=Path(__file__).resolve().parent

def main():
    jobs=Queue()
    for ds in ('MiDrug','MDR','MDS'):
        runs=[(0,s) for s in (42,43,44)] if ds=='MiDrug' else [(f,42) for f in range(5)]
        for fold,seed in runs:jobs.put((ds,fold,seed))
    (HERE/'logs').mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    def worker(device):
        failures=[]
        while True:
            try:ds,fold,seed=jobs.get_nowait()
            except Empty:break
            tag=f'incrd_m123_{ds}_f{fold}_s{seed}'
            if (HERE/f'results/{tag}.json').exists():
                print('RESUME',tag,flush=True);continue
            cmd=[sys.executable,str(HERE/'run.py'),'--dataset',ds,'--fold',str(fold),
                 '--seed',str(seed),'--device',device]
            print('START',tag,device,flush=True)
            with (HERE/f'logs/{tag}.log').open('w') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            print('END',tag,result.returncode,flush=True)
            if result.returncode:failures.append(tag)
        return failures
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,device) for device in ('cuda:0','cuda:1')]
        failures=sum([f.result() for f in futures],[])
    (HERE/'failures.json').write_text(json.dumps(failures))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
