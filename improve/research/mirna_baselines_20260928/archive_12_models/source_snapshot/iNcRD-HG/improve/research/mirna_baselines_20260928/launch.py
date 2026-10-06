"""Resumable subprocess scheduler; each result has its own checkpoint and log."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from bench_data import HERE
from run import DEFAULTS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models',default=','.join(DEFAULTS));p.add_argument('--devices',default='cuda:0,cuda:1')
    p.add_argument('--smoke',action='store_true');p.add_argument('--datasets',default='MiDrug,MDR,MDS')
    p.add_argument('--output',default='results')
    p.add_argument('--folds',default='0,1,2,3,4')
    p.add_argument('--compact-subknet',action='store_true')
    p.add_argument('--exclude-tags',default='')
    a=p.parse_args(); devices=a.devices.split(','); jobs=[]
    if a.smoke and a.output=='results':a.output='smoke'
    for model in a.models.split(','):
        for ds in a.datasets.split(','):
            runs=[(0,42)] if a.smoke else ([(0,s) for s in (42,43,44)] if ds=='MiDrug' else [(int(f),42) for f in a.folds.split(',')])
            for fold,seed in runs:
                if f'{model}_{ds}_f{fold}_s{seed}' not in a.exclude_tags.split(','):
                    jobs.append((model,ds,fold,seed))
    jobs=iter(jobs); failures=[]
    def worker(dev):
        done=[]
        for model,ds,fold,seed in jobs:
            tag=f'{model}_{ds}_f{fold}_s{seed}'
            result=HERE/a.output/f'{tag}.json'
            if result.exists():
                print('RESUME skip',tag,flush=True);continue
            cmd=[sys.executable,str(HERE/'run.py'),'--model',model,'--dataset',ds,'--fold',str(fold),'--seed',str(seed),'--device',dev,'--output',a.output]
            if a.smoke:cmd+=['--epochs','2']
            if a.compact_subknet and model=='subknet':cmd+=['--compact-subknet']
            env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            log=HERE/'logs'/f'{a.output.replace("/","_")}_{tag}.log'
            if log.exists() and not result.exists():
                log=log.with_name(log.stem+f'_worker{os.getpid()}'+log.suffix)
            print('START',tag,dev,flush=True)
            with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
            print('END',tag,'exit',r.returncode,flush=True)
            if r.returncode:done.append({'job':tag,'exit_code':r.returncode,'log':str(log)})
        return done
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for task in as_completed([pool.submit(worker,d) for d in devices]):failures+=task.result()
    (HERE/f'{a.output.replace("/","_")}_failures.json').write_text(json.dumps(failures,indent=2))
    print('FINISHED failures',len(failures),flush=True)
    if failures:sys.exit(1)

if __name__=='__main__':main()
