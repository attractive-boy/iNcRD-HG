"""Schedule independent frozen LinkPFN jobs; does not change model settings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--devices',default='cuda:0,cuda:1')
    parser.add_argument('--start',choices=('generate','pretrain','screen','confirm','evaluate','validate'),default='generate')
    args=parser.parse_args()
    out=Path(args.out_dir).resolve()
    repo=Path(__file__).resolve().parents[1]
    from improve import link_pfn as exp
    p=exp.protocol(out)
    devices=args.devices.split(',')
    logs=out/'logs'
    logs.mkdir(exist_ok=True)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        env[key]='4'
    events=[]

    def run(name,arguments,device):
        argv=[sys.executable,'-u','-m','improve.link_pfn',*arguments,'--out-dir',str(out),'--device',device]
        with (logs/(name+'.log')).open('a') as f:
            f.write('\n'+datetime.now(timezone.utc).isoformat()+'\n'+json.dumps(argv)+'\n')
            f.flush()
            result=subprocess.run(argv,cwd=str(repo),env=env,stdout=f,stderr=subprocess.STDOUT)
        event=dict(name=name,device=device,exit_code=result.returncode,time=datetime.now(timezone.utc).isoformat())
        print(json.dumps(event),flush=True)
        if result.returncode:
            print('\n'.join((logs/(name+'.log')).read_text().splitlines()[-30:]),flush=True)
            raise RuntimeError('Job failed: '+name)
        return event

    def parallel(jobs):
        queue=Queue()
        for job in jobs:
            queue.put(job)
        def worker(device):
            result=[]
            while True:
                try:
                    name,argv=queue.get_nowait()
                except Empty:
                    return result
                result.append(run(name,argv,device))
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures=[pool.submit(worker,device) for device in devices]
            for future in futures:
                events.extend(future.result())

    steps=[
        ('generate',[(prior,['generate','--arm',prior]) for prior in p['priors']]),
        ('features',[(task,['features','--task',task]) for task in p['tasks']]),
        ('pretrain',[(prior+'_s'+str(seed),['pretrain','--arm',prior,'--seed',str(seed)]) for prior in p['priors'] for seed in p['seeds']]),
        ('screen',[(task+'_'+arm,['tune','--stage','screen','--task',task,'--arm',arm]) for task in p['tasks'] for arm in p['arms']]),
        ('select_screen',[('all',['select','--stage','screen'])]),
        ('confirm',[(task+'_'+arm,['tune','--stage','confirm','--task',task,'--arm',arm]) for task in p['tasks'] for arm in p['arms']]),
        ('select_confirm',[('all',['select','--stage','confirm'])]),
        ('evaluate',[(task+'_'+arm,['evaluate','--task',task,'--arm',arm]) for task in p['tasks'] for arm in p['arms']]),
        ('report',[('all',['report'])]),
        ('validate',[(task,['validate','--task',task]) for task in p['tasks']]),
    ]
    names=[s[0] for s in steps]
    start=names.index(args.start)
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage,jobs in steps[start:]:
            print('START',stage,datetime.now(timezone.utc).isoformat(),flush=True)
            parallel([(stage+'_'+name,argv) for name,argv in jobs])
            exp.write_json(out/'PIPELINE.json',events)
        parts=[json.loads((out/('VALIDATION_'+t+'.json')).read_text()) for t in p['tasks']]
        exp.write_json(out/'VALIDATION.json',dict(passed=all(v['passed'] for v in parts),smoke=p['smoke'],
            protocol_sha256=p['sha256'],features_recomputed=sum(v['features_recomputed'] for v in parts),
            adaptation_checkpoints_replayed=sum(v['adaptation_checkpoints_replayed'] for v in parts),
            assessment_checkpoints_replayed=sum(len(v['assessment_replays']) for v in parts),
            zero_shot_replays=sum(len(v['zero_shot_replays']) for v in parts),
            synthetic_checkpoint_replays=len(parts[0]['synthetic_checkpoint_replays']),
            parts={t:exp.digest(out/('VALIDATION_'+t+'.json')) for t in p['tasks']},
            no_historical_validation_loaded=True,no_test_evaluation=True))
    print('PIPELINE_COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
