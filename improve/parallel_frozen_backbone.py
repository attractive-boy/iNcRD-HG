"""Resume independent runs of a frozen backbone experiment across GPUs.

Uses the original frozen training function, recipes and result format. Task
locks exclude the sequential runner. Per-run locks exclude duplicate workers.
This changes scheduling only; it does not select a model or change a budget.
"""
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from improve import backbone_replacement as runner


def worker(args, p):
    task, arm, cfg, seed, epochs = list(runner.jobs(args.out_dir, p, args.stage))[args.job]
    dest = runner.directory(args.out_dir, args.stage, task, arm, cfg['id'], seed)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest/'result.json').exists():
            runner.checked_result(dest, p)
            return
        arrays, data, weights = runner.stage_inputs(args.out_dir, task, args.stage)
        r, outputs, trace, state = runner.train(arrays, data, weights, task, arm, cfg, seed,
                                               args.stage, epochs, p, args.device)
        np.savez_compressed(dest/'scores.npz', **outputs, y_val=arrays['y_val'], pairs_val=arrays['pairs_val'])
        pd.DataFrame(trace).to_csv(dest/'trace.csv', index=False)
        torch.save(state, dest/'checkpoint.pt')
        r['artifacts_sha256'] = {name:runner.digest(dest/name) for name in ('scores.npz','trace.csv','checkpoint.pt')}
        r['executor_sha256'] = runner.digest(__file__)
        runner.write_json(dest/'result.json', r)
        print('DONE', task, arm, seed, r['auc'], r['ap'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--stage', choices=('screen','confirm','outer'), default='outer')
    parser.add_argument('--devices', nargs='+', default=['cuda:0','cuda:1'])
    parser.add_argument('--workers-per-device', type=int, default=2)
    parser.add_argument('--job', type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    p = runner.protocol(args.out_dir)
    if args.job is not None:
        worker(args, p)
        return
    if args.workers_per_device < 1:
        raise ValueError('workers-per-device must be positive')
    out = Path(args.out_dir)
    locks, active = [], []
    try:
        for task in p['tasks']:
            lock = (out/(task+'.lock')).open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        pending = []
        for index, job in enumerate(runner.jobs(out, p, args.stage)):
            task, arm, cfg, seed, epochs = job
            dest = runner.directory(out, args.stage, task, arm, cfg['id'], seed)
            if (dest/'result.json').exists():
                runner.checked_result(dest, p)
            else:
                pending.append((index, job))
        pending.sort(key=lambda item: item[1][-1]*(10 if item[1][1]=='pencil' else 1), reverse=True)
        slots = [device for _ in range(args.workers_per_device) for device in args.devices]
        runner.write_json(out/('parallel_'+args.stage+'_execution.json'), dict(
            started_at_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=p['sha256'],
            executor_sha256=runner.digest(__file__), devices=slots,
            jobs=[dict(index=i, task=j[0], arm=j[1], seed=j[3], epochs=j[4]) for i,j in pending]))
        while pending or active:
            while pending and slots:
                device = slots.pop(0)
                index, (task, arm, cfg, seed, epochs) = pending.pop(0)
                log = (out/('parallel_%s_%s_%s_s%d.log' % (args.stage, task, arm, seed))).open('a')
                proc = subprocess.Popen([sys.executable,'-m','improve.parallel_frozen_backbone',
                    '--out-dir',str(out),'--stage',args.stage,'--job',str(index),'--device',device],
                    stdout=log, stderr=subprocess.STDOUT)
                active.append((proc, log, device, task, arm, seed))
                print('LAUNCHED',task,arm,seed,device,flush=True)
            remaining = []
            for proc,log,device,task,arm,seed in active:
                status = proc.poll()
                if status is None:
                    remaining.append((proc,log,device,task,arm,seed))
                else:
                    log.close()
                    slots.append(device)
                    if status:
                        raise RuntimeError('Worker failed: %s %s %d' % (task,arm,seed))
                    print('FINISHED',task,arm,seed,flush=True)
            active = remaining
            if active:
                time.sleep(1)
    finally:
        for proc,log,*_ in active:
            if proc.poll() is None:
                proc.terminate()
            log.close()
        for lock in locks:
            lock.close()


if __name__=='__main__':
    main()
