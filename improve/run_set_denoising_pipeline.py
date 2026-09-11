"""Schedule the frozen experiment; this file changes no training settings."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--start', choices=('features', 'confirm', 'outer', 'validate'), default='features')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir).resolve()
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[variable] = '4'
    steps = [
        ('features_inner', 'features', 'inner', True),
        ('screen', 'run', 'screen', True),
        ('select_screen', 'select', 'screen', False),
        ('confirm', 'run', 'confirm', True),
        ('select_confirm', 'select', 'confirm', False),
        ('features_outer', 'features', 'outer', True),
        ('outer', 'run', 'outer', True),
        ('report_before_replay', 'report', None, False),
        ('validate', 'validate', None, True),
        ('report_final', 'report', None, False),
    ]
    begin = {'features':0, 'confirm':3, 'outer':5, 'validate':8}[args.start]

    def run_one(name, command, stage, task=None, gpu=None):
        argv = [sys.executable, '-m', 'improve.set_denoising', command, '--out-dir', str(out)]
        if stage:
            argv += ['--stage', stage]
        if task:
            argv += ['--task', task, '--device', 'cuda:'+str(gpu)]
        logfile = logs / (name + ('_'+task if task else '') + '.log')
        with logfile.open('a') as stream:
            stream.write('\n' + datetime.now(timezone.utc).isoformat() + '\n' + json.dumps(argv) + '\n')
            stream.flush()
            result = subprocess.run(argv, cwd=str(repo), env=env, stdout=stream, stderr=subprocess.STDOUT)
        event = dict(time=datetime.now(timezone.utc).isoformat(), step=name, task=task,
                     exit_code=result.returncode, log=str(logfile))
        print(json.dumps(event), flush=True)
        if result.returncode:
            print('\n'.join(logfile.read_text().splitlines()[-35:]), flush=True)
            raise RuntimeError('Stage failed: '+name+' '+str(task))
        return event

    events = []
    for name, command, stage, parallel in steps[begin:]:
        print('START', name, datetime.now(timezone.utc).isoformat(), flush=True)
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_one, name, command, stage, task, gpu)
                           for gpu, task in enumerate(('LncDrug', 'MiDrug'))]
                for future in as_completed(futures):
                    events.append(future.result())
        else:
            events.append(run_one(name, command, stage))
        (out / ('PIPELINE_'+args.start+'.json')).write_text(json.dumps(events, indent=2)+'\n')
    print('PIPELINE_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
