"""Advance only historically confirmed candidates using frozen partitions.

The v3 audit-corrected reference workers are already running separately. This queue waits
for their sealed offsets, never fits another reference, and never changes a
candidate recipe. Both neural workers are restricted to cuda:1.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from improve import anchor_residual_validate_v2 as validation


HERE = Path(__file__).resolve().parent
FAMILIES = ('geometry', 'block', 'bicentric')
TASKS = ('MiDrug', 'LncDrug')
OUT = validation.shared.data_source.OUT
CACHE = validation.CACHE
base = validation.base


def event(*values):
    print(datetime.now(timezone.utc).isoformat(), *values, flush=True)


def protocol():
    p = base.check_seal(base.read_json(HERE / 'PARTITION_QUEUE_PROTOCOL_V2.json'))
    if base.digest(__file__) != p['source_sha256']:
        raise ValueError('Partition coordinator source changed')
    if p['families'] != list(FAMILIES) or p['tasks'] != list(TASKS):
        raise ValueError('Queue membership changed')
    if validation.experiment.nested.protocol(CACHE)['sha256'] != p['nested_protocol_sha256']:
        raise ValueError('Nested reference protocol changed')
    return p


def historical_status(family):
    label = validation.label_for(family)
    candidate = OUT / 'candidates' / (label + '.json')
    if candidate.exists():
        validation.checked(OUT, family)
        return 'confirmed'
    parent = validation.ROOT / 'results' / ('anchor_residual_' + family + '_v1_20260915')
    for name in ('SCREEN_STOP_DECISION.json', 'HISTORICAL_STOP_DECISION.json'):
        path = parent / name
        if path.exists():
            base.check_seal(base.read_json(path))
            return 'stopped'
    return 'pending'


def worker(task):
    protocol()
    dp = validation.shared.inputs.protocol(OUT)
    paths = [CACHE / 'offsets' / task / str(split) / 'manifest.json'
             for split in dp['split_seeds']]
    event('WAITING_FOR_EXISTING_NESTED_OFFSETS', task)
    while not all(path.exists() for path in paths):
        time.sleep(15)
    event('NESTED_OFFSETS_AVAILABLE', task)
    for family in FAMILIES:
        status = historical_status(family)
        while status == 'pending':
            time.sleep(15)
            status = historical_status(family)
        if status == 'stopped':
            event('SKIPPED_HISTORICAL_FAILURE', task, family)
            continue
        event('START_FIXED_PARTITIONS', task, family)
        validation.run(OUT, family, task, 'cuda:1')
        event('FINISHED_FIXED_PARTITIONS', task, family)


def main():
    p = protocol()
    streams, processes = [], []
    for task in TASKS:
        stream = (HERE / ('partition_queue_' + task + '_v2.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-m',
                   'improve.research.new_mechanisms_20260915.residual_frontier.run_partition_queue_v2',
                   '--task', task]
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('Fixed-partition workers failed: ' + str(codes))
    results = {}
    for family in FAMILIES:
        if historical_status(family) == 'stopped':
            results[family] = dict(status='historical_failure')
            continue
        label = validation.label_for(family)
        assessment_path = OUT / 'reports' / label / 'ASSESSMENT.json'
        if not assessment_path.exists():
            with (HERE / ('partition_report_' + family + '_v2.log')).open('a', buffering=1) as stream:
                with redirect_stdout(stream):
                    validation.report(OUT, family)
        assessment = base.read_json(assessment_path)
        results[family] = dict(status='partitions_complete',
            gate=assessment['additional_development_gate'],
            assessment_sha256=base.digest(assessment_path))
        event('FIXED_PARTITION_GATE', family, assessment['performance_gate'],
              assessment['primary_mechanism_gate'], assessment['offset_null_gate'])
    dest = HERE / 'PARTITION_QUEUE_RESULT_V2.json'
    if dest.exists():
        raise ValueError('Preserve the existing queue result')
    base.write_json(dest, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=p['sha256'], results=results,
        outer_recheck_still_required=True, confirmed_final_innovations=0)))
    event('PARTITION_QUEUE_COMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    if args.task:
        worker(args.task)
    else:
        main()
