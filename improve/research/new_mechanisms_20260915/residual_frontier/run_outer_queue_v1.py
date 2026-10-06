"""Automatically recheck only candidates that pass the fixed partitions.

All recipes are already frozen. The shared outer training reference is built
once, with no outer outcome access during its construction or fitting. The
outer surface has historical exposure and is never called an independent test.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from improve import anchor_residual_outer_v1 as outer
from improve import nested_anchor_fit_v3 as reference


HERE = Path(__file__).resolve().parent
FAMILIES = ('geometry', 'block', 'bicentric')
TASKS = ('MiDrug', 'LncDrug')
PARENT_FEATURES = outer.ROOT / 'results/nested_anchor_v2_20260915/outer'
base = outer.base


def event(*values):
    print(datetime.now(timezone.utc).isoformat(), *values, flush=True)


def protocol():
    p = base.check_seal(base.read_json(HERE / 'OUTER_QUEUE_PROTOCOL_V1.json'))
    if base.digest(__file__) != p['source_sha256']:
        raise ValueError('Outer coordinator changed')
    for path, sha in p['frozen_input_sha256'].items():
        if base.digest(path) != sha:
            raise ValueError('Frozen outer queue input changed')
    return p


def eligible():
    record = base.check_seal(base.read_json(HERE / 'PARTITION_QUEUE_RESULT_V2.json'))
    result = []
    for family in FAMILIES:
        status = record['results'][family]
        if status.get('gate'):
            path = outer.DEV / 'reports' / outer.label_for(family) / 'ASSESSMENT.json'
            if base.digest(path) != status['assessment_sha256']:
                raise ValueError('Partition assessment changed after the queue result')
            dp, cp = outer.development.checked(outer.DEV, family)
            outer.verify_partition_report(path, dp, cp)
            result.append(family)
    return result


def worker(task):
    protocol()
    families = eligible()
    reference.run(outer.CACHE, task, 'cuda:1')
    reference.previous.assemble(outer.CACHE, task)
    event('OUTER_TRAINING_REFERENCE_COMPLETE', task)
    for family in families:
        event('START_FROZEN_OUTER_FITS', task, family)
        outer.run(outer.OUT, family, task, 'cuda:1')
        event('FINISHED_FROZEN_OUTER_FITS', task, family)


def main():
    p = protocol()
    event('WAITING_FOR_COMPLETE_PARTITION_GATES')
    while not (HERE / 'PARTITION_QUEUE_RESULT_V2.json').exists():
        time.sleep(15)
    families = eligible()
    event('ELIGIBLE_FOR_OUTER', families)
    results = {}
    if families:
        if not PARENT_FEATURES.exists():
            reference.inputs.freeze(PARENT_FEATURES, 'outer')
        for task in TASKS:
            reference.inputs.features(PARENT_FEATURES, task)
        if not outer.CACHE.exists():
            reference.freeze(outer.CACHE, PARENT_FEATURES)
        for family in families:
            path = outer.OUT / 'candidates' / (outer.label_for(family) + '.json')
            if not path.exists():
                outer.freeze(outer.OUT, family)
            outer.checked(outer.OUT, family)
        streams, processes = [], []
        for task in TASKS:
            stream = (HERE / ('outer_queue_' + task + '_v1.log')).open('a', buffering=1)
            streams.append(stream)
            command = [sys.executable, '-m',
                'improve.research.new_mechanisms_20260915.residual_frontier.run_outer_queue_v1',
                '--task', task]
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
        for stream in streams:
            stream.close()
        if any(codes):
            raise RuntimeError('Frozen outer workers failed: ' + str(codes))
        for family in families:
            path = outer.OUT / 'reports' / outer.label_for(family) / 'ASSESSMENT.json'
            if not path.exists():
                with (HERE / ('outer_report_' + family + '_v1.log')).open('a', buffering=1) as stream:
                    with redirect_stdout(stream):
                        outer.report(outer.OUT, family)
            assessment = base.check_seal(base.read_json(path))
            results[family] = dict(gate=assessment['frozen_recheck_gate'],
                                  assessment_sha256=base.digest(path))
            event('FROZEN_OUTER_GATE', family, assessment['performance_gate'],
                  assessment['primary_mechanism_gate'], assessment['offset_null_gate'])
    dest = HERE / 'OUTER_QUEUE_RESULT_V1.json'
    if dest.exists():
        raise ValueError('Preserve existing outer queue result')
    base.write_json(dest, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=p['sha256'], eligible_families=families, results=results,
        historical_surface_previously_exposed=True, independent_test=False,
        goal_completion_not_automatic=True)))
    event('OUTER_QUEUE_COMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    if args.task:
        worker(args.task)
    else:
        main()
