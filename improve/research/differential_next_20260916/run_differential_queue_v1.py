"""Screen and confirm frozen feature-interaction studies, counting actual fits."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np

from improve import anchor_priority_graph_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def run_stage(out, family, name):
    processes, streams = [], []
    for task in ('MiDrug', 'LncDrug'):
        stream = (out / (name + '_' + task + '.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-u', '-m', 'improve.anchor_priority_graph_v1', name,
            '--family', family, '--out', str(out), '--task', task, '--device', 'cuda:1']
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('Feature workers failed: ' + str((family, name, codes)))


def classify(out, family, name):
    p = experiment.common.protocol(out)
    ph = base.digest(out / 'protocol.json')
    expected = {(task, kind, lr, p['screen_seed']) for task in p['tasks']
                for kind in p['kinds'] for lr in p['lrs']}
    if name == 'confirm':
        selection = base.read_json(out / 'SELECTION.json')
        expected |= {(task, kind, selection['chosen'][kind]['lr'], seed) for task in p['tasks']
                     for kind in p['kinds'] for seed in p['confirm_seeds']}
    found, neural, identity = set(), [], []
    for path in sorted((out / 'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent, ph)
        found.add((r['task'], r['kind'], r['lr'], r['seed']))
        if r.get('adopted_without_new_training', False):
            raise ValueError('This protocol does not adopt previous fits')
        item = dict(path=str(path), sha256=base.digest(path), task=r['task'],
                    seed=r['seed'], kind=r['kind'], lr=r['lr'])
        if r['trained_neural_model']:
            if (r['task'] != 'MiDrug' or r['training_steps'] != 400
                    or r['gradient_evaluations'] != 400 or r['trainable_parameters'] < 1):
                raise ValueError('Unexpected neural fitting record')
            neural.append(item)
        else:
            if (r['task'] != 'LncDrug' or not r['identity_preserves_anchor']
                    or any(r[k] != 0 for k in ('epoch', 'parameters', 'trainable_parameters',
                                              'training_steps', 'gradient_evaluations'))):
                raise ValueError('Unexpected identity record')
            scores = experiment.common.load(path.parent / 'scores.npz')
            np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
            identity.append(item)
    count = 4 if name == 'screen' else 8
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or unexpected scoring records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=name,
        family=family, protocol_sha256=ph, cumulative_new_neural_fits=len(neural),
        cumulative_identity_scoring_records=len(identity), adopted_neural_fits=0,
        neural=neural, identity=identity, formal_neural_steps=400,
        identity_records_count_as_training=False, gradients_per_step=1, confirmed_final_innovations=0))
    path = HERE / (family.upper() + '_RECORD_CLASSIFICATION_' + name.upper() + '_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        if any(old[k] != result[k] for k in ('neural', 'identity')):
            raise ValueError('Preserved classification disagrees')
    else:
        base.write_json(path, result)
    return path


def main(family):
    out = experiment.ROOT / ('results/anchor_priority_' + family + '_v1_20260916')
    frozen = base.check_seal(base.read_json(HERE / (family.upper() + '_QUEUE_PROTOCOL_V1.json')))
    if frozen['family'] != family or frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Frozen coordinator changed')
    for path, digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen input changed')
    experiment.common.protocol(out)
    result_path = HERE / (family.upper() + '_QUEUE_RESULT_V1.json')
    if result_path.exists():
        raise ValueError('Preserve completed queue result')
    for name in ('screen', 'confirm'):
        print('START_FEATURE_STAGE', family, name, datetime.now(timezone.utc).isoformat(), flush=True)
        run_stage(out, family, name)
        classification = classify(out, family, name)
        with (out / (name + '_assessment.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
            if name == 'screen' and not (out / 'SELECTION.json').exists():
                experiment.select(out)
            filename = 'SCREEN_ASSESSMENT.json' if name == 'screen' else 'HISTORICAL_ASSESSMENT.json'
            if (out / filename).exists():
                experiment.common.all_rows(out)
                result = base.check_seal(base.read_json(out / filename))
            else:
                result = audit.assess(out, name)
        print('FEATURE_GATES', family, name, result['performance_gate'],
              result['primary_mechanism_gate'], result['offset_null_gate'], flush=True)
        if not result['declared_gate']:
            stop = out / ('SCREEN_STOP_DECISION.json' if name == 'screen' else 'HISTORICAL_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    decision='Stop this frozen version before additional partition fitting', stage=name,
                    primary_kind=result['primary_kind'], assessment_sha256=base.digest(out / filename),
                    performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
                    offset_gate=result['offset_null_gate'], confirmed_innovations=0, default_replaced=False)))
            break
    counts = base.check_seal(base.read_json(classification))
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=frozen['sha256'], family=family, stage=name,
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(out / filename),
        classification_sha256=base.digest(classification),
        new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'],
        adopted_neural_fits=0, partition_verification_pending=bool(result['declared_gate']),
        confirmed_final_innovations=0, existing_default_replaced=False)))
    print('FEATURE_QUEUE_COMPLETE', family, name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('differential_attention',), required=True)
    main(parser.parse_args().family)
