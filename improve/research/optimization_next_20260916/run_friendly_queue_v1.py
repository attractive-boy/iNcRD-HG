"""Frozen F-SAM screening/confirmation with explicit fit/identity accounting."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np

from improve import anchor_sharpness_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base


HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_friendly_sharpness_v1_20260916'


def stage(name):
    processes, streams = [], []
    for task in ('MiDrug', 'LncDrug'):
        stream = (OUT / (name + '_' + task + '.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-u', '-m', 'improve.anchor_sharpness_experiment_v1', name,
            '--family', 'friendly_sharpness', '--out', str(OUT), '--task', task, '--device', 'cuda:1']
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('F-SAM workers failed: ' + str((name, codes)))


def classify_records(name):
    p = experiment.common.protocol(OUT)
    ph = base.digest(OUT / 'protocol.json')
    neural, identity = [], []
    for path in sorted((OUT / 'runs').glob('*/*/*/*/result.json')):
        record = experiment.common.verify(path.parent, ph)
        item = dict(path=str(path), sha256=base.digest(path), task=record['task'],
                    seed=record['seed'], kind=record['kind'], lr=record['lr'])
        if record['trained_neural_model']:
            if (record['task'] != 'MiDrug' or record['training_steps'] != p['epochs']
                    or record['gradient_evaluations'] != 2 * p['epochs']
                    or record['trainable_parameters'] < 1):
                raise ValueError('Unexpected neural training record')
            neural.append(item)
        else:
            if (record['task'] != 'LncDrug' or not record['identity_preserves_anchor']
                    or any(record[k] != 0 for k in ('epoch', 'parameters',
                        'trainable_parameters', 'training_steps', 'gradient_evaluations'))):
                raise ValueError('Unexpected identity record')
            scores = experiment.common.load(path.parent / 'scores.npz')
            np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
            identity.append(item)
    expected = 6 if name == 'screen' else 12
    if len(neural) != expected or len(identity) != expected:
        raise ValueError('Missing or unexpected scoring records')
    output = HERE / ('FRIENDLY_RECORD_CLASSIFICATION_' + name.upper() + '_V1.json')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=name,
        protocol_sha256=ph, cumulative_new_neural_fits=len(neural),
        cumulative_identity_scoring_records=len(identity), neural=neural, identity=identity,
        identity_records_count_as_training=False, formal_neural_steps=p['epochs'],
        gradients_per_neural_step=2, confirmed_final_innovations=0))
    if output.exists():
        old = base.check_seal(base.read_json(output))
        if old['neural'] != neural or old['identity'] != identity:
            raise ValueError('Preserved record classification disagrees')
    else:
        base.write_json(output, result)
    return output


def main():
    frozen = base.check_seal(base.read_json(HERE / 'FRIENDLY_QUEUE_PROTOCOL_V1.json'))
    if frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Frozen F-SAM coordinator changed')
    for path, digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen queue input changed')
    experiment.common.protocol(OUT)
    result_path = HERE / 'FRIENDLY_QUEUE_RESULT_V1.json'
    if result_path.exists():
        raise ValueError('Preserve the completed queue result')
    for name in ('screen', 'confirm'):
        print('START_FRIENDLY_STAGE', name, datetime.now(timezone.utc).isoformat(), flush=True)
        stage(name)
        classification = classify_records(name)
        with (OUT / (name + '_assessment.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
            if name == 'screen' and not (OUT / 'SELECTION.json').exists():
                experiment.select(OUT)
            filename = 'SCREEN_ASSESSMENT.json' if name == 'screen' else 'HISTORICAL_ASSESSMENT.json'
            if (OUT / filename).exists():
                experiment.common.all_rows(OUT)
                result = base.check_seal(base.read_json(OUT / filename))
            else:
                result = audit.assess(OUT, name)
        print('FRIENDLY_GATES', name, result['performance_gate'], result['primary_mechanism_gate'],
              result['offset_null_gate'], flush=True)
        if not result['declared_gate']:
            stop = OUT / ('SCREEN_STOP_DECISION.json' if name == 'screen' else 'HISTORICAL_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    decision='Stop this frozen version before additional partition fitting', stage=name,
                    primary_kind=result['primary_kind'], assessment_sha256=base.digest(OUT / filename),
                    performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
                    offset_gate=result['offset_null_gate'], confirmed_innovations=0, default_replaced=False)))
            break
    counts = base.check_seal(base.read_json(classification))
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=frozen['sha256'], stage=name, historical_gate=result['declared_gate'],
        assessment_sha256=base.digest(OUT / filename), classification_sha256=base.digest(classification),
        new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'],
        partition_verification_pending=bool(result['declared_gate']),
        confirmed_final_innovations=0, existing_default_replaced=False)))
    print('FRIENDLY_QUEUE_COMPLETE', name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
