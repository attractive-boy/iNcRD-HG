"""Frozen Muon screening/confirmation with explicit fit/identity accounting."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np

from improve import anchor_muon_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base


HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_spectral_muon_v1_20260916'


def stage(name):
    processes, streams = [], []
    for task in ('MiDrug', 'LncDrug'):
        stream = (OUT / (name + '_' + task + '.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-u', '-m', 'improve.anchor_muon_experiment_v1', name,
            '--family', 'spectral_muon', '--out', str(OUT), '--task', task, '--device', 'cuda:1']
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('Muon workers failed: ' + str((name, codes)))


def classify_records(name):
    p = experiment.common.protocol(OUT)
    ph = base.digest(OUT / 'protocol.json')
    neural, adopted, identity = [], [], []
    for path in sorted((OUT / 'runs').glob('*/*/*/*/result.json')):
        record = experiment.common.verify(path.parent, ph)
        item = dict(path=str(path), sha256=base.digest(path), task=record['task'],
                    seed=record['seed'], kind=record['kind'], lr=record['lr'],
                    adopted_without_new_training=record.get('adopted_without_new_training', False))
        if item['adopted_without_new_training']:
            if (record['kind'] != 'adamw' or record['seed'] != 666 or record['new_training_steps'] != 0
                    or base.digest(record['adoption_source_result']) != record['adoption_source_result_sha256']):
                raise ValueError('Unexpected baseline adoption')
        if record['trained_neural_model']:
            gradients = 800 if item['adopted_without_new_training'] else 400
            if (record['task'] != 'MiDrug' or record['training_steps'] != 400
                    or record['gradient_evaluations'] != gradients or record['trainable_parameters'] < 1):
                raise ValueError('Unexpected neural training record')
            (adopted if item['adopted_without_new_training'] else neural).append(item)
        else:
            if (record['task'] != 'LncDrug' or not record['identity_preserves_anchor']
                    or any(record[k] != 0 for k in ('epoch', 'parameters', 'trainable_parameters',
                                                   'training_steps', 'gradient_evaluations'))):
                raise ValueError('Unexpected identity scoring record')
            scores = experiment.common.load(path.parent / 'scores.npz')
            np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
            identity.append(item)
    if (len(neural) != (4 if name == 'screen' else 10) or len(adopted) != 2
            or len(identity) != (6 if name == 'screen' else 12)
            or sum(x['adopted_without_new_training'] for x in identity) != 2):
        raise ValueError('Missing or unexpected scoring records')
    output = HERE / ('MUON_RECORD_CLASSIFICATION_' + name.upper() + '_V1.json')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=name,
        protocol_sha256=ph, cumulative_new_neural_fits=len(neural), adopted_neural_fits=len(adopted),
        cumulative_identity_scoring_records=len(identity), adopted_identity_records=2,
        neural=neural, adopted=adopted, identity=identity, formal_neural_steps=400,
        identity_records_count_as_training=False, adopted_weights_count_as_new_training=False,
        gradients_per_new_neural_step=1, original_adopted_gradients_per_step=2,
        confirmed_final_innovations=0))
    if output.exists():
        old = base.check_seal(base.read_json(output))
        if any(old[k] != result[k] for k in ('neural', 'adopted', 'identity')):
            raise ValueError('Preserved record classification disagrees')
    else:
        base.write_json(output, result)
    return output

def main():
    frozen = base.check_seal(base.read_json(HERE / 'MUON_QUEUE_PROTOCOL_V1.json'))
    if frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Frozen Muon coordinator changed')
    for path, digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen queue input changed')
    experiment.common.protocol(OUT)
    result_path = HERE / 'MUON_QUEUE_RESULT_V1.json'
    if result_path.exists():
        raise ValueError('Preserve the completed queue result')
    for name in ('screen', 'confirm'):
        print('START_MUON_STAGE', name, datetime.now(timezone.utc).isoformat(), flush=True)
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
        print('MUON_GATES', name, result['performance_gate'], result['primary_mechanism_gate'],
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
        adopted_neural_fits=counts['adopted_neural_fits'], adopted_identity_records=2,
        identity_scoring_records=counts['cumulative_identity_scoring_records'],
        partition_verification_pending=bool(result['declared_gate']),
        confirmed_final_innovations=0, existing_default_replaced=False)))
    print('MUON_QUEUE_COMPLETE', name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
