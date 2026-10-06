"""Frozen joint-tuple comparison, with verified parent-primary adoption."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from improve import adopt_persistence_pairing_v1 as adoption
from improve import anchor_spectral_persistence_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_persistence_pairing_v1_20260916'


def run_stage(name):
    processes, streams = [], []
    for task in ('MiDrug', 'LncDrug'):
        stream = (OUT / (name + '_' + task + '.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-u', '-m', 'improve.anchor_spectral_persistence_experiment_v1',
                   name, '--out', str(OUT), '--task', task, '--device', 'cuda:1']
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('Persistence pairing workers failed: ' + str((name, codes)))


def classify(name):
    p, ph = experiment.common.protocol(OUT), base.digest(OUT / 'protocol.json')
    expected = {(task, kind, lr, p['screen_seed']) for task in p['tasks'] for kind in p['kinds'] for lr in p['lrs']}
    if name == 'confirm':
        selection = base.read_json(OUT / 'SELECTION.json')
        expected |= {(task, kind, selection['chosen'][kind]['lr'], seed) for task in p['tasks']
                     for kind in p['kinds'] for seed in p['confirm_seeds']}
    found, neural, identity, adopted_neural, adopted_identity = set(), [], [], [], []
    for path in sorted((OUT / 'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent, ph)
        found.add((r['task'], r['kind'], r['lr'], r['seed']))
        adopted = r.get('adopted_without_new_training', False)
        item = dict(path=str(path), sha256=base.digest(path), task=r['task'], seed=r['seed'],
                    kind=r['kind'], lr=r['lr'], adopted=adopted)
        if adopted:
            assert r['seed'] == 666 and r['kind'] == 'full' and r['new_training_steps'] == 0
            assert base.digest(r['adoption_source_result']) == r['adoption_source_result_sha256']
            assert r['adoption_prediction_max_abs'] == 0.
        diagnostics = r['persistence_diagnostics']
        trace = base.read_json(path.parent / 'trace.json')
        if r['trained_neural_model']:
            assert r['task'] == 'MiDrug' and r['trainable_parameters'] > 0
            assert r['training_steps'] == r['gradient_evaluations'] == r['optimizer_steps'] == 400
            assert len(trace) == 20 and [row['epoch'] for row in trace] == list(range(20, 401, 20))
            assert diagnostics['calls'] == 505 and diagnostics['graphs'] > 505
            assert diagnostics['max_local_nodes'] > 2 and diagnostics['eigendecompositions'] > 0
            for row in trace:
                assert row['persistence_diagnostics']['calls'] == row['epoch'] + 5 * (row['epoch'] // 20)
                assert np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
            final = torch.load(path.parent / 'training_final.pt', map_location='cpu', weights_only=True)
            assert final['steps'] == 400
            assert all(int(state['step']) == 400 for state in final['optimizer']['state'].values())
            assert all(torch.isfinite(value).all() for value in final['model'].values())
            (adopted_neural if adopted else neural).append(item)
        else:
            assert r['task'] == 'LncDrug' and r['identity_preserves_anchor'] and not trace
            assert all(r[key] == 0 for key in ('epoch', 'parameters', 'trainable_parameters',
                                               'training_steps', 'gradient_evaluations', 'optimizer_steps'))
            assert all(value == 0 for value in diagnostics.values())
            scores = experiment.common.load(path.parent / 'scores.npz')
            np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
            (adopted_identity if adopted else identity).append(item)
    count = 2 if name == 'screen' else 6
    if (found != expected or len(neural) != count or len(identity) != count
            or len(adopted_neural) != 2 or len(adopted_identity) != 2):
        raise ValueError('Missing or unexpected completed/adopted records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=name,
        family='persistence_pairing', protocol_sha256=ph, cumulative_new_neural_fits=len(neural),
        cumulative_new_identity_records=len(identity), adopted_neural_fits=len(adopted_neural),
        adopted_identity_records=len(adopted_identity), neural=neural, identity=identity,
        adopted_neural=adopted_neural, adopted_identity=adopted_identity, formal_neural_steps=400,
        gradients_per_step=1, identity_records_count_as_training=False,
        persistence_diagnostics_and_final_optimizer_verified=True, confirmed_final_innovations=0))
    path = HERE / ('PERSISTENCE_PAIRING_RECORD_CLASSIFICATION_' + name.upper() + '_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        for key in ('neural', 'identity', 'adopted_neural', 'adopted_identity'):
            assert old[key] == result[key]
    else:
        base.write_json(path, result)
    return path


def main():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Declared deterministic CuBLAS environment required')
    frozen = base.check_seal(base.read_json(HERE / 'PERSISTENCE_PAIRING_QUEUE_PROTOCOL_V1.json'))
    assert frozen['family'] == 'persistence_pairing' and frozen['source_sha256'] == base.digest(__file__)
    for path, digest in frozen['input_sha256'].items():
        assert base.digest(path) == digest, path
    experiment.common.protocol(OUT)
    destination = HERE / 'PERSISTENCE_PAIRING_QUEUE_RESULT_V1.json'
    if destination.exists():
        raise ValueError('Preserve completed queue result')
    if not (OUT / 'PARENT_ADOPTION_V1.json').exists():
        adoption.adopt(OUT)
    else:
        record = base.check_seal(base.read_json(OUT / 'PARENT_ADOPTION_V1.json'))
        assert record['protocol_sha256'] == base.digest(OUT / 'protocol.json')
        for item in record['records']:
            assert base.digest(item['path']) == item['sha256']
    for name in ('screen', 'confirm'):
        print('START_PERSISTENCE_PAIRING_STAGE', name, datetime.now(timezone.utc).isoformat(), flush=True)
        run_stage(name)
        classification = classify(name)
        with (OUT / (name + '_assessment.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
            if name == 'screen' and not (OUT / 'SELECTION.json').exists():
                experiment.select(OUT)
            filename = 'SCREEN_ASSESSMENT.json' if name == 'screen' else 'HISTORICAL_ASSESSMENT.json'
            if (OUT / filename).exists():
                experiment.common.all_rows(OUT)
                result = base.check_seal(base.read_json(OUT / filename))
            else:
                result = audit.assess(OUT, name)
        print('PERSISTENCE_PAIRING_GATES', name, result['performance_gate'],
              result['primary_mechanism_gate'], result['offset_null_gate'], flush=True)
        if not result['declared_gate']:
            stop = OUT / ('SCREEN_STOP_DECISION.json' if name == 'screen' else 'HISTORICAL_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    decision='Stop this frozen version before further confirmation, partition or outer fitting',
                    stage=name, primary_kind=result['primary_kind'], assessment_sha256=base.digest(OUT / filename),
                    performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
                    offset_gate=result['offset_null_gate'], confirmed_innovations=0, default_replaced=False)))
            break
    counts = base.check_seal(base.read_json(classification))
    base.write_json(destination, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=frozen['sha256'], family='persistence_pairing', stage=name,
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(OUT / filename),
        classification_sha256=base.digest(classification), new_neural_fits=counts['cumulative_new_neural_fits'],
        new_identity_records=counts['cumulative_new_identity_records'], adopted_neural_fits=2,
        adopted_identity_records=2, partition_verification_pending=bool(result['declared_gate']),
        prior_parent_outcomes_seen=True, independent_screen=False, confirmed_final_innovations=0,
        existing_default_replaced=False)))
    print('PERSISTENCE_PAIRING_QUEUE_COMPLETE', name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
