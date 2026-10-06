"""Execute the frozen resistance-flow screen and eligible confirmation."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_resistance_flow_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_resistance_flow_v1_20260916'


def run_stage(name):
    for task in ('MiDrug', 'LncDrug'):
        with (OUT / (name + '_' + task + '.log')).open('a', buffering=1) as stream:
            command = [sys.executable, '-u', '-m', 'improve.anchor_resistance_flow_experiment_v1',
                name, '--out', str(OUT), '--task', task, '--device', 'cuda:1']
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError('Resistance-flow worker failed: ' + str((name, task, completed.returncode)))


def classify(name):
    p, ph = experiment.common.protocol(OUT), base.digest(OUT / 'protocol.json')
    expected = {(task, kind, lr, p['screen_seed']) for task in p['tasks']
                for kind in p['kinds'] for lr in p['lrs']}
    if name == 'confirm':
        selection = base.read_json(OUT / 'SELECTION.json')
        expected |= {(task, kind, selection['chosen'][kind]['lr'], seed) for task in p['tasks']
                     for kind in p['kinds'] for seed in p['confirm_seeds']}
    found, neural, identity = set(), [], []
    for path in sorted((OUT / 'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent, ph)
        found.add((r['task'], r['kind'], r['lr'], r['seed']))
        if r.get('adopted_without_new_training', False):
            raise ValueError('No earlier fit is adopted in this protocol')
        item = dict(path=str(path), sha256=base.digest(path), task=r['task'],
                    seed=r['seed'], kind=r['kind'], lr=r['lr'])
        diagnostics = r['flow_diagnostics']
        trace = base.read_json(path.parent / 'trace.json')
        if r['trained_neural_model']:
            assert r['task'] == 'MiDrug' and r['trainable_parameters'] > 0
            assert r['training_steps'] == r['gradient_evaluations'] == r['optimizer_steps'] == 400
            assert len(trace) == 20 and [row['epoch'] for row in trace] == list(range(20, 401, 20))
            assert diagnostics['calls'] == 400 + 5 * 21
            assert diagnostics['entity_flows'] == 2 * diagnostics['calls']
            assert diagnostics['inverse_calls'] == diagnostics['flow_steps'] == 6 * diagnostics['calls']
            assert diagnostics['maximum_inverse_relative_residual'] <= 1e-10
            assert diagnostics['minimum_multiplier'] > 0. and diagnostics['minimum_true_resistance'] > 0.
            assert all(np.isfinite(value) for value in diagnostics.values())
            for row in trace:
                assert row['flow_diagnostics']['calls'] == row['epoch'] + 5 * (row['epoch'] // 20)
                assert np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
            final = torch.load(path.parent / 'training_final.pt', map_location='cpu', weights_only=True)
            assert final['steps'] == 400
            assert all(int(state['step']) == 400 for state in final['optimizer']['state'].values())
            assert all(torch.isfinite(value).all() for value in final['model'].values())
            neural.append(item)
        else:
            assert r['task'] == 'LncDrug' and r['identity_preserves_anchor'] and not trace
            assert all(r[k] == 0 for k in ('epoch', 'parameters', 'trainable_parameters',
                                          'training_steps', 'gradient_evaluations', 'optimizer_steps'))
            assert all(diagnostics[key] == 0 for key in ('calls', 'entity_flows', 'inverse_calls',
                'flow_steps', 'maximum_inverse_relative_residual', 'maximum_absolute_curvature'))
            assert diagnostics['minimum_multiplier'] is None and diagnostics['minimum_true_resistance'] is None
            scores = experiment.common.load(path.parent / 'scores.npz')
            np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
            identity.append(item)
    count = 6 if name == 'screen' else 12
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or unexpected completed records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=name,
        family='resistance_flow', protocol_sha256=ph, cumulative_new_neural_fits=len(neural),
        cumulative_identity_scoring_records=len(identity), adopted_neural_fits=0,
        neural=neural, identity=identity, formal_neural_steps=400,
        identity_records_count_as_training=False, gradients_per_step=1,
        flow_diagnostics_and_final_optimizer_verified=True, confirmed_final_innovations=0))
    path = HERE / ('RESISTANCE_FLOW_RECORD_CLASSIFICATION_' + name.upper() + '_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        assert old['neural'] == result['neural'] and old['identity'] == result['identity']
    else:
        base.write_json(path, result)
    return path


def main():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Declared deterministic CuBLAS environment required')
    frozen = base.check_seal(base.read_json(HERE / 'RESISTANCE_FLOW_QUEUE_PROTOCOL_V1.json'))
    assert frozen['family'] == 'resistance_flow' and frozen['source_sha256'] == base.digest(__file__)
    for path, digest in frozen['input_sha256'].items():
        assert base.digest(path) == digest, path
    experiment.common.protocol(OUT)
    destination = HERE / 'RESISTANCE_FLOW_QUEUE_RESULT_V1.json'
    if destination.exists():
        raise ValueError('Preserve completed queue result')
    for name in ('screen', 'confirm'):
        print('START_RESISTANCE_FLOW_STAGE', name, datetime.now(timezone.utc).isoformat(), flush=True)
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
        print('RESISTANCE_FLOW_GATES', name, result['performance_gate'],
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
        queue_protocol_sha256=frozen['sha256'], family='resistance_flow', stage=name,
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(OUT / filename),
        classification_sha256=base.digest(classification), new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'], adopted_neural_fits=0,
        partition_verification_pending=bool(result['declared_gate']), confirmed_final_innovations=0,
        existing_default_replaced=False)))
    print('RESISTANCE_FLOW_QUEUE_COMPLETE', name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
