"""Run frozen meta-loss stages and stop at the first failed joint gate."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_meta_loss_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_meta_loss_v1_20260916'


def run_stage(name):
    processes, streams = [], []
    for task in ('MiDrug', 'LncDrug'):
        stream = (OUT / (name + '_' + task + '.log')).open('a', buffering=1)
        streams.append(stream)
        command = [sys.executable, '-u', '-m', 'improve.anchor_meta_loss_experiment_v1',
            name, '--out', str(OUT), '--task', task, '--device', 'cuda:1']
        processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        raise RuntimeError('Meta-loss workers failed: ' + str((name, codes)))


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
            raise ValueError('This protocol does not adopt earlier fits')
        item = dict(path=str(path), sha256=base.digest(path), task=r['task'], seed=r['seed'], kind=r['kind'], lr=r['lr'])
        if r['trained_neural_model']:
            if r['task'] != 'MiDrug' or r['training_steps'] != 400 or r['gradient_evaluations'] != 1200:
                raise ValueError('Unexpected neural fitting count')
            if (r['task_gradient_evaluations'] != 400 or r['rescaler_gradient_evaluations'] != 800
                    or r['task_optimizer_steps'] != 400 or r['rescaler_optimizer_steps'] != 400
                    or r['dropout_forward_passes'] != 2400):
                raise ValueError('The declared gradient or forward budget changed')
            trace = base.read_json(path.parent / 'trace.json')
            if len(trace) != 400 or [row['task_moment_step'] for row in trace] != list(range(1, 401)):
                raise ValueError('Task optimizer moments were not persistent')
            final = torch.load(path.parent / 'training_final.pt', map_location='cpu', weights_only=True)
            if final['steps'] != 400 or final['task_optimizer']['step'] != 400:
                raise ValueError('Final task optimizer checkpoint mismatch')
            if any(int(state['step']) != 400 for state in final['rescaler_optimizer']['state'].values()):
                raise ValueError('Final rescaler optimizer checkpoint mismatch')
            if r['kind'] in ('no_meta_gradient', 'uniform'):
                assert r['hypergradient_steps'] == 0 and all(row['hypergradient_norm'] == 0. for row in trace)
            else:
                assert r['hypergradient_steps'] == 400 and any(row['hypergradient_norm'] > 0. for row in trace)
            if r['kind'] in ('no_outer_rescaling', 'uniform'):
                assert r['direct_gradient_steps'] == 0 and all(row['direct_gradient_norm'] == 0. for row in trace)
            else:
                assert r['direct_gradient_steps'] == 400 and any(row['direct_gradient_norm'] > 0. for row in trace)
            if r['kind'] == 'uniform':
                assert all(row['inner_weight_mean'] == 1. and row['meta_weight_mean'] == 1. for row in trace)
            neural.append(item)
        else:
            if r['task'] != 'LncDrug' or not r['identity_preserves_anchor'] or any(r[k] != 0 for k in
                    ('epoch', 'parameters', 'trainable_parameters', 'training_steps', 'gradient_evaluations')):
                raise ValueError('Unexpected identity record')
            scores = experiment.common.load(path.parent / 'scores.npz')
            np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
            identity.append(item)
    count = 8 if name == 'screen' else 16
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or unexpected scoring records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=name,
        family='meta_loss', protocol_sha256=ph, cumulative_new_neural_fits=len(neural),
        cumulative_identity_scoring_records=len(identity), adopted_neural_fits=0,
        neural=neural, identity=identity, formal_neural_steps=400,
        identity_records_count_as_training=False, gradient_requests_per_step=3,
        all_gradient_path_and_optimizer_gates_passed=True, confirmed_final_innovations=0))
    path = HERE / ('META_LOSS_RECORD_CLASSIFICATION_' + name.upper() + '_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        if any(old[k] != result[k] for k in ('neural', 'identity')):
            raise ValueError('Preserved classification disagrees')
    else:
        base.write_json(path, result)
    return path


def main():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Set the declared deterministic CuBLAS environment first')
    frozen = base.check_seal(base.read_json(HERE / 'META_LOSS_QUEUE_PROTOCOL_V1.json'))
    if frozen['family'] != 'meta_loss' or frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Frozen coordinator changed')
    for path, digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen input changed')
    experiment.common.protocol(OUT)
    result_path = HERE / 'META_LOSS_QUEUE_RESULT_V1.json'
    if result_path.exists():
        raise ValueError('Preserve completed queue result')
    for name in ('screen', 'confirm'):
        print('START_META_LOSS_STAGE', name, datetime.now(timezone.utc).isoformat(), flush=True)
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
        print('META_LOSS_GATES', name, result['performance_gate'], result['primary_mechanism_gate'], result['offset_null_gate'], flush=True)
        if not result['declared_gate']:
            stop = OUT / ('SCREEN_STOP_DECISION.json' if name == 'screen' else 'HISTORICAL_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    decision='Stop this frozen version before further confirmation or partition fitting', stage=name,
                    primary_kind=result['primary_kind'], assessment_sha256=base.digest(OUT / filename),
                    performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
                    offset_gate=result['offset_null_gate'], confirmed_innovations=0, default_replaced=False)))
            break
    counts = base.check_seal(base.read_json(classification))
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=frozen['sha256'], family='meta_loss', stage=name,
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(OUT / filename),
        classification_sha256=base.digest(classification), new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'], adopted_neural_fits=0,
        partition_verification_pending=bool(result['declared_gate']),
        confirmed_final_innovations=0, existing_default_replaced=False)))
    print('META_LOSS_QUEUE_COMPLETE', name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
