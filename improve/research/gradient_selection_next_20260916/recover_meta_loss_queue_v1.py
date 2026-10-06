"""Recover completed frozen STORM screening without any fitting.

The frozen coordinator rejected an FP32 CUDA mean of an all-one vector.
Its classification below is copied verbatim except the uniform-weight check:
require exact min=max=1 and a mean within one FP32 epsilon. No performance,
checkpoint, source, optimizer, selection or training rule is changed.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import torch

from improve import anchor_meta_loss_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base
from improve.research.gradient_selection_next_20260916 import run_meta_loss_queue_v1 as original

HERE = Path(__file__).resolve().parent
OUT = original.OUT

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
                for row in trace:
                    for role in ('inner', 'meta'):
                        # Exact elementwise extrema prove the uniform weights.
                        assert row[role + '_weight_min'] == row[role + '_weight_max'] == 1.
                        assert abs(row[role + '_weight_mean'] - 1.) <= np.finfo(np.float32).eps
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
        raise ValueError('Declared deterministic CuBLAS environment required')
    frozen = base.check_seal(base.read_json(HERE / 'META_LOSS_QUEUE_PROTOCOL_V1.json'))
    assert frozen['source_sha256'] == base.digest(original.__file__)
    for path, digest in frozen['input_sha256'].items():
        assert base.digest(path) == digest, path
    p = experiment.common.protocol(OUT)
    dest = HERE / 'META_LOSS_QUEUE_RESULT_V1.json'
    recovery = HERE / 'META_LOSS_RECOVERY_V1.json'
    assert not dest.exists() and not recovery.exists()
    paths = sorted((OUT / 'runs').glob('*/*/*/*/result.json'))
    before = {str(path): base.digest(path) for path in paths}
    probe = []
    for path in paths:
        r = base.read_json(path)
        if r['task'] != 'MiDrug' or r['kind'] != 'uniform':
            continue
        trace = base.read_json(path.parent / 'trace.json')
        for count in sorted({row['inner_count'] for row in trace}):
            values = torch.ones(count, device='cuda:1', dtype=torch.float32)
            probe.append(dict(lr=r['lr'], count=count, min=float(values.min()),
                max=float(values.max()), cuda_fp32_mean=float(values.mean()),
                cuda_fp64_mean=float(values.double().mean()),
                stored_means=sorted({row['inner_weight_mean'] for row in trace if row['inner_count'] == count})))
    assert any(x['cuda_fp32_mean'] != 1. for x in probe)
    assert all(x['min'] == x['max'] == x['cuda_fp64_mean'] == 1. for x in probe)
    classification = classify('screen')
    with (OUT / 'screen_assessment_recovery_v1.log').open('x', buffering=1) as stream, redirect_stdout(stream):
        assert not (OUT / 'SELECTION.json').exists()
        experiment.select(OUT)
        result = audit.assess(OUT, 'screen')
    filename = 'SCREEN_ASSESSMENT.json'
    # The completed results fail the original gate; this audit starts no follow-on fit.
    assert not result['declared_gate']
    base.write_json(OUT / 'SCREEN_STOP_DECISION.json', base.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        decision='Stop this frozen version before confirmation or partition fitting',
        stage='screen', primary_kind=result['primary_kind'], assessment_sha256=base.digest(OUT / filename),
        performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
        offset_gate=result['offset_null_gate'], confirmed_innovations=0, default_replaced=False)))
    counts = base.check_seal(base.read_json(classification))
    after = {str(path): base.digest(path) for path in paths}
    assert before == after
    base.write_json(recovery, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(__file__), original_coordinator_sha256=base.digest(original.__file__),
        original_failure_log_sha256=base.digest(HERE / 'meta_loss_queue_v1.log'),
        reason='All uniform elements equal one; FP32 CUDA mean reduction rounded by 2^-24',
        evidence=probe, result_sha256=before,
        modified_frozen_sources=False, modified_trained_artifacts=False,
        new_neural_fits=0, changed_performance_or_selection_rule=False)))
    base.write_json(dest, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=frozen['sha256'], family='meta_loss', stage='screen',
        recovered_by_readonly_audit=True, recovery_sha256=base.digest(recovery),
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(OUT / filename),
        classification_sha256=base.digest(classification), new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'], adopted_neural_fits=0,
        partition_verification_pending=False, confirmed_final_innovations=0, existing_default_replaced=False)))
    print('META_LOSS_READONLY_RECOVERY', result['declared_gate'], probe, flush=True)


if __name__ == '__main__':
    main()
