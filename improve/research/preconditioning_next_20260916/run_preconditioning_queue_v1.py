"""Run only preregistered optimizer stages; stop at a failed gate."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_preconditioning_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
FAMILY = 'kronecker_preconditioning'


def run_stage(out, stage):
    for task in ('LncDrug', 'MiDrug'):
        with (out / (stage+'_'+task+'.log')).open('a', buffering=1) as stream:
            command = [sys.executable, '-u', '-m', 'improve.anchor_preconditioning_experiment_v1',
                       stage, '--family', FAMILY, '--out', str(out), '--task', task, '--device', 'cuda:1']
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError('Optimizer worker failed: '+str((stage, task, completed.returncode)))


def classify(out, stage):
    p = experiment.common.protocol(out)
    ph = base.digest(out / 'protocol.json')
    expected = {(task, kind, lr, p['screen_seed']) for task in p['tasks']
                for kind in p['kinds'] for lr in p['lrs']}
    if stage == 'confirm':
        selection = base.read_json(out / 'SELECTION.json')
        expected |= {(task, kind, selection['chosen'][kind]['lr'], seed) for task in p['tasks']
                     for kind in p['kinds'] for seed in p['confirm_seeds']}
    found, neural, identity = set(), [], []
    for path in sorted((out / 'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent, ph)
        found.add((r['task'], r['kind'], r['lr'], r['seed']))
        if r.get('adopted_without_new_training', False):
            raise ValueError('No weight adoption is registered')
        item = dict(path=str(path), sha256=base.digest(path), task=r['task'],
                    seed=r['seed'], kind=r['kind'], lr=r['lr'])
        state = torch.load(path.parent / 'optimizer_final.pt', map_location='cpu', weights_only=True)
        if r['trained_neural_model']:
            if (r['task'] != 'MiDrug' or r['training_steps'] != 400
                    or r['gradient_evaluations'] != 401 or r['initialization_gradients'] != 1):
                raise ValueError('Wrong gradient or update budget')
            trace = base.read_json(path.parent / 'trace.json')
            if len(trace) != 401 or [row['epoch'] for row in trace] != list(range(401)):
                raise ValueError('Incomplete per-gradient trace')
            if [row['epoch'] for row in trace if 'anchor_composed' in row] != list(range(20, 401, 20)):
                raise ValueError('Changed checkpoint-selection budget')
            for row in trace:
                step = row['epoch']
                if (row['omitted_fold'] != max(0, step-1) % 5
                        or row['initialization_only'] != (step == 0)
                        or row['parameter_updates_min'] != step or row['parameter_updates_max'] != step
                        or row['parameter_gradient_calls_min'] != step+1
                        or row['parameter_gradient_calls_max'] != step+1 or not row['state_finite']):
                    raise ValueError('Inconsistent per-step optimizer state')
                basis_updates = 0 if r['kind'] == 'adamw' else step // 10
                if row['basis_updates_min'] != basis_updates or row['basis_updates_max'] != basis_updates:
                    raise ValueError('Changed basis-update budget')
            orthogonality_error = 0.
            for s in state['state'].values():
                if s['step'] != 400 or s['gradient_calls'] != 401:
                    raise ValueError('Saved optimizer counters changed')
                for key, value in s.items():
                    if isinstance(value, torch.Tensor):
                        if not torch.isfinite(value).all():
                            raise ValueError('Nonfinite saved optimizer')
                        if key.endswith('_basis'):
                            error = float((value.T @ value - torch.eye(len(value))).abs().max())
                            orthogonality_error = max(orthogonality_error, error)
            if orthogonality_error > 5e-5:
                raise ValueError('Saved basis is not numerically orthogonal')
            if any(g['mode'] != r['kind'] for g in state['param_groups']):
                raise ValueError('Saved optimizer mode disagrees')
            item['maximum_basis_orthogonality_error'] = orthogonality_error
            neural.append(item)
        else:
            if (r['task'] != 'LncDrug' or not r['identity_preserves_anchor'] or state is not None
                    or any(r[k] for k in ('epoch', 'parameters', 'trainable_parameters',
                                         'training_steps', 'gradient_evaluations', 'initialization_gradients'))):
                raise ValueError('Invalid identity record')
            scores = experiment.common.load(path.parent / 'scores.npz')
            np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
            identity.append(item)
    count = len(p['kinds']) * (len(p['lrs']) + (len(p['confirm_seeds']) if stage == 'confirm' else 0))
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or unexpected scoring records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=FAMILY,
        stage=stage, protocol_sha256=ph, cumulative_new_neural_fits=len(neural),
        cumulative_identity_scoring_records=len(identity), adopted_neural_fits=0,
        neural=neural, identity=identity, formal_neural_updates=400, formal_gradient_evaluations=401,
        identical_initialization_gradient_budget=True, identity_records_count_as_training=False,
        confirmed_final_innovations=0))
    path = HERE / ('PRECONDITIONING_RECORD_CLASSIFICATION_'+stage.upper()+'_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        if any(old[k] != result[k] for k in ('neural', 'identity')):
            raise ValueError('Preserved classification disagrees')
    else:
        base.write_json(path, result)
    return path


def main():
    torch.set_num_threads(4)
    out = experiment.ROOT / 'results/anchor_priority_kronecker_preconditioning_v1_20260916'
    frozen = base.check_seal(base.read_json(HERE / 'PRECONDITIONING_QUEUE_PROTOCOL_V1.json'))
    if frozen['family'] != FAMILY or frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Frozen coordinator changed')
    for path, digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen input changed')
    experiment.common.protocol(out)
    result_path = HERE / 'PRECONDITIONING_QUEUE_RESULT_V1.json'
    if result_path.exists():
        raise ValueError('Preserve completed queue result')
    for stage in ('screen', 'confirm'):
        print('START_PRECONDITIONING_STAGE', stage, datetime.now(timezone.utc).isoformat(), flush=True)
        run_stage(out, stage)
        classification = classify(out, stage)
        with (out / (stage+'_assessment.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
            if stage == 'screen' and not (out / 'SELECTION.json').exists():
                experiment.select(out)
            filename = 'SCREEN_ASSESSMENT.json' if stage == 'screen' else 'HISTORICAL_ASSESSMENT.json'
            if (out / filename).exists():
                experiment.common.all_rows(out)
                result = base.check_seal(base.read_json(out / filename))
            else:
                result = audit.assess(out, stage)
        print('PRECONDITIONING_GATES', stage, result['performance_gate'],
              result['primary_mechanism_gate'], result['offset_null_gate'], flush=True)
        if not result['declared_gate']:
            stop = out / ('SCREEN_STOP_DECISION.json' if stage == 'screen' else 'HISTORICAL_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    decision='Stop this frozen version before additional partition fitting', stage=stage,
                    primary_kind=result['primary_kind'], assessment_sha256=base.digest(out / filename),
                    performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
                    offset_gate=result['offset_null_gate'], confirmed_innovations=0, default_replaced=False)))
            break
    counts = base.check_seal(base.read_json(classification))
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=frozen['sha256'], family=FAMILY, stage=stage,
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(out / filename),
        classification_sha256=base.digest(classification),
        new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'], adopted_neural_fits=0,
        partition_verification_pending=bool(result['declared_gate']), confirmed_final_innovations=0,
        existing_default_replaced=False)))
    print('PRECONDITIONING_QUEUE_COMPLETE', stage, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
