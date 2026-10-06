"""Advance only eligible SOAP stages and classify reused fits separately."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_eigenbasis_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
FAMILY = 'eigenbasis_adam'


def classify(out, stage):
    p = experiment.common.protocol(out)
    ph = base.digest(out/'protocol.json')
    expected = {(t, k, lr, 666) for t in p['tasks'] for k in p['kinds'] for lr in p['lrs']}
    if stage == 'confirm':
        selection = base.read_json(out/'SELECTION.json')
        expected |= {(t, k, selection['chosen'][k]['lr'], s) for t in p['tasks']
                     for k in p['kinds'] for s in p['confirm_seeds']}
    found, neural, identity = set(), [], []
    for path in sorted((out/'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent, ph)
        found.add((r['task'], r['kind'], r['lr'], r['seed']))
        adopted = r.get('adopted_without_new_training', False)
        if adopted and (r['seed'] != 666 or r['kind'] not in ('full', 'adamw')
                        or base.digest(r['adopted_from']) != r['adopted_from_sha256']):
            raise ValueError('Unexpected or changed adoption')
        item = dict(path=str(path), sha256=base.digest(path), task=r['task'], kind=r['kind'],
                    lr=r['lr'], seed=r['seed'], adopted=adopted)
        state = torch.load(path.parent/'optimizer_final.pt', map_location='cpu', weights_only=True)
        if r['trained_neural_model']:
            if (r['task'] != 'MiDrug' or r['training_steps'] != 400 or r['gradient_evaluations'] != 401):
                raise ValueError('Wrong neural update/gradient budget')
            trace = base.read_json(path.parent/'trace.json')
            if [x['epoch'] for x in trace] != list(range(401)):
                raise ValueError('Incomplete optimizer trace')
            if [x['epoch'] for x in trace if 'anchor_composed' in x] != list(range(20, 401, 20)):
                raise ValueError('Different checkpoint-selection budget')
            for row in trace:
                step = row['epoch']
                if (row['omitted_fold'] != max(0, step-1)%5 or not row['state_finite']
                        or row['initialization_only'] != (step == 0)
                        or row['parameter_updates_min'] != step or row['parameter_updates_max'] != step
                        or row['parameter_gradient_calls_min'] != step+1
                        or row['parameter_gradient_calls_max'] != step+1):
                    raise ValueError('Changed step or warm-start policy')
                if row['basis_updates_min'] != (0 if r['kind'] == 'adamw' else step//10):
                    raise ValueError('Different QR budget')
            for s in state['state'].values():
                if s['step'] != 400 or s['gradient_calls'] != 401:
                    raise ValueError('Incorrect final optimizer counters')
                for key, value in s.items():
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        raise ValueError('Nonfinite optimizer state')
                    if key.endswith('_basis') and float((value.T@value-torch.eye(len(value))).abs().max()) > 5e-5:
                        raise ValueError('Invalid saved basis')
            for group in state['param_groups']:
                if group['mode'] != ('adamw' if r['kind'] == 'adamw' else 'soap'):
                    raise ValueError('Changed effective optimizer')
                if (r['kind'] == 'separable_variance') != (group.get('variance_representation') == 'separable_row_column'):
                    raise ValueError('Changed variance representation')
            neural.append(item)
        else:
            if (r['task'] != 'LncDrug' or state is not None or not r['identity_preserves_anchor']
                    or any(r[k] for k in ('epoch', 'parameters', 'trainable_parameters', 'training_steps', 'gradient_evaluations'))):
                raise ValueError('Invalid identity record')
            z = experiment.common.load(path.parent/'scores.npz')
            np.testing.assert_array_equal(z['standalone'], np.zeros_like(z['standalone']))
            np.testing.assert_array_equal(z['anchor_composed'], z['anchor'])
            identity.append(item)
    count = 6 if stage == 'screen' else 12
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or extra scoring records')
    adopted_neural = sum(r['adopted'] for r in neural)
    adopted_identity = sum(r['adopted'] for r in identity)
    if adopted_neural != 4 or adopted_identity != 4:
        raise ValueError('Wrong adoption count')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=FAMILY,
        stage=stage, protocol_sha256=ph, cumulative_new_neural_fits=len(neural)-adopted_neural,
        adopted_neural_fits=adopted_neural, cumulative_identity_scoring_records=len(identity),
        adopted_identity_scoring_records=adopted_identity, neural=neural, identity=identity,
        identity_records_count_as_training=False, independent_screen=False, confirmed_final_innovations=0))
    path = HERE/('EIGENBASIS_RECORD_CLASSIFICATION_'+stage.upper()+'_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        if any(old[k] != result[k] for k in ('neural', 'identity')):
            raise ValueError('Preserved classification disagrees')
    else:
        base.write_json(path, result)
    return path


def main():
    torch.set_num_threads(4)
    out = experiment.ROOT/'results/anchor_priority_eigenbasis_adam_v1_20260916'
    frozen = base.check_seal(base.read_json(HERE/'EIGENBASIS_QUEUE_PROTOCOL_V1.json'))
    if frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Coordinator source changed')
    for path, digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen input changed')
    result_path = HERE/'EIGENBASIS_QUEUE_RESULT_V1.json'
    if result_path.exists():
        raise ValueError('Preserve completed queue result')
    for stage in ('screen', 'confirm'):
        print('START_EIGENBASIS_STAGE', stage, datetime.now(timezone.utc).isoformat(), flush=True)
        for task in ('LncDrug', 'MiDrug'):
            with (out/(stage+'_'+task+'.log')).open('a', buffering=1) as stream:
                command = [sys.executable, '-u', '-m', 'improve.anchor_eigenbasis_experiment_v1', stage,
                           '--out', str(out), '--task', task, '--device', 'cuda:1']
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError('Eigenbasis worker failed: '+str((stage, task, completed.returncode)))
        classification = classify(out, stage)
        with (out/(stage+'_assessment.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
            if stage == 'screen' and not (out/'SELECTION.json').exists():
                experiment.select(out)
            filename = 'SCREEN_ASSESSMENT.json' if stage == 'screen' else 'HISTORICAL_ASSESSMENT.json'
            if (out/filename).exists():
                experiment.common.all_rows(out)
                result = base.check_seal(base.read_json(out/filename))
            else:
                result = audit.assess(out, stage)
        print('EIGENBASIS_GATES', stage, result['performance_gate'], result['primary_mechanism_gate'], result['offset_null_gate'], flush=True)
        if not result['declared_gate']:
            stop = out/('SCREEN_STOP_DECISION.json' if stage == 'screen' else 'HISTORICAL_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                    decision='Stop the frozen SOAP follow-up before further partition fitting',
                    assessment_sha256=base.digest(out/filename), confirmed_innovations=0, default_replaced=False)))
            break
    counts = base.check_seal(base.read_json(classification))
    base.write_json(result_path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        family=FAMILY, stage=stage, queue_protocol_sha256=frozen['sha256'], historical_gate=result['declared_gate'],
        assessment_sha256=base.digest(out/filename), classification_sha256=base.digest(classification),
        new_neural_fits=counts['cumulative_new_neural_fits'], adopted_neural_fits=counts['adopted_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'],
        adopted_identity_records=counts['adopted_identity_scoring_records'],
        partition_verification_pending=bool(result['declared_gate']), independent_screen=False,
        confirmed_final_innovations=0, existing_default_replaced=False)))
    print('EIGENBASIS_QUEUE_COMPLETE', stage, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
