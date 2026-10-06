"""Run the fixed auxiliary expected-graph candidate and its matched controls."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_expected_graph_experiment_v1 as experiment
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
FAMILY = 'expected_graph_selftraining'


def run_stage(out, name):
    for task in ('LncDrug', 'MiDrug'):
        with (out / (name + '_' + task + '.log')).open('a', buffering=1) as stream:
            command = [sys.executable, '-u', '-m', 'improve.anchor_expected_graph_experiment_v1',
                name, '--family', FAMILY, '--out', str(out), '--task', task, '--device', 'cuda:1']
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Expected-graph worker failed: ' + str((name, task, result.returncode)))


def classify(out, name):
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
        history = torch.load(path.parent / 'augmentation.pt', map_location='cpu', weights_only=False)
        if r['trained_neural_model']:
            if (r['task'] != 'MiDrug' or r['training_steps'] != 400 or r['gradient_evaluations'] != 400
                    or r['graph_forward_evaluations'] != 800 or r['teacher_refreshes'] != 8
                    or r['epoch'] < 60 or r['epoch'] % 20 or r['inference_uses_expected_graph']):
                raise ValueError('Unexpected neural fitting or checkpoint budget')
            trace = base.read_json(path.parent / 'trace.json')
            if len(trace) != 400 or [row['epoch'] for row in trace] != list(range(1, 401)):
                raise ValueError('Incomplete per-step self-training trace')
            if [row['epoch'] for row in trace if 'anchor_composed' in row] != list(range(20, 401, 20)):
                raise ValueError('Changed evaluation cadence')
            if [entry['before_step'] for entry in history] != list(range(1, 401, 50)):
                raise ValueError('Missing teacher stage evidence')
            for stage, entry in enumerate(history):
                if entry['stage'] != stage or len(entry['contexts']) != 5 or not entry['teacher_state']:
                    raise ValueError('Wrong teacher snapshot structure')
                for graph in entry['contexts']:
                    pairs = set(map(tuple, graph['query_pairs'].T))
                    if any(tuple(q) in pairs or tuple(q[::-1]) in pairs for q in graph['added_pairs'].T):
                        raise ValueError('Teacher proposed a current query as an edge')
                    if graph['added_pairs'].shape[1] != len(graph['added_weights']):
                        raise ValueError('Wrong probability-weight pairing')
                    if np.any((graph['added_weights'] <= 0) | (graph['added_weights'] >= 1)):
                        raise ValueError('Non-probabilistic expected edges')
                    if r['kind'] == 'observed_only' and len(graph['added_weights']):
                        raise ValueError('Observed-only arm acquired virtual edges')
            for row in trace:
                expected_aux = row['expected_bce'] if r['kind'] == 'no_correction' else .5 * (
                    row['expected_bce'] + row['correction_bce'])
                if abs(row['auxiliary'] - expected_aux) > 1e-10:
                    raise ValueError('Correction control changed auxiliary scale')
                if abs(row['loss'] - row['bce'] - .1 * row['auxiliary']) > 1e-10:
                    raise ValueError('Auxiliary objective changed')
                if row['stage'] != (row['epoch'] - 1) // 50 or row['omitted_fold'] != (row['epoch'] - 1) % 5:
                    raise ValueError('Wrong stage or support fold')
            item['soft_unknown_presentations_before_checkpoint'] = sum(
                row['soft_unknown_count'] for row in trace[:r['epoch']])
            item['virtual_edge_presentations_before_checkpoint'] = sum(
                row['virtual_edges'] for row in trace[:r['epoch']])
            neural.append(item)
        else:
            if (r['task'] != 'LncDrug' or not r['identity_preserves_anchor'] or history
                    or any(r[k] != 0 for k in ('epoch', 'parameters', 'trainable_parameters',
                                              'training_steps', 'gradient_evaluations', 'teacher_refreshes'))):
                raise ValueError('Unexpected identity record')
            scores = experiment.common.load(path.parent / 'scores.npz')
            np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
            identity.append(item)
    count = len(p['kinds']) * (len(p['lrs']) + (len(p['confirm_seeds']) if name == 'confirm' else 0))
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or unexpected scoring records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=name,
        family=FAMILY, protocol_sha256=ph, cumulative_new_neural_fits=len(neural),
        cumulative_identity_scoring_records=len(identity), adopted_neural_fits=0,
        neural=neural, identity=identity, formal_neural_steps=400,
        per_step_objective_and_stage_budget_verified=True, identity_records_count_as_training=False,
        teacher_snapshots_count_as_training=False, gradients_per_step=1, confirmed_final_innovations=0))
    path = HERE / ('EXPECTED_GRAPH_SELFTRAINING_RECORD_CLASSIFICATION_' + name.upper() + '_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        if any(old[k] != result[k] for k in ('neural', 'identity')):
            raise ValueError('Preserved classification disagrees')
    else:
        base.write_json(path, result)
    return path


def main():
    out = experiment.ROOT / 'results/anchor_priority_expected_graph_selftraining_v1_20260916'
    frozen = base.check_seal(base.read_json(HERE / 'EXPECTED_GRAPH_SELFTRAINING_QUEUE_PROTOCOL_V1.json'))
    if frozen['family'] != FAMILY or frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Frozen coordinator changed')
    for path, digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen input changed')
    experiment.common.protocol(out)
    result_path = HERE / 'EXPECTED_GRAPH_SELFTRAINING_QUEUE_RESULT_V1.json'
    if result_path.exists():
        raise ValueError('Preserve completed queue result')
    for name in ('screen', 'confirm'):
        print('START_EXPECTED_GRAPH_STAGE', name, datetime.now(timezone.utc).isoformat(), flush=True)
        run_stage(out, name)
        classification = classify(out, name)
        with (out / (name + '_assessment.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
            if name == 'screen' and not (out / 'SELECTION.json').exists():
                experiment.select(out)
            filename = 'SCREEN_ASSESSMENT.json' if name == 'screen' else 'HISTORICAL_ASSESSMENT.json'
            if (out / filename).exists():
                experiment.common.all_rows(out)
                result = base.check_seal(base.read_json(out / filename))
            else:
                result = audit.assess(out, name)
        print('EXPECTED_GRAPH_GATES', name, result['performance_gate'],
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
        queue_protocol_sha256=frozen['sha256'], family=FAMILY, stage=name,
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(out / filename),
        classification_sha256=base.digest(classification),
        new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'],
        adopted_neural_fits=0, partition_verification_pending=bool(result['declared_gate']),
        confirmed_final_innovations=0, existing_default_replaced=False)))
    print('EXPECTED_GRAPH_QUEUE_COMPLETE', name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
