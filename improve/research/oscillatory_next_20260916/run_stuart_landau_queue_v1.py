"""Execute the frozen Stuart-Landau screen and eligible confirmation serially."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_stuart_landau_experiment_v1 as experiment
from improve import stuart_landau_graph_v1 as core
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_stuart_landau_v1_20260916'


COUNT_KEYS = ('calls', 'local_steps', 'radial_solves', 'solver_entries', 'coupling_messages')


def counts_for(a, p, step, evaluation_passes):
    """Independent CPU count from each fold's constructed physical support."""
    cp = p['core_parameters']
    folds, depth, width = cp['folds'], cp['depth'], cp['width']
    nodes, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    edge_counts = []
    for fold in range(folds):
        edges, labels = core.graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == fold))
        physical = core.graph_lib.graph_inputs(edges, labels,
            np.empty((2, 0), dtype=np.int64), nodes, relations, 'cpu')
        edge_counts.append(physical[0].shape[1])
    calls = step + folds * evaluation_passes
    messages = depth * sum(count * (step // folds + int(fold < step % folds) + evaluation_passes)
                           for fold, count in enumerate(edge_counts))
    return dict(calls=calls, local_steps=depth*calls, radial_solves=depth*calls,
        solver_entries=depth*calls*nodes*width, coupling_messages=messages)


def audit_record(path, r, a, p):
    """CPU-only read-only budget, finite-state, optimizer and identity checks."""
    trace = base.read_json(path / 'trace.json')
    d = r['oscillator_diagnostics']
    if not r['trained_neural_model']:
        assert r['task'] == 'LncDrug' and r['identity_preserves_anchor'] and not trace
        assert all(r[key] == 0 for key in ('epoch', 'parameters', 'trainable_parameters',
            'fixed_parameters', 'training_steps', 'gradient_evaluations', 'optimizer_steps'))
        assert d == core.empty_diagnostics() and not (path / 'training_final.pt').exists()
        assert not torch.load(path / 'model.pt', map_location='cpu', weights_only=True)
        scores = experiment.common.load(path / 'scores.npz')
        np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
        np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
        return None
    assert r['task'] == 'MiDrug' and r['trainable_parameters'] > 0
    assert r['training_steps'] == r['gradient_evaluations'] == r['optimizer_steps'] == 400
    assert r['final_optimizer_state_epoch'] == 400
    assert len(trace) == 20 and [row['epoch'] for row in trace] == list(range(20, 401, 20))
    assert {key: d[key] for key in COUNT_KEYS} == counts_for(a, p, 400, 21)
    tolerance = p['core_parameters']['radial_residual_tolerance']
    assert all(np.isfinite(v) for v in d.values())
    assert d['maximum_radial_relative_residual'] <= tolerance
    for row in trace:
        step, diagnostic = row['epoch'], row['oscillator_diagnostics']
        assert np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
        assert {key: diagnostic[key] for key in COUNT_KEYS} == counts_for(a, p, step, step//20)
        assert all(np.isfinite(v) for v in diagnostic.values())
        assert diagnostic['maximum_radial_relative_residual'] <= tolerance
    final = torch.load(path / 'training_final.pt', map_location='cpu', weights_only=True)
    selected = torch.load(path / 'model.pt', map_location='cpu', weights_only=True)
    assert final['steps'] == 400
    assert all(torch.isfinite(value).all() for value in final['model'].values())
    assert all(torch.isfinite(value).all() for value in selected.values())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(r['seed'])
        initial = core.StuartLandauGraph([a['x_'+str(i)].shape[1] for i in range(len(a['node_counts']))],
            int(a['relation'].max())+1, p['core_parameters'], r['kind'])
    names = [name for name, value in initial.named_parameters() if value.requires_grad]
    frozen = [name for name, value in initial.named_parameters() if not value.requires_grad]
    assert final['optimizer_parameter_names'] == names and final['frozen_parameter_names'] == frozen
    assert not frozen and r['fixed_parameters'] == 0
    assert r['parameters'] == sum(value.numel() for value in initial.parameters())
    assert r['trainable_parameters'] == r['parameters'] == 114369
    groups = final['optimizer']['param_groups']
    assert len(groups) == 1 and groups[0]['lr'] == r['lr']
    assert groups[0]['weight_decay'] == p['weight_decay']
    indexes = groups[0]['params']
    assert len(names) == len(indexes) == len(final['optimizer']['state'])
    for name, index in zip(names, indexes):
        state = final['optimizer']['state'][index]
        assert int(state['step']) == 400
        assert all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor))
        assert state['exp_avg'].shape == state['exp_avg_sq'].shape == initial.state_dict()[name].shape
    return {key: final[key] for key in ('cpu_rng', 'cuda_rng')}


def run_stage(name):
    for task in ('MiDrug', 'LncDrug'):
        with (OUT / (name + '_' + task + '.log')).open('a', buffering=1) as stream:
            command = [sys.executable, '-u', '-m', 'improve.anchor_stuart_landau_experiment_v1',
                name, '--out', str(OUT), '--task', task, '--device', 'cuda:1']
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError('Stuart-Landau worker failed: ' + str((name, task, completed.returncode)))


def classify(name):
    p, ph = experiment.common.protocol(OUT), base.digest(OUT / 'protocol.json')
    expected = {(task, kind, lr, p['screen_seed']) for task in p['tasks']
                for kind in p['kinds'] for lr in p['lrs']}
    if name == 'confirm':
        selection = base.read_json(OUT / 'SELECTION.json')
        expected |= {(task, kind, selection['chosen'][kind]['lr'], seed) for task in p['tasks']
                     for kind in p['kinds'] for seed in p['confirm_seeds']}
    found, neural, identity, rngs = set(), [], [], {}
    arrays = {task: experiment.common.load(OUT / 'inputs' / task / 'arrays.npz') for task in p['tasks']}
    for path in sorted((OUT / 'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent, ph)
        found.add((r['task'], r['kind'], r['lr'], r['seed']))
        assert not r.get('adopted_without_new_training', False)
        rng = audit_record(path.parent, r, arrays[r['task']], p)
        item = dict(path=str(path), sha256=base.digest(path), task=r['task'],
                    seed=r['seed'], kind=r['kind'], lr=r['lr'])
        if rng is not None:
            key = (r['lr'], r['seed'])
            if key in rngs:
                for direction in ('cpu_rng', 'cuda_rng'):
                    torch.testing.assert_close(rng[direction], rngs[key][direction], atol=0., rtol=0.)
            else:
                rngs[key] = rng
            neural.append(item)
        else:
            identity.append(item)
    count = 6 if name == 'screen' else 12
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or unexpected completed records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=name,
        family='stuart_landau', protocol_sha256=ph, cumulative_new_neural_fits=len(neural),
        cumulative_identity_scoring_records=len(identity), adopted_neural_fits=0,
        neural=neural, identity=identity, formal_neural_steps=400,
        identity_records_count_as_training=False, gradients_per_step=1,
        oscillator_counts_final_optimizer_parameters_and_rng_verified=True, confirmed_final_innovations=0))
    path = HERE / ('STUART_LANDAU_RECORD_CLASSIFICATION_' + name.upper() + '_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        assert old['neural'] == result['neural'] and old['identity'] == result['identity']
    else:
        base.write_json(path, result)
    return path


def main():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Declared deterministic CuBLAS environment required')
    torch.set_num_threads(4)
    frozen = base.check_seal(base.read_json(HERE / 'STUART_LANDAU_QUEUE_PROTOCOL_V1.json'))
    assert frozen['family'] == 'stuart_landau' and frozen['source_sha256'] == base.digest(__file__)
    for path, digest in frozen['input_sha256'].items():
        assert base.digest(path) == digest, path
    experiment.common.protocol(OUT)
    destination = HERE / 'STUART_LANDAU_QUEUE_RESULT_V1.json'
    if destination.exists():
        raise ValueError('Preserve completed queue result')
    for name in ('screen', 'confirm'):
        print('START_STUART_LANDAU_STAGE', name, datetime.now(timezone.utc).isoformat(), flush=True)
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
        print('STUART_LANDAU_GATES', name, result['performance_gate'],
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
        queue_protocol_sha256=frozen['sha256'], family='stuart_landau', stage=name,
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(OUT / filename),
        classification_sha256=base.digest(classification), new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'], adopted_neural_fits=0,
        partition_verification_pending=bool(result['declared_gate']), confirmed_final_innovations=0,
        existing_default_replaced=False)))
    print('STUART_LANDAU_QUEUE_COMPLETE', name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
