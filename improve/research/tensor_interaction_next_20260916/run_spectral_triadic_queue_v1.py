"""Execute the frozen spectral-triadic screen and eligible confirmation serially."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_spectral_triadic_experiment_v1 as experiment
from improve import spectral_triadic_graph_v1 as core
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_spectral_triadic_v1_20260916'


def audit_record(path, r, a, p):
    """CPU-only read-only checks, shared with the independent metric/replay audit."""
    trace = base.read_json(path / 'trace.json')
    d = r['triadic_diagnostics']
    if not r['trained_neural_model']:
        assert r['task'] == 'LncDrug' and r['identity_preserves_anchor'] and not trace
        assert all(r[key] == 0 for key in ('epoch', 'parameters', 'trainable_parameters',
            'fixed_parameters', 'training_steps', 'gradient_evaluations', 'optimizer_steps'))
        assert d == core.empty_diagnostics() and not (path / 'training_final.pt').exists()
        scores = experiment.common.load(path / 'scores.npz')
        np.testing.assert_array_equal(scores['standalone'], np.zeros_like(scores['standalone']))
        np.testing.assert_array_equal(scores['anchor'], scores['anchor_composed'])
        return None
    assert r['task'] == 'MiDrug' and r['trainable_parameters'] > 0
    assert r['training_steps'] == r['gradient_evaluations'] == r['optimizer_steps'] == 400
    assert len(trace) == 20 and [row['epoch'] for row in trace] == list(range(20, 401, 20))
    folds = p['core_parameters']['folds']
    width = p['core_parameters']['width']
    monomials = (2*width)*(2*width+1)//2 + width*(width+1)//2
    query_count = a['pairs_val'].shape[1]
    fold_counts = np.bincount(a['folds'], minlength=folds)
    def count_at(step, selections):
        trained = sum(int(fold_counts[index % folds]) for index in range(step))
        return trained + selections*folds*query_count
    assert d['calls'] == 400 + folds*21
    assert d['layer_calls'] == 2*d['calls']
    assert d['query_rows'] == count_at(400, 21)
    assert d['quadratic_terms'] == monomials*d['query_rows']
    assert all(np.isfinite(v) for v in d.values())
    for row in trace:
        step, diagnostic = row['epoch'], row['triadic_diagnostics']
        assert np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
        assert diagnostic['calls'] == step + folds*(step//20)
        assert diagnostic['layer_calls'] == 2*diagnostic['calls']
        assert diagnostic['query_rows'] == count_at(step, step//20)
        assert diagnostic['quadratic_terms'] == monomials*diagnostic['query_rows']
    final = torch.load(path / 'training_final.pt', map_location='cpu', weights_only=True)
    selected = torch.load(path / 'model.pt', map_location='cpu', weights_only=True)
    assert final['steps'] == 400
    assert all(torch.isfinite(value).all() for value in final['model'].values())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(r['seed'])
        initial = core.SpectralTriadicGraph([a['x_'+str(i)].shape[1] for i in range(len(a['node_counts']))],
            int(a['relation'].max())+1, p['core_parameters'], r['kind'])
    names = [name for name, value in initial.named_parameters() if value.requires_grad]
    frozen = [name for name, value in initial.named_parameters() if not value.requires_grad]
    assert final['optimizer_parameter_names'] == names and final['frozen_parameter_names'] == frozen
    assert r['parameters'] == sum(value.numel() for value in initial.parameters())
    assert r['trainable_parameters'] == sum(value.numel() for value in initial.parameters() if value.requires_grad)
    assert r['fixed_parameters'] == r['parameters'] - r['trainable_parameters']
    for name in frozen:
        torch.testing.assert_close(final['model'][name], initial.state_dict()[name], atol=0., rtol=0.)
        torch.testing.assert_close(selected[name], initial.state_dict()[name], atol=0., rtol=0.)
    indexes = final['optimizer']['param_groups'][0]['params']
    assert len(names) == len(indexes) == len(final['optimizer']['state'])
    for name, index in zip(names, indexes):
        state = final['optimizer']['state'][index]
        assert int(state['step']) == 400
        assert all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor))
        if r['kind'] == 'linear_only' and 'triadic' in name:
            assert torch.count_nonzero(state['exp_avg']) == 0
            assert torch.count_nonzero(state['exp_avg_sq']) == 0
    return {key: final[key] for key in ('cpu_rng', 'cuda_rng')}


def run_stage(name):
    for task in ('MiDrug', 'LncDrug'):
        with (OUT / (name + '_' + task + '.log')).open('a', buffering=1) as stream:
            command = [sys.executable, '-u', '-m', 'improve.anchor_spectral_triadic_experiment_v1',
                name, '--out', str(OUT), '--task', task, '--device', 'cuda:1']
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError('Spectral-triadic worker failed: ' + str((name, task, completed.returncode)))


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
        family='spectral_triadic', protocol_sha256=ph, cumulative_new_neural_fits=len(neural),
        cumulative_identity_scoring_records=len(identity), adopted_neural_fits=0,
        neural=neural, identity=identity, formal_neural_steps=400,
        identity_records_count_as_training=False, gradients_per_step=1,
        diagnostics_final_optimizer_frozen_parameters_and_rng_verified=True, confirmed_final_innovations=0))
    path = HERE / ('SPECTRAL_TRIADIC_RECORD_CLASSIFICATION_' + name.upper() + '_V1.json')
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
    frozen = base.check_seal(base.read_json(HERE / 'SPECTRAL_TRIADIC_QUEUE_PROTOCOL_V1.json'))
    assert frozen['family'] == 'spectral_triadic' and frozen['source_sha256'] == base.digest(__file__)
    for path, digest in frozen['input_sha256'].items():
        assert base.digest(path) == digest, path
    experiment.common.protocol(OUT)
    destination = HERE / 'SPECTRAL_TRIADIC_QUEUE_RESULT_V1.json'
    if destination.exists():
        raise ValueError('Preserve completed queue result')
    for name in ('screen', 'confirm'):
        print('START_SPECTRAL_TRIADIC_STAGE', name, datetime.now(timezone.utc).isoformat(), flush=True)
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
        print('SPECTRAL_TRIADIC_GATES', name, result['performance_gate'],
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
        queue_protocol_sha256=frozen['sha256'], family='spectral_triadic', stage=name,
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(OUT / filename),
        classification_sha256=base.digest(classification), new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'], adopted_neural_fits=0,
        partition_verification_pending=bool(result['declared_gate']), confirmed_final_innovations=0,
        existing_default_replaced=False)))
    print('SPECTRAL_TRIADIC_QUEUE_COMPLETE', name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
