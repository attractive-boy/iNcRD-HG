"""Replay nine selected models, stage-zero choices and both RMSprop states."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, CLEAN, load_inner, seed_all, sha256
from model import CONFIGS, make_adapter, make_optimizer, source_record, reconstruction_weight
from check import deletion_check, QUERY_CHECK_SOURCE

PREVIOUS = HERE.parent / 'midrug_training_size_20261007'


def saved_scores(path, data, record):
    with np.load(path.with_name(path.stem + '_inner.npz')) as z:
        for key, data_key in [('pairs', 'val_pairs'), ('y', 'val_y'), ('indices', 'val_indices')]:
            np.testing.assert_array_equal(z[key], data[data_key])
        p = z['probability'].copy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert abs(roc_auc_score(data['val_y'], p) - record['inner_auc']) < 1e-12
    assert abs(average_precision_score(data['val_y'], p) - record['inner_ap']) < 1e-12
    return p


@torch.no_grad()
def batch_check(a):
    together = a.logits('val')[:3]
    alone = torch.cat([a.model(a.pairs['val'][:, i:i+1], a.ps,
                              a.qis['val'][i:i+1], a.ease['val'][i:i+1], a.dense)
                       for i in range(3)])
    torch.testing.assert_close(together, alone, atol=2e-4, rtol=5e-5)
    return float(abs(together - alone).max())


def optimizer_check(a, name, states, record):
    """Discover parameters connected to each objective; account for unused tokens."""
    parameters = dict(a.model.named_parameters())
    before = {k: v.detach().clone() for k, v in a.model.state_dict().items()}
    primary = F.binary_cross_entropy_with_logits(a.logits('train'), a.y)
    indices = {n: torch.arange(a.graph[n].num_nodes, device=a.y.device) for n in ('miRNA', 'drug')}
    auxiliary, _ = a.model.reconstruction.loss(a.model.encoder, a.graph, indices, False)
    used = []
    for loss in (primary, auxiliary):
        grad = torch.autograd.grad(loss, list(parameters.values()), allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grad)
        used.append({n for n, g in zip(parameters, grad) if g is not None})
    template = make_optimizer(a.model)
    template_state = template.state_dict()
    by_object = {id(p): n for n, p in parameters.items()}
    names = {pid: by_object[id(p)]
             for group, saved in zip(template.param_groups, template_state['param_groups'])
             for p, pid in zip(group['params'], saved['params'])}
    checks = {}
    for part, count in [('selected', record['best_epoch']), ('final', record['epochs_run'])]:
        state = states[part]
        assert state['param_groups'] == template_state['param_groups']
        auxiliary_steps = count if name == 'keep' else (0 if name == 'off' else min(count, 99))
        expected = {pid: (count if n in used[0] else auxiliary_steps if n in used[1] else 0)
                    for pid, n in names.items()}
        assert set(state['state']) == {pid for pid, steps in expected.items() if steps > 0}
        step_counts = {}
        for pid, saved in state['state'].items():
            assert set(saved) == {'step', 'square_avg'}
            assert int(saved['step']) == expected[pid]
            avg = saved['square_avg']
            assert avg.shape == parameters[names[pid]].shape
            assert torch.isfinite(avg).all() and (avg >= 0).all()
            step_counts[names[pid]] = int(saved['step'])
        checks[part] = dict(main_updates=count, auxiliary_updates=auxiliary_steps,
                           state_parameter_count=len(step_counts), step_counts=step_counts)
    for key, value in before.items():
        torch.testing.assert_close(value, a.model.state_dict()[key], atol=0, rtol=0)
    checks['unused_parameters'] = sorted(set(parameters) - used[0] - used[1])
    assert all(n.startswith('reconstruction.token.') for n in checks['unused_parameters'])
    return checks


def main():
    plan = json.loads((HERE / 'results/plan.json').read_text())
    for path, digest in {**plan['source_hashes'], **plan['evidence_hashes']}.items():
        assert sha256(path) == digest
    prior = json.loads((PREVIOUS / 'verification.json').read_text())
    assert prior['status'] == 'passed'
    prior_by_path = {r['metadata_path']: r for r in prior['records']}
    records = []
    for fold in range(3):
        data = load_inner(fold)
        for name, sources in plan['controls'].items():
            src = sources[str(fold)]; path = Path(src['path']); r = json.loads(path.read_text())
            old = prior_by_path[str(path)]
            assert old['status'] == 'passed' and old['existing_full_size_model']
            assert sha256(path) == src['metadata_sha256'] == old['metadata_sha256']
            assert sha256(path.with_suffix('.pt')) == r['checkpoint_sha256'] == src['checkpoint_sha256'] == old['checkpoint_sha256']
            assert sha256(path.with_name(path.stem + '_inner.npz')) == src['probability_sha256']
            assert sha256(PREVIOUS / 'replay_predictions' / (old['tag'] + '.npz')) == old['replay_predictions_sha256']
            assert r['data_sha256'] == plan['data_sha256'] and r['graph_sha256'] == plan['graph_sha256']
            saved_scores(path, data, r)
            records.append(dict(tag=path.stem, metadata_path=str(path), status='passed', reused_control=True,
                                prior_replay_tag=old['tag'], verification='Immutable prior GPU replay; current hashes, pairs, labels, indices and saved metric checks'))
        for name in CONFIGS:
            path = HERE / 'results' / f'{name}_f{fold}_s42.json'
            r = json.loads(path.read_text()); source, origin = source_record(fold)
            assert r['config'] == name and r['fold'] == fold and r['seed'] == 42
            assert r['source_hashes'] == plan['source_hashes']
            assert r['source_metadata'] == str(source)
            assert r['source_metadata_sha256'] == sha256(source)
            assert r['source_checkpoint_sha256'] == origin['checkpoint_sha256']
            assert r['source_best_epoch'] == origin['best_epoch']
            assert r['graph_sha256'] == plan['graph_sha256'] == sha256(CLEAN / 'graph.pt')
            assert r['data_sha256'] == plan['data_sha256'] == sha256(CLEAN / 'inner_data.npz')
            assert r['no_lncrna'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
            assert sha256(path.with_suffix('.pt')) == r['checkpoint_sha256']
            assert sha256(path.with_name(path.stem + '_optimizers.pt')) == r['optimizers_sha256']
            curves = json.loads(path.with_name(path.stem + '_curves.json').read_text())
            assert len(curves) == r['epochs_run'] + 1
            assert max(curves, key=lambda x: x['selection_score'])['stage_epoch'] == r['best_epoch']
            assert curves[r['best_epoch']]['selection_score'] == r['selection_score']
            assert 100 <= r['epochs_run'] <= 300
            assert r['epochs_run'] == 300 or r['epochs_run'] - r['best_epoch'] >= 100
            for ep, row in enumerate(curves):
                assert row['stage_epoch'] == ep and row['optimizer_update'] == (ep > 0)
                assert abs(row['selection_score'] - .5 * (row['inner_auc'] + row['inner_ap'])) < 1e-12
                assert all(np.isfinite(row[k]) and 0 <= row[k] <= 1 for k in ('inner_auc', 'inner_ap'))
                if ep == 0:
                    assert row['inner_auc'] == origin['inner_auc'] and row['inner_ap'] == origin['inner_ap']
                    assert not any(k in row for k in ('loss', 'bce', 'reconstruction', 'weight'))
                else:
                    assert row['weight'] == reconstruction_weight(name, ep)
                    assert all(np.isfinite(row[k]) and row[k] >= 0 for k in ('bce', 'reconstruction', 'loss', 'weight'))
                    assert row['loss'] == float(np.float32(row['bce']) + np.float32(row['weight']) * np.float32(row['reconstruction']))
            seed_all(42); a = make_adapter(data, name, 'cuda:0')
            state = torch.load(path.with_suffix('.pt'), map_location='cuda:0', weights_only=True)
            a.model.load_state_dict(state); a.model.eval()
            assert set(a.graph.node_types) == {'miRNA', 'drug'} and all('lnc' not in k.lower() for k in state)
            assert all(torch.isfinite(v).all() for v in state.values())
            assert int(a.model.eb.num_batches_tracked) == r['source_best_epoch'] + r['best_epoch']
            if r['best_epoch'] == 0:
                original_state = torch.load(source.with_suffix('.pt'), map_location='cuda:0', weights_only=True)
                assert set(original_state) == set(state)
                for key in state: torch.testing.assert_close(state[key], original_state[key], atol=0, rtol=0)
                del original_state
            stored = saved_scores(path, data, r)
            with torch.no_grad(): probability = a.predict('val')
            np.testing.assert_allclose(probability, stored, atol=2e-5, rtol=2e-5)
            # Torch 1.13's weights-only unpickler lacks BINFLOAT (optimizer group
            # scalars). This file was generated by our hash-verified local train.py.
            states = torch.load(path.with_name(path.stem + '_optimizers.pt'), map_location='cpu', weights_only=False)
            assert set(states) == {'selected', 'final'}
            optimizers = optimizer_check(a, name, states, r)
            rec = dict(tag=path.stem, status='passed', reused_control=False, metadata_path=str(path),
                       best_stage_epoch=r['best_epoch'], epochs_run=r['epochs_run'],
                       max_prediction_error=float(abs(probability - stored).max()), optimizers=optimizers,
                       explicit_deletions=deletion_check(a, data, name, count=4), batch_max_logit_error=batch_check(a))
            bare = make_adapter(data, 'control', 'cuda:0')
            bare.model.load_state_dict({k: v for k, v in state.items() if not k.startswith('reconstruction.')}); bare.model.eval()
            with torch.no_grad(): plain = bare.predict('val')
            np.testing.assert_allclose(probability, plain, atol=2e-5, rtol=2e-5)
            rec['without_reconstruction_error'] = float(abs(probability - plain).max())
            records.append(rec); print('PASS', path.stem, flush=True)
            del a, bare, state, states; torch.cuda.empty_cache()
    assert len(records) == 15 and sum(not r['reused_control'] for r in records) == 9
    (HERE / 'verification.json').write_text(json.dumps(dict(status='passed', records=records,
        new_checkpoint_replays=9, immutable_reused_control_checks=6,
        audit_sources={str(p): sha256(p) for p in [HERE / 'audit.py', HERE / 'check.py', QUERY_CHECK_SOURCE]},
        no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False), indent=2))


if __name__ == '__main__': main()
