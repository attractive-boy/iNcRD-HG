"""Prospective MiDrug graph mechanisms with an exact LncDrug reference identity.

The common likelihood, prediction, and selection rules match v1. New core
modules are explicitly named and hash-verified by each frozen protocol. No
previous implementation, failed result, or selected recipe is modified.
"""
import argparse
import importlib
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import residual_graph_experiment as prior
from improve import nested_anchor_inputs_v2 as nested
from improve import graph_prediction_precision as precision

common, ROOT = prior.common, prior.ROOT
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}
OWN = ('anchor_priority_graph_v1.py', 'test_anchor_priority_graph_v1.py',
       'assess_declared_graph.py', 'graph_prediction_precision.py',
       'residual_graph_experiment.py', 'nested_anchor_inputs_v2.py')


def library(p):
    name = p['implementation_module']
    if not name.isidentifier():
        raise ValueError('Expected a local mechanism module name')
    filename = name + '.py'
    if common.base.digest(ROOT / filename) != p['source_sha256'][filename]:
        raise ValueError('Mechanism implementation changed')
    return importlib.import_module('improve.' + name)


def offset_arrays(cache, task, split, a):
    cache = Path(cache)
    p = nested.protocol(cache)
    dest = cache / 'offsets' / task / str(split)
    r = nested.base.check_seal(nested.base.read_json(dest / 'manifest.json'))
    if r['protocol_sha256'] != p['sha256']:
        raise ValueError('OOF reference protocol mismatch')
    nested.verify_files(dest, r)
    for path, digest in r['fit_records'].items():
        if nested.base.digest(path) != digest:
            raise ValueError('OOF reference fit record changed')
    z = common.load(dest / 'arrays.npz')
    for key in ('pairs_train', 'folds'):
        np.testing.assert_array_equal(a[key], z[key])
    if z['anchor_oof'].shape != (len(a['y_train']),):
        raise ValueError('Wrong OOF reference length')
    return z['anchor_oof'], dest / 'manifest.json'


def prepare(out, family, cache, core):
    out, cache = Path(out).resolve(), Path(cache).resolve()
    if out.exists():
        raise ValueError('Use a new graph residual study directory')
    if core is None:
        raise ValueError('A frozen mechanism core directory is required')
    parent_dir = Path(core).resolve()
    parent = common.protocol(parent_dir)
    if parent['family'] != family:
        raise ValueError('Core family mismatch')
    cache_p = nested.protocol(cache)
    if cache_p['scope'] != 'historical':
        raise ValueError('Historical graph screening requires historical OOF offsets')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        method='anchor_priority_' + family + '_v1', family=family, smoke=False,
        implementation_module=parent['implementation_module'],
        tasks=parent['tasks'], kinds=[parent['primary_kind']] + parent['primary_controls'],
        primary_kind=parent['primary_kind'],
        primary_controls=parent['primary_controls'], core_parameters=parent.get('core_parameters', parent),
        screen_seed=666, confirm_seeds=[0, 1], epochs=400, check_interval=20,
        lrs=[.001, .003], weight_decay=.001, strong_weight=.1, gnn_weight=.1,
        lnc_absolute_tolerance=.001, private_rng_offset=29015,
        parent=str(parent_dir), nested_reference=str(cache),
        lnc_policy=parent['lnc_policy'], identity_records_are_not_neural_fits=True,
        selection=parent['selection'],
        likelihood='MiDrug uses BCEWithLogits(fixed nested OOF strong reference + learned graph residual). Every neural query fold is absent from its reference fit and all nested target-dependent feature supports. Marginal degree readout heads are identically zero and frozen.',
        prediction='MiDrug: unchanged evaluation anchor + 0.1 times mean raw graph residual logits over support folds. LncDrug: exact reference score with zero new residual and no training.',
        null='Zero residual exactly preserves the anchor. The prospectively declared LncDrug identity has no parameters or training steps. Its scoring records do not count as neural fits.',
        gate='Mean MiDrug AUC and AP increase and each improves in >=2 of 3 paired initializations; no LncDrug metric loss exceeds 0.001. Declared primary also beats each declared mechanism control on both mean MiDrug metrics; later fixed partitions and frozen outer recheck remain mandatory.',
        previous_candidate_failures_seen=True, previous_outer_results_seen=True,
        primary_and_controls_fixed_before_screening=True,
        diagnostic_budget='Only the declared primary and mandatory mechanism controls are fitted. All prior failures remain preserved.',
        original_test_source_access=False, this_candidate_outer_outcomes_used=False,
        innovation_count_policy='One per mechanism family, only after all validation gates. Neither the reference infrastructure nor a revision counts separately.',
        confirmed_innovations=0, existing_default_replaced=False,
        source_sha256={**parent['source_sha256'], **{n: common.base.digest(ROOT / n) for n in OWN}},
        inputs={str(parent_dir / 'protocol.json'): common.base.digest(parent_dir / 'protocol.json'),
                str(cache / 'protocol.json'): common.base.digest(cache / 'protocol.json')})
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, out / 'source_snapshot' / name)
    for task in p['tasks']:
        source = prior.INPUTS / task / 'arrays.npz'
        a = common.load(source)
        a = {k: v for k, v in a.items() if not k.startswith('offset_')}
        a['anchor_oof'], manifest = offset_arrays(cache, task, 'historical', a)
        dest = out / 'inputs' / task / 'arrays.npz'
        dest.parent.mkdir(parents=True)
        np.savez_compressed(dest, **a)
        for path in (source, dest, manifest, manifest.parent / 'arrays.npz'):
            p['inputs'][str(path)] = common.base.digest(path)
    common.base.write_json(out / 'protocol.json', p)
    print('FROZEN_ANCHOR_RESIDUAL', family, p['primary_kind'], flush=True)


def setup(a, p, kind, device):
    model, xs, graphs, folds = library(p).setup(a, p['core_parameters'], kind, device)
    for head in (model.rna_degree, model.drug_degree):
        for value in head.parameters():
            torch.nn.init.zeros_(value)
            value.requires_grad_(False)
    return model, xs, graphs, folds


@torch.no_grad()
def predict(model, xs, graphs, pairs, p):
    with precision.deterministic_evaluation():
        model.eval()
        values = []
        for fold, graph in enumerate(graphs):
            values.append(model(xs, graph, pairs).double())
        return torch.stack(values).mean(0)


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke', False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a) - TRAIN_KEYS:
        raise ValueError('Fixed graph residual fitting accepts training inputs only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered graph mechanism')
    if a['anchor_oof'].shape != a['y_train'].shape or not np.isfinite(a['anchor_oof']).all():
        raise ValueError('Invalid nested reference offset')
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
    model, xs, graphs, folds = setup(a, p, kind, device)
    if library(p).identity_task(a, p['core_parameters']):
        value = None if fixed_epochs is not None else np.zeros(len(a['pairs_val'][0]), dtype=np.float64)
        return model, value, [], dict(epoch=0, seconds=0., parameters=0, trainable_parameters=0,
            fixed_step_training=fixed_epochs is not None, mean_raw_residual=True,
            query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True,
            trained_neural_model=False, training_steps=0, gradient_evaluations=0,
            identity_preserves_anchor=True, no_new_lnc_training=True)
    pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(a['y_train'], device=device)
    offset = torch.as_tensor(a['anchor_oof'], device=device, dtype=torch.float64)
    query = (None if fixed_epochs is not None else
             torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device))
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one fitting step is required')
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                                 lr=lr, weight_decay=p['weight_decay'])
    best, selected, state, trace = -np.inf, epochs, None, []
    start = time.monotonic()
    for step in range(1, epochs + 1):
        fold = (step - 1) % len(graphs)
        ids = folds[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual = model(xs, graphs[fold], pairs[:, ids])
        penalty = 0.
        loss = F.binary_cross_entropy_with_logits(offset[ids] + residual.double(), labels[ids]) + penalty
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite anchor residual loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % p['check_interval'] == 0 or step == epochs:
            row = dict(epoch=step, loss=float(loss.detach()))
            if query is not None:
                value = predict(model, xs, graphs, query, p).cpu().numpy()
                composed = a['anchor_' + str(seed)] + p['strong_weight'] * value
                metrics = common.base.metrics(a['y_val'], composed, a['pairs_val'])
                row['anchor_composed'] = metrics
                criterion = (metrics['auc'] + metrics['ap']) / 2
                if criterion > best:
                    best, selected = criterion, step
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if step % 100 == 0:
                    print('TRAIN', p['family'], kind, lr, seed, step,
                          round(metrics['auc'], 9), round(metrics['ap'], 9), flush=True)
            trace.append(row)
    if state is not None:
        model.load_state_dict(state)
    value = None if query is None else predict(model, xs, graphs, query, p).cpu().numpy()
    return model, value, trace, dict(epoch=selected, seconds=time.monotonic() - start,
        parameters=sum(v.numel() for v in model.parameters()),
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None, mean_raw_residual=True,
        query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True,
        trained_neural_model=True, training_steps=epochs, gradient_evaluations=epochs,
        identity_preserves_anchor=False)


def score_arrays(a, residual, seed, weight=.1):
    scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=residual)
    for name in ('gnn', 'tabicl', 'anchor'):
        scores[name] = a[name + '_' + str(seed)]
        scores[name + '_composed'] = scores[name] + weight * residual
    scores['offset'] = scores['anchor'].copy()
    scores['offset_composed'] = scores['anchor'].copy()
    return scores


def run(out, task, device, stage):
    out = Path(out)
    p = common.protocol(out)
    a = common.load(out / 'inputs' / task / 'arrays.npz')
    ph = common.base.digest(out / 'protocol.json')
    selected = None if stage == 'screen' else nested.base.read_json(out / 'SELECTION.json')
    if selected is not None and selected['protocol_sha256'] != ph:
        raise ValueError('Candidate selection protocol mismatch')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for kind in p['kinds']:
        for lr in p['lrs'] if stage == 'screen' else [selected['chosen'][kind]['lr']]:
            for seed in [p['screen_seed']] if stage == 'screen' else p['confirm_seeds']:
                dest = out / 'runs' / task / kind / ('lr' + str(lr)) / ('s' + str(seed))
                if (dest / 'result.json').exists():
                    common.verify(dest, ph)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                model, value, trace, detail = fit(a, p, kind, lr, seed, device)
                torch.save(model.state_dict(), dest / 'model.pt')
                common.base.write_json(dest / 'trace.json', trace)
                loaded, xs, graphs, _ = setup(a, p, kind, device)
                loaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
                query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
                replay = predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
                chunks = np.concatenate([predict(loaded, xs, graphs, query[:, i:i + 800], p).cpu().numpy()
                    for i in range(0, query.shape[1], 800)])
                for actual in (replay, chunks):
                    np.testing.assert_allclose(value, actual, atol=3e-5, rtol=3e-5)
                scores = score_arrays(a, value, seed, p['strong_weight'])
                np.savez_compressed(dest / 'scores.npz', **scores)
                result = dict(task=task, kind=kind, lr=lr, seed=seed, **detail,
                    protocol_sha256=ph, standalone_is_residual_ranking_not_absolute_predictor=True,
                    replay_max_abs=float(max(abs(value - replay).max(), abs(value - chunks).max())),
                    metrics={n: common.base.metrics(scores['y'], v, scores['pairs']) for n, v in scores.items()
                             if n not in ('y', 'pairs', 'offset', 'offset_composed')},
                    offset_metrics={n: common.base.metrics(scores['y'], scores[n], scores['pairs'])
                                    for n in ('offset', 'offset_composed')},
                    files={n: common.base.digest(dest / n) for n in ('model.pt', 'trace.json', 'scores.npz')})
                common.base.write_json(dest / 'result.json', result)
                print('VERIFIED_ANCHOR_RESIDUAL', p['family'], task, kind, lr, seed, detail['epoch'],
                      result['metrics']['anchor_composed'], flush=True)
                del model, loaded, xs, graphs
                gc.collect()
                torch.cuda.empty_cache()


def select(out):
    out = Path(out)
    if (out / 'SELECTION.json').exists():
        raise ValueError('Selection is already frozen')
    p = common.protocol(out)
    table = common.all_rows(out)
    chosen = {}
    for kind in p['kinds']:
        candidates = []
        for lr in p['lrs']:
            differences = {}
            for task in p['tasks']:
                rows = table[(table.task == task) & (table.kind == kind) &
                    (table.lr == lr) & (table.seed == p['screen_seed'])].set_index('score')
                differences[task] = {m: float(rows.loc['anchor_composed', m] - rows.loc['anchor', m])
                                     for m in ('auc', 'ap')}
            gate = all(differences['MiDrug'][m] > 0 and
                       differences['LncDrug'][m] >= -p['lnc_absolute_tolerance'] for m in ('auc', 'ap'))
            candidates.append(dict(lr=lr, gate=gate, differences=differences,
                                   priority=sum(differences['MiDrug'].values()) / 2))
        chosen[kind] = max(candidates, key=lambda x: (x['gate'], x['priority'], -x['lr']))
    result = dict(protocol_sha256=common.base.digest(out / 'protocol.json'), chosen=chosen,
        primary_kind=p['primary_kind'], passes_primary_screen=chosen[p['primary_kind']]['gate'],
        screen_only=True, confirmed_innovations=0, original_test_source_access=False)
    common.base.write_json(out / 'SELECTION.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'screen', 'select', 'confirm'))
    parser.add_argument('--family', required=True)
    parser.add_argument('--core')
    parser.add_argument('--out')
    parser.add_argument('--cache', default=str(nested.DEFAULT))
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    out = args.out or str(ROOT / 'results' / ('anchor_priority_' + args.family + '_v1_20260916'))
    if args.command == 'prepare':
        prepare(out, args.family, args.cache, args.core)
    elif args.command == 'select':
        select(out)
    else:
        run(out, args.task, args.device, args.command)
