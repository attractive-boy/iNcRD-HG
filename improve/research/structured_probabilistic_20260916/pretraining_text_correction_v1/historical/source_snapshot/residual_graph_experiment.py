"""Graph mechanisms fitted around a support-only observation likelihood.

This is a new likelihood study after the original graph candidates failed
partition checks. The classical offset is shared by every mechanism control
and is itself a mandatory null; it is never counted as an innovation.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve import lorentz_graph_experiment as geometry
from improve import state_space_experiment as state_space
from improve import graph_prediction_precision as precision
from improve import graph_observation_offset as offset

ROOT = common.base.ROOT
INPUTS = ROOT / 'results/observation_offset_inputs_20260915'
LIBRARIES = dict(block=common, geometry=geometry, state_space=state_space)
PARENTS = dict(block=ROOT / 'results/overlap_block_precision_v2_20260915',
               geometry=ROOT / 'results/lorentz_graph_precision_v2_20260915',
               state_space=state_space.OUT)
PRIMARY = dict(block='full', geometry='attention_only', state_space='full')
CONTROLS = dict(block=['deterministic'], geometry=['euclidean'], state_space=['static', 'no_state'])
TRAIN_KEYS = frozenset(('pairs_train', 'y_train', 'edges', 'relation', 'target_codes',
    'node_counts', 'pair_offset', 'nd', 'folds', 'x_0', 'x_1', 'x_2', 'x_3',
    'offset_nodes', 'offset_intercepts'))


def prepare_inputs():
    if INPUTS.exists():
        record = json.loads((INPUTS / 'manifest.json').read_text())
        for name, expected in record['source_sha256'].items():
            assert common.base.digest(ROOT / name) == expected
        for path, expected in record['files'].items():
            assert common.base.digest(path) == expected
        return record
    INPUTS.mkdir(parents=True)
    sources = ('graph_observation_offset.py', 'test_graph_observation_offset.py', 'residual_graph_experiment.py')
    record = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), penalty=1., folds=5,
        method='Gaussian-prior row/column logistic MAP, intercept unpenalized, fitted independently inside each support fold.',
        query_outcomes_used_for_offset_fitting=False, classical_nuisance_not_innovation=True,
        source_sha256={n: common.base.digest(ROOT / n) for n in sources}, files={}, diagnostics={})
    (INPUTS / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(ROOT / name, INPUTS / 'source_snapshot' / name)
    for task in common.base.TASKS:
        source = common.OUT / 'inputs' / task / 'arrays.npz'
        a = common.load(source)
        nodes, intercepts, diagnostics = offset.context_offsets(a)
        a.update(offset_nodes=nodes, offset_intercepts=intercepts)
        a['offset_val'] = offset.offset_prediction(a, a['pairs_val'])
        dest = INPUTS / task / 'arrays.npz'
        dest.parent.mkdir(parents=True)
        np.savez_compressed(dest, **a)
        record['diagnostics'][task] = diagnostics
        for path in (source, dest):
            record['files'][str(path)] = common.base.digest(path)
    common.base.write_json(INPUTS / 'manifest.json', record)
    return record


def prepare(out, family):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new residual experiment directory')
    parent = common.protocol(PARENTS[family])
    prepare_inputs()
    own = ('graph_observation_offset.py', 'test_graph_observation_offset.py',
           'residual_graph_experiment.py', 'test_residual_graph_experiment.py')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), method='residual_' + family,
        family=family, smoke=False, tasks=parent['tasks'], kinds=parent['kinds'],
        screen_seed=666, confirm_seeds=[0, 1], epochs=400, check_interval=20,
        lrs=[.001, .003], weight_decay=.001, strong_weight=.1, gnn_weight=.5,
        lnc_absolute_tolerance=.001, primary_kind=PRIMARY[family], primary_controls=CONTROLS[family],
        core_parameters=parent, parent=str(PARENTS[family]),
        selection=parent['selection'],
        gate='MiDrug mean AUC/AP increase and each positive in >=2/3 confirmations; every LncDrug loss <=0.001. Primary must beat the offset-only null and every prespecified primary mechanism control on both mean MiDrug metrics. All remaining component alternatives are reported but not counted as additional techniques.',
        success_scope='New residual-likelihood candidate after prior development failures; no claim that the old frozen candidates passed. Additional partition and reserved evaluation are still required.',
        likelihood='A support-fold-only row/column logistic model replaces the learned marginal degree heads. Core graph interaction logits add to its fixed score. Offset regularization is fixed at 1.0, no offset hyperparameter search. Both observation classes come from the benchmark; zeros are not known biological negatives.',
        controls='Every arm sees identical fitted offsets and raw graph inputs. The offset-only prediction is a mandatory null. Graph architectures and their stated component controls retain the predecessor definitions. Main comparisons isolate posterior versus deterministic representation, curved versus Euclidean operations, or selective state memory versus static/no-state operations.',
        boundary='All positive and zero fitting query labels are excluded from their offset fit. Graph query-positive deletion remains before degree/reverse/propagation. Validation outcomes do not enter offsets or graph supports. Original 400-step block KL annealing denominator is retained.',
        numerical_policy='Float32 neural computation, fixed FP64 observation offset added to logits, deterministic probability evaluation. Private sampling seed is model seed + 29015.',
        prior_partition_failures_seen=True, existing_default_replaced=False,
        original_test_access=False, original_outer_query_label_access=False, confirmed_innovations=0,
        source_sha256={**parent['source_sha256'], **{n: common.base.digest(ROOT / n) for n in own}},
        inputs={str(PARENTS[family] / 'protocol.json'): common.base.digest(PARENTS[family] / 'protocol.json'),
                str(INPUTS / 'manifest.json'): common.base.digest(INPUTS / 'manifest.json')})
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, out / 'source_snapshot' / name)
    for task in p['tasks']:
        source = INPUTS / task / 'arrays.npz'
        dest = out / 'inputs' / task / 'arrays.npz'
        dest.parent.mkdir(parents=True)
        shutil.copy2(source, dest)
        for path in (source, dest):
            p['inputs'][str(path)] = common.base.digest(path)
    common.base.write_json(out / 'protocol.json', p)
    print('FROZEN_RESIDUAL', family, out, flush=True)


def setup(a, p, kind, device):
    core, xs, graphs, folds = LIBRARIES[p['family']].setup(a, p['core_parameters'], kind, device)
    graphs = [(graph, torch.as_tensor(a['offset_nodes'][i], device=device),
               torch.as_tensor(a['offset_intercepts'][i], device=device)) for i, graph in enumerate(graphs)]
    return offset.OffsetGraph(core, p['family']), xs, graphs, folds


def predict(model, xs, graphs, pairs, p):
    if p['family'] == 'block':
        return precision.block_prediction(model, xs, graphs, pairs, p['core_parameters']['posterior_samples'])
    return precision.geometry_prediction(model, xs, graphs, pairs)


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if fixed_epochs is not None and set(a) - TRAIN_KEYS:
        raise ValueError('Fixed residual fitting accepts training inputs only')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, xs, graphs, folds = setup(a, p, kind, device)
    pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(a['y_train'], device=device)
    query = None if fixed_epochs is not None else torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
    generator = torch.Generator(device=device).manual_seed(seed + 29015)
    best, selected_epoch, state = -np.inf, epochs, None
    start, trace = time.monotonic(), []
    for step in range(1, epochs + 1):
        fold = (step - 1) % len(graphs)
        ids = folds[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if p['family'] == 'block':
            values, kl = model(xs, graphs[fold], pairs[:, ids], generator, p['core_parameters']['temperature'])
            loss = F.binary_cross_entropy_with_logits(values, labels[ids]) + \
                p['core_parameters']['beta'] * min(1., step / (p['epochs'] * .2)) * kl
        else:
            loss = F.binary_cross_entropy_with_logits(model(xs, graphs[fold], pairs[:, ids]), labels[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite residual graph likelihood')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % p['check_interval'] == 0 or step == epochs:
            row = dict(epoch=step, loss=float(loss.detach()))
            if query is not None:
                value = predict(model, xs, graphs, query, p).cpu().numpy()
                composed = (1 - p['strong_weight']) * a['anchor_' + str(seed)] + p['strong_weight'] * value
                mm = common.base.metrics(a['y_val'], composed, a['pairs_val'])
                raw = common.base.metrics(a['y_val'], value, a['pairs_val'])
                row.update(anchor_composed=mm, standalone=raw)
                criterion = (mm['auc'] + mm['ap']) / 2
                if criterion > best:
                    best, selected_epoch = criterion, step
                    state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
                if step % max(100, p['check_interval']) == 0:
                    print('TRAIN', kind, lr, seed, step, round(raw['auc'], 6), round(raw['ap'], 6),
                          round(time.monotonic() - start, 1), flush=True)
            trace.append(row)
    if state is not None:
        model.load_state_dict(state)
    values = None if query is None else predict(model, xs, graphs, query, p).cpu().numpy()
    return model, values, trace, dict(epoch=selected_epoch, seconds=time.monotonic() - start,
        parameters=sum(v.numel() for v in model.parameters()),
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None, query_outcomes_used_for_offset_fitting=False)


def run(out, task, device, stage):
    out = Path(out)
    p = common.protocol(out)
    a = common.load(out / 'inputs' / task / 'arrays.npz')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ph = common.base.digest(out / 'protocol.json')
    selection = None if stage == 'screen' else json.loads((out / 'SELECTION.json').read_text())
    if selection is not None:
        assert selection['protocol_sha256'] == ph
    seeds = [p['screen_seed']] if stage == 'screen' else p['confirm_seeds']
    for kind in p['kinds']:
        for lr in p['lrs'] if stage == 'screen' else [selection['chosen'][kind]['lr']]:
            for seed in seeds:
                dest = out / 'runs' / task / kind / ('lr' + str(lr)) / ('s' + str(seed))
                if (dest / 'result.json').exists():
                    common.verify(dest, ph)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                model, pred, trace, detail = fit(a, p, kind, lr, seed, device)
                torch.save(model.state_dict(), dest / 'model.pt')
                common.base.write_json(dest / 'trace.json', trace)
                common.base.write_json(dest / 'training_detail.json', detail)
                loaded, xs, graphs, _ = setup(a, p, kind, device)
                loaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
                query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
                replay = predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
                chunk = np.concatenate([predict(loaded, xs, graphs, query[:, i:i + 800], p).cpu().numpy()
                                        for i in range(0, query.shape[1], 800)])
                for value in (replay, chunk):
                    np.testing.assert_allclose(pred, value, atol=3e-5, rtol=3e-5)
                scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=pred,
                              offset=a['offset_val'])
                for name in ('gnn', 'tabicl', 'anchor'):
                    weight = p['gnn_weight'] if name == 'gnn' else p['strong_weight']
                    scores[name] = a[name + '_' + str(seed)]
                    scores[name + '_composed'] = (1 - weight) * scores[name] + weight * pred
                scores['offset_composed'] = (1 - p['strong_weight']) * scores['anchor'] + p['strong_weight'] * scores['offset']
                np.savez_compressed(dest / 'scores.npz', **scores)
                result = dict(task=task, kind=kind, lr=lr, seed=seed, **detail, protocol_sha256=ph,
                    replay_max_abs=float(max(abs(pred - replay).max(), abs(pred - chunk).max())),
                    metrics={n: common.base.metrics(scores['y'], v, scores['pairs'])
                             for n, v in scores.items() if n not in ('y', 'pairs', 'offset', 'offset_composed')},
                    offset_metrics={n: common.base.metrics(scores['y'], scores[n], scores['pairs'])
                                    for n in ('offset', 'offset_composed')},
                    files={n: common.base.digest(dest / n) for n in ('model.pt', 'trace.json', 'training_detail.json', 'scores.npz')})
                common.base.write_json(dest / 'result.json', result)
                print('VERIFIED', p['family'], task, kind, lr, seed, detail['epoch'], result['metrics']['anchor_composed'], flush=True)
                del model, loaded, xs, graphs
                gc.collect()
                torch.cuda.empty_cache()


def report(out):
    out = Path(out)
    common.report(out)
    p = common.protocol(out)
    rows = []
    for path in (out / 'runs').glob('*/*/*/*/result.json'):
        r = json.loads(path.read_text())
        z = common.load(path.parent / 'scores.npz')
        for name, expected in r['offset_metrics'].items():
            actual = common.base.metrics(z['y'], z[name], z['pairs'])
            for m in ('auc', 'ap', 'macro_drug_ap'):
                assert abs(actual[m] - expected[m]) < 1e-12
        rows.append(dict(task=r['task'], kind=r['kind'], lr=r['lr'], seed=r['seed'],
                         **{m: r['metrics']['anchor_composed'][m] - r['offset_metrics']['offset_composed'][m]
                            for m in ('auc', 'ap')}))
    table = pd.DataFrame(rows)
    table.to_csv(out / 'offset_contrasts.csv', index=False)
    print('VERSUS_OFFSET_ONLY', flush=True)
    print(table.groupby(['task', 'kind', 'lr'])[['auc', 'ap']].mean().to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'screen', 'select', 'confirm', 'report'))
    parser.add_argument('--family', choices=tuple(LIBRARIES), required=True)
    parser.add_argument('--out')
    parser.add_argument('--task', choices=common.base.TASKS)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    out = Path(args.out) if args.out else ROOT / ('results/residual_' + args.family + '_20260915')
    if args.command == 'prepare':
        prepare(out, args.family)
    elif args.command in ('screen', 'confirm'):
        run(out, args.task, args.device, args.command)
    elif args.command == 'select':
        common.select(out)
    else:
        report(out)
