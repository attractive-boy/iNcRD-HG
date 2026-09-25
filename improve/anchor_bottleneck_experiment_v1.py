"""Nested-reference residual training with a query-specific graph KL objective.

Only the new bottleneck core is trained here. Fixed-step fitting accepts
training arrays only; the historical selector retains the common budget.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import anchor_priority_graph_v1 as shared
from improve import graph_prediction_precision as precision

common, ROOT, nested = shared.common, shared.ROOT, shared.nested
TRAIN_KEYS = shared.TRAIN_KEYS
library, offset_arrays = shared.library, shared.offset_arrays
score_arrays, select = shared.score_arrays, shared.select
OWN = ('anchor_bottleneck_experiment_v1.py', 'test_anchor_bottleneck_experiment_v1.py',
       'anchor_priority_graph_v1.py', 'test_anchor_priority_graph_v1.py',
       'assess_declared_graph.py', 'graph_prediction_precision.py',
       'residual_graph_experiment.py', 'nested_anchor_inputs_v2.py')


def prepare(out, family, cache, core):
    out, cache, parent_dir = Path(out).resolve(), Path(cache).resolve(), Path(core).resolve()
    if out.exists():
        raise ValueError('Use a new bottleneck study directory')
    parent = common.protocol(parent_dir)
    if parent['family'] != family or family != 'information_bottleneck':
        raise ValueError('Unexpected bottleneck family')
    if nested.protocol(cache)['scope'] != 'historical':
        raise ValueError('Historical screening requires historical OOF offsets')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        method='anchor_bottleneck_' + family + '_v1', family=family, smoke=False,
        implementation_module=parent['implementation_module'],
        tasks=parent['tasks'], kinds=[parent['primary_kind']] + parent['primary_controls'],
        primary_kind=parent['primary_kind'], primary_controls=parent['primary_controls'],
        core_parameters=parent['core_parameters'],
        screen_seed=666, confirm_seeds=[0, 1], epochs=400, check_interval=20,
        lrs=[.001, .003], weight_decay=.001, strong_weight=.1, gnn_weight=.1,
        lnc_absolute_tolerance=.001, private_rng_offset=29015,
        parent=str(parent_dir), nested_reference=str(cache),
        lnc_policy=parent['lnc_policy'], identity_records_are_not_neural_fits=True,
        selection=parent['selection'],
        likelihood='BCEWithLogits(fixed nested OOF strong reference + learned residual) plus 0.001 times mean query-graph KL. Each query graph sums the analytic Bernoulli KL over its undirected non-self edges. no_kl removes the entire KL gradient; no_completion removes only virtual edges.',
        training_budget='400 AdamW steps. Each step uses every query in one cyclically omitted support fold, one relaxed graph sample, one backward pass, and gradient clipping at 5. No gradient accumulation or extra optimizer update.',
        private_rng='CPU initialization and cuda:1 dropout are seeded separately without all-device seeding. Edge sampling uses a private cuda:1 Generator seeded by seed + 29015.',
        trace='Every update records BCE, graph-mean edge-sum KL, weighted KL, edge retention, edge counts, fold and gradient norm; validation is consulted only every 20 updates.',
        prediction='MiDrug: unchanged evaluation anchor + 0.1 times mean raw graph residual over five supports. Inference uses sigmoid edge probabilities, with deterministic evaluation and chunks of at most 128 queries. This is a plug-in approximation, not exact graph marginalization. LncDrug is an exact zero-residual identity with no graph or training.',
        local_adaptation=parent['local_adaptation'],
        null='Zero residual exactly preserves the strong reference. Identity records are not neural fits.',
        gate='Mean MiDrug AUC and sklearn AP increase and each improves in >=2 of 3 paired initializations; no LncDrug metric loss exceeds 0.001. Full must beat both declared controls on both mean MiDrug metrics. Fixed partitions and frozen outer recheck remain mandatory.',
        previous_candidate_failures_seen=True, previous_outer_results_seen=True,
        primary_and_controls_fixed_before_screening=True,
        original_test_source_access=False, this_candidate_outer_outcomes_used=False,
        diagnostic_budget='Only three declared arms and two learning rates; no completion, prior, KL-weight, temperature or coefficient sweep.',
        innovation_count_policy='At most one per mechanism family after all gates. No separate counts for controls, infrastructure, revisions or identity.',
        confirmed_innovations=0, existing_default_replaced=False,
        source_sha256={**parent['source_sha256'], **{n: common.base.digest(ROOT / n) for n in OWN}},
        inputs={str(parent_dir / 'protocol.json'): common.base.digest(parent_dir / 'protocol.json'),
                str(cache / 'protocol.json'): common.base.digest(cache / 'protocol.json')})
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, out / 'source_snapshot' / name)
    for task in p['tasks']:
        source = shared.prior.INPUTS / task / 'arrays.npz'
        a = common.load(source)
        a = {k: v for k, v in a.items() if not k.startswith('offset_')}
        a['anchor_oof'], manifest = offset_arrays(cache, task, 'historical', a)
        dest = out / 'inputs' / task / 'arrays.npz'
        dest.parent.mkdir(parents=True)
        np.savez_compressed(dest, **a)
        for path in (source, dest, manifest, manifest.parent / 'arrays.npz'):
            p['inputs'][str(path)] = common.base.digest(path)
    common.base.write_json(out / 'protocol.json', p)
    print('FROZEN_BOTTLENECK', family, common.base.digest(out / 'protocol.json'), flush=True)


def setup(a, p, kind, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Bottleneck setup accepts training inputs only')
    return library(p).setup(a, p['core_parameters'], kind, device)


@torch.no_grad()
def predict(model, xs, graphs, pairs, p):
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError('Expected a pair matrix')
    if pairs.shape[1] == 0:
        return torch.empty(0, dtype=torch.float64, device=pairs.device)
    chunk = int(p['core_parameters']['query_chunk'])
    if chunk < 1:
        raise ValueError('Invalid inference chunk size')
    with precision.deterministic_evaluation():
        model.eval()
        values = [torch.cat([model(xs, graph, pairs[:, start:start + chunk]).double()
                            for start in range(0, pairs.shape[1], chunk)]) for graph in graphs]
        return torch.stack(values).mean(0)


def objective(offset, residual, labels, kl_per_query, kind, p):
    if kind not in p['kinds'] or kl_per_query.shape != residual.shape:
        raise ValueError('Invalid bottleneck objective inputs')
    bce = F.binary_cross_entropy_with_logits(offset.double() + residual.double(), labels.double())
    mean_kl = kl_per_query.double().mean()
    coefficient = 0. if kind == 'no_kl' else p['core_parameters']['information_weight']
    weighted = coefficient * mean_kl
    loss = bce if kind == 'no_kl' else bce + weighted
    return loss, bce, mean_kl, weighted


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke', False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a) - TRAIN_KEYS:
        raise ValueError('Fixed bottleneck fitting accepts training inputs only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered bottleneck arm')
    if a['anchor_oof'].shape != a['y_train'].shape or not np.isfinite(a['anchor_oof']).all():
        raise ValueError('Invalid nested reference offset')
    training = {k: v for k, v in a.items() if k in TRAIN_KEYS}
    torch.random.default_generator.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.set_device('cuda:1')
        torch.cuda.manual_seed(seed)
    model, xs, graphs, folds = setup(training, p, kind, device)
    if library(p).identity_task(training, p['core_parameters']):
        value = None if fixed_epochs is not None else np.zeros(a['pairs_val'].shape[1], dtype=np.float64)
        return model, value, [], dict(epoch=0, seconds=0., parameters=0, trainable_parameters=0,
            fixed_step_training=fixed_epochs is not None, mean_raw_residual=True,
            query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True,
            trained_neural_model=False, training_steps=0, gradient_evaluations=0,
            identity_preserves_anchor=True, no_new_lnc_training=True,
            graph_constructed=False, kl_contributes_to_gradient=False)
    pairs = torch.as_tensor(training['pairs_train'] + training['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(training['y_train'], device=device)
    offset = torch.as_tensor(training['anchor_oof'], device=device, dtype=torch.float64)
    query = None if fixed_epochs is not None else torch.as_tensor(
        a['pairs_val'] + a['pair_offset'][:, None], device=device)
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1 or any(len(ids) == 0 for ids in folds):
        raise ValueError('Expected positive steps and nonempty support folds')
    generator = torch.Generator(device=device).manual_seed(seed + p['private_rng_offset'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
    best, selected, state, trace = -np.inf, epochs, None, []
    start = time.monotonic()
    for step in range(1, epochs + 1):
        fold = (step - 1) % len(graphs)
        ids = folds[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual, kl, edge_detail = model.forward_with_kl(xs, graphs[fold], pairs[:, ids], generator)
        loss, bce, mean_kl, weighted = objective(offset[ids], residual, labels[ids], kl, kind, p)
        if not torch.isfinite(loss) or not torch.isfinite(mean_kl):
            raise FloatingPointError('Nonfinite bottleneck training objective')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = dict(epoch=step, omitted_fold=fold, query_count=len(ids), loss=float(loss.detach()),
            bce=float(bce.detach()), mean_query_edge_sum_kl=float(mean_kl.detach()),
            weighted_kl=float(weighted.detach()), gradient_norm=float(grad_norm.detach()), **edge_detail)
        if step % p['check_interval'] == 0 or step == epochs:
            if query is not None:
                value = predict(model, xs, graphs, query, p).cpu().numpy()
                composed = a['anchor_' + str(seed)] + p['strong_weight'] * value
                metrics = common.base.metrics(a['y_val'], composed, a['pairs_val'])
                row['anchor_composed'] = metrics
                criterion = (metrics['auc'] + metrics['ap']) / 2
                if criterion > best:
                    best, selected = criterion, step
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print('TRAIN_BOTTLENECK', kind, lr, seed, step, round(metrics['auc'], 9),
                    round(metrics['ap'], 9), 'KL', round(row['mean_query_edge_sum_kl'], 5),
                    'keep', round(row['mean_retention'], 5),
                    'seconds', round(time.monotonic() - start, 1), flush=True)
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
        identity_preserves_anchor=False, graph_constructed=True,
        kl_contributes_to_gradient=kind != 'no_kl',
        kl_reduction='sum unique non-self edges per query, then mean queries',
        private_generator_seed=seed + p['private_rng_offset'],
        private_generator_final_state_sha256=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest(),
        total_query_presentations=sum(row['query_count'] for row in trace),
        graph_audits=[graph.audit for graph in graphs])


def run(out, task, device, stage):
    out = Path(out)
    p = common.protocol(out)
    a = common.load(out / 'inputs' / task / 'arrays.npz')
    training = {k: v for k, v in a.items() if k in TRAIN_KEYS}
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
                if dest.exists() and any(dest.iterdir()):
                    raise ValueError('Preserve partial fit artifacts: ' + str(dest))
                dest.mkdir(parents=True, exist_ok=True)
                model, value, trace, detail = fit(a, p, kind, lr, seed, device)
                torch.save(model.state_dict(), dest / 'model.pt')
                common.base.write_json(dest / 'trace.json', trace)
                loaded, xs, graphs, _ = setup(training, p, kind, device)
                loaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
                query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
                replay = predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
                chunks = np.concatenate([predict(loaded, xs, graphs, query[:, i:i + 73], p).cpu().numpy()
                    for i in range(0, query.shape[1], 73)])
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
                print('VERIFIED_BOTTLENECK', task, kind, lr, seed, detail['epoch'],
                    result['metrics']['anchor_composed'], flush=True)
                del model, loaded, xs, graphs
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'screen', 'select', 'confirm'))
    parser.add_argument('--family', default='information_bottleneck')
    parser.add_argument('--core')
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache', default=str(nested.DEFAULT))
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.command == 'prepare':
        prepare(args.out, args.family, args.cache, args.core)
    elif args.command == 'select':
        select(args.out)
    else:
        run(args.out, args.task, args.device, args.command)
