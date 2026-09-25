"""Train a residual predictor with PULL-derived auxiliary graph self-training."""
import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import anchor_priority_graph_v1 as shared

common, ROOT, nested = shared.common, shared.ROOT, shared.nested
TRAIN_KEYS = shared.TRAIN_KEYS
library, offset_arrays = shared.library, shared.offset_arrays
predict, score_arrays, select = shared.predict, shared.score_arrays, shared.select
OWN = ('anchor_expected_graph_experiment_v1.py', 'test_anchor_expected_graph_experiment_v1.py',
       'anchor_priority_graph_v1.py', 'test_anchor_priority_graph_v1.py',
       'assess_declared_graph.py', 'graph_prediction_precision.py',
       'residual_graph_experiment.py', 'nested_anchor_inputs_v2.py')


def prepare(out, family, cache, core):
    out, cache, parent_dir = Path(out).resolve(), Path(cache).resolve(), Path(core).resolve()
    if out.exists():
        raise ValueError('Use a new expected-graph study directory')
    parent = common.protocol(parent_dir)
    if parent['family'] != family or family != 'expected_graph_selftraining':
        raise ValueError('Unexpected expected-graph family')
    if nested.protocol(cache)['scope'] != 'historical':
        raise ValueError('Historical screening requires historical OOF offsets')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        method='anchor_expected_graph_' + family + '_v1', family=family, smoke=False,
        implementation_module=parent['implementation_module'], tasks=parent['tasks'],
        kinds=[parent['primary_kind']] + parent['primary_controls'],
        primary_kind=parent['primary_kind'], primary_controls=parent['primary_controls'],
        core_parameters=parent['core_parameters'], screen_seed=666, confirm_seeds=[0, 1],
        epochs=400, check_interval=20, selection_start_epoch=60,
        lrs=[.001, .003], weight_decay=.001,
        strong_weight=.1, gnn_weight=.1, lnc_absolute_tolerance=.001,
        parent=str(parent_dir), nested_reference=str(cache),
        lnc_policy=parent['lnc_policy'], identity_records_are_not_neural_fits=True,
        selection=parent['selection'],
        likelihood='BCE(anchor_oof + original-graph residual, fitting label) + 0.1 times auxiliary loss. Full auxiliary loss is the mean of expected-graph soft-target BCE and original-graph observed-label BCE. no_correction replaces that mean by expected-graph BCE at the same total weight. observed_only uses original graphs and observed targets in both auxiliary views.',
        training_budget='400 AdamW steps, one cyclic omitted fold per update, two graph forwards and one backward. Every 50 steps, freeze a teacher snapshot and rebuild each support-only expected graph. Teacher refreshes are inference, not extra optimization steps.',
        private_rng='CPU and cuda:1 seeded separately without all-device seeding; dropout is the only random training draw. Graph proposals and cutoff ties are deterministic.',
        trace='Every step stores main BCE, expected and correction BCE, pseudo-target counts and mass, virtual-edge count, fold and gradient norm. Eight stage-start teacher snapshots and all proposal/query probability arrays are saved.',
        prediction='Unchanged anchor + 0.1 times mean raw residual over the five original support graphs; expected graphs and the absolute-probability head are training auxiliaries only. LncDrug is exact zero-residual identity.',
        local_adaptation=parent['local_adaptation'],
        null='Zero residual exactly preserves the strong reference. Identity records are not neural fits.',
        gate='Mean MiDrug AUC and sklearn AP increase and each improves in >=2 of 3 paired initializations; no LncDrug metric loss exceeds 0.001. Full must beat both registered controls on both mean MiDrug metrics. Fixed partitions and frozen outer recheck remain mandatory.',
        previous_candidate_failures_seen=True, previous_outer_results_seen=True,
        primary_and_controls_fixed_before_screening=True,
        original_test_source_access=False, this_candidate_outer_outcomes_used=False,
        diagnostic_budget='Only three declared arms and two learning rates; no pseudo-label, degree, auxiliary-weight, stage-length or composition search.',
        innovation_count_policy='One auxiliary expected-graph self-training family only after all gates; no counts for its probability head, correction control, snapshots, offset or identity.',
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
    print('FROZEN_EXPECTED_GRAPH', family, common.base.digest(out / 'protocol.json'), flush=True)


def setup(a, p, kind, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Expected-graph setup accepts training inputs only')
    # The graph constructor and proposal model never receive reference offsets.
    graph_training = {k: v for k, v in a.items() if k != 'anchor_oof'}
    return library(p).setup(graph_training, p['core_parameters'], kind, device)


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke', False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a) - TRAIN_KEYS:
        raise ValueError('Fixed expected-graph fitting accepts training inputs only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered expected-graph arm')
    if a['anchor_oof'].shape != a['y_train'].shape or not np.isfinite(a['anchor_oof']).all():
        raise ValueError('Invalid nested reference offset')
    training = {k: v for k, v in a.items() if k in TRAIN_KEYS}
    torch.random.default_generator.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.set_device('cuda:1')
        torch.cuda.manual_seed(seed)
    model, xs, graphs, folds = setup(training, p, kind, device)
    model.augmentation_evidence = []
    if library(p).identity_task(training, p['core_parameters']):
        value = None if fixed_epochs is not None else np.zeros(a['pairs_val'].shape[1], dtype=np.float64)
        return model, value, [], dict(epoch=0, seconds=0., parameters=0, trainable_parameters=0,
            fixed_step_training=fixed_epochs is not None, mean_raw_residual=True,
            query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True,
            trained_neural_model=False, training_steps=0, gradient_evaluations=0,
            identity_preserves_anchor=True, no_new_lnc_training=True,
            graph_constructed=False, teacher_refreshes=0, graph_forward_evaluations=0)
    pairs = torch.as_tensor(training['pairs_train'] + training['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(training['y_train'], device=device)
    offset = torch.as_tensor(training['anchor_oof'], device=device, dtype=torch.float64)
    query = None if fixed_epochs is not None else torch.as_tensor(
        a['pairs_val'] + a['pair_offset'][:, None], device=device)
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1 or any(len(ids) == 0 for ids in folds):
        raise ValueError('Expected positive steps and nonempty support folds')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
    best, selected, state, trace = -np.inf, epochs, None, []
    core = library(p)
    start = time.monotonic()
    for step in range(1, epochs + 1):
        if (step - 1) % p['core_parameters']['stage_steps'] == 0:
            stage = (step - 1) // p['core_parameters']['stage_steps']
            snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            refresh = [graph.refresh(model, xs, stage, p['core_parameters'], kind) for graph in graphs]
            model.augmentation_evidence.append(dict(stage=stage, before_step=step,
                teacher_state=snapshot, refresh=refresh, contexts=[graph.evidence() for graph in graphs]))
        fold = (step - 1) % len(graphs)
        ids, graph = folds[fold], graphs[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual, original_logits = model.forward_both(xs, graph.original, pairs[:, ids])
        _, expected_logits = model.forward_both(xs, graph.expected, pairs[:, ids])
        soft_targets = graph.soft_targets(labels[ids])
        auxiliary, expected_loss, correction = core.auxiliary_objective(
            expected_logits, original_logits, soft_targets, labels[ids], kind)
        main = F.binary_cross_entropy_with_logits(offset[ids] + residual.double(), labels[ids].double())
        loss = main + p['core_parameters']['auxiliary_weight'] * auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite expected-graph objective')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        row = dict(epoch=step, omitted_fold=fold, stage=graph.stage, query_count=len(ids),
            loss=float(loss.detach()), bce=float(main.detach()), auxiliary=float(auxiliary.detach()),
            expected_bce=float(expected_loss.detach()), correction_bce=float(correction.detach()),
            soft_unknown_count=int(((labels[ids] == 0) & (soft_targets > 0)).sum()),
            soft_unknown_mass=float(soft_targets[labels[ids] == 0].double().sum()),
            virtual_edges=graph.added.shape[1], cutoff=graph.cutoff,
            gradient_norm=float(grad_norm.detach()))
        if step % p['check_interval'] == 0 or step == epochs:
            if query is not None:
                value = predict(model, xs, graphs, query, p).cpu().numpy()
                composed = a['anchor_' + str(seed)] + p['strong_weight'] * value
                metrics = common.base.metrics(a['y_val'], composed, a['pairs_val'])
                row['anchor_composed'] = metrics
                criterion = (metrics['auc'] + metrics['ap']) / 2
                # A pre-augmentation checkpoint cannot demonstrate this mechanism.
                if step > p['core_parameters']['stage_steps'] and criterion > best:
                    best, selected = criterion, step
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if step % 100 == 0:
                    print('TRAIN_EXPECTED_GRAPH', kind, lr, seed, step, round(metrics['auc'], 9),
                          round(metrics['ap'], 9), 'virtual', row['virtual_edges'], flush=True)
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
        teacher_refreshes=len(model.augmentation_evidence), graph_forward_evaluations=2 * epochs,
        teacher_snapshots_are_not_separate_neural_fits=True,
        inference_uses_expected_graph=False, proposals_use_residual_as_probability=False,
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
                torch.save(model.augmentation_evidence, dest / 'augmentation.pt')
                common.base.write_json(dest / 'trace.json', trace)
                loaded, xs, graphs, _ = setup(training, p, kind, device)
                loaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
                query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
                replay = predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
                chunks = np.concatenate([predict(loaded, xs, graphs, query[:, i:i + 317], p).cpu().numpy()
                    for i in range(0, query.shape[1], 317)])
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
                    files={n: common.base.digest(dest / n) for n in
                           ('model.pt', 'augmentation.pt', 'trace.json', 'scores.npz')})
                common.base.write_json(dest / 'result.json', result)
                print('VERIFIED_EXPECTED_GRAPH', task, kind, lr, seed, detail['epoch'],
                      result['metrics']['anchor_composed'], flush=True)
                del model, loaded, xs, graphs
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'screen', 'select', 'confirm'))
    parser.add_argument('--family', default='expected_graph_selftraining')
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
