"""SOAP follow-up with a declared separable-variance control and preserved parent fits."""
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
from improve import graph_prediction_precision as precision
from improve import eigenbasis_adam_v1 as optimization

common, ROOT, nested = shared.common, shared.ROOT, shared.nested
TRAIN_KEYS = shared.TRAIN_KEYS
library, offset_arrays = shared.library, shared.offset_arrays
score_arrays, select = shared.score_arrays, shared.select
OWN = ('anchor_eigenbasis_experiment_v1.py', 'test_anchor_eigenbasis_experiment_v1.py',
       'eigenbasis_adam_v1.py', 'test_eigenbasis_adam_v1.py',
       'eigenbasis_graph_v1.py', 'sharpness_graph_v1.py',
       'anchor_priority_graph_v1.py', 'test_anchor_priority_graph_v1.py',
       'assess_declared_graph.py', 'graph_prediction_precision.py',
       'residual_graph_experiment.py', 'nested_anchor_inputs_v2.py')


OWN += ('kronecker_preconditioning_v1.py', 'preconditioned_graph_v1.py',
        'anchor_preconditioning_experiment_v1.py')

def prepare(out, family, cache, core):
    out, cache, parent_dir = Path(out).resolve(), Path(cache).resolve(), Path(core).resolve()
    if out.exists():
        raise ValueError('Use a new preconditioning study directory')
    parent = common.protocol(parent_dir)
    if parent['family'] != family or family != 'eigenbasis_adam':
        raise ValueError('Unexpected optimizer family')
    if nested.protocol(cache)['scope'] != 'historical':
        raise ValueError('Historical screening requires historical OOF offsets')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        method='anchor_priority_' + family + '_v1', family=family, smoke=False,
        implementation_module=parent['implementation_module'],
        tasks=parent['tasks'], kinds=[parent['primary_kind']] + parent['primary_controls'],
        primary_kind=parent['primary_kind'], primary_controls=parent['primary_controls'],
        core_parameters=parent['core_parameters'], optimizer_parameters=parent['optimizer_parameters'],
        screen_seed=666, confirm_seeds=[0, 1], epochs=400, check_interval=20,
        lrs=[.001, .003], weight_decay=.001, strong_weight=.1, gnn_weight=.1,
        lnc_absolute_tolerance=.001, private_rng_offset=29015,
        parent=str(parent_dir), nested_reference=str(cache),
        lnc_policy=parent['lnc_policy'], identity_records_are_not_neural_fits=True,
        selection=parent['selection'],
        likelihood='BCEWithLogits(fixed nested OOF strong reference + learned raw graph residual). All optimizers share the same graph and objective; no extra representation loss.',
        training_budget='One initialization gradient on fold 0 changes no parameters, then 400 updates on cyclically omitted folds. All three arms consume 401 gradients and 400 updates, with clipping at 5. No warm-start performance checkpoint.',
        rng='CPU initialization and cuda:1 dropout are seeded separately, with no all-device seeding. Graphs and initialization are identical across arms.',
        trace='Every gradient call records loss, omitted fold, gradient norm, and optimizer-state diagnostics. Validation is used every 20 parameter updates only. The final optimizer state is saved independently of the selected inference model.',
        prediction='MiDrug: frozen evaluation anchor + 0.1 times mean raw residual over five support graphs. LncDrug: exact strong-reference identity, without graph construction or fitting.',
        local_adaptation=parent['local_adaptation'],
        null='Zero residual exactly preserves the strong reference. Identity scoring records are not neural fits.',
        gate='Mean MiDrug global ROC AUC and sklearn AP increase and each improves in >=2 of 3 initializations; no LncDrug metric drop exceeds 0.001. Full must beat separable_variance and AdamW on both mean MiDrug metrics. Fixed partitions and frozen outer recheck remain mandatory.',
        previous_candidate_failures_seen=True, previous_outer_results_seen=True,
        primary_and_controls_fixed_before_new_fits=True, parent_screening_results_seen=True,
        this_revision_has_independent_screen=False, baseline_adoption=parent['baseline_adoption'],
        original_test_source_access=False, this_candidate_outer_outcomes_used=False,
        diagnostic_budget='Three declared arms and two learning rates. No beta, QR frequency, initialization factor, damping or composition-weight sweep.',
        innovation_count_policy='At most one optimizer-family innovation after all gates. SAM/Muon/SOAP/KL variants, controls, identity and residual infrastructure do not count separately.',
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
    print('FROZEN_PRECONDITIONING', family, common.base.digest(out / 'protocol.json'), flush=True)


def setup(a, p, kind, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Preconditioning setup accepts training arrays only')
    return library(p).setup(a, p['core_parameters'], kind, device)


@torch.no_grad()
def predict(model, xs, graphs, pairs, p):
    with precision.deterministic_evaluation():
        model.eval()
        return torch.stack([model(xs, graph, pairs).double() for graph in graphs]).mean(0)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke', False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a) - TRAIN_KEYS:
        raise ValueError('Fixed-step fitting accepts training arrays only')
    if kind not in p['kinds'] or kind not in optimization.KINDS:
        raise ValueError('Unregistered optimizer arm')
    if a['anchor_oof'].shape != a['y_train'].shape or not np.isfinite(a['anchor_oof']).all():
        raise ValueError('Invalid nested reference offsets')
    if device != 'cpu':
        torch.cuda.set_device(device)
    torch.random.default_generator.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
    training = {k: v for k, v in a.items() if k in TRAIN_KEYS}
    model, xs, graphs, folds = setup(training, p, kind, device)
    if library(p).identity_task(training, p['core_parameters']):
        model.optimization_final_state = None
        value = None if fixed_epochs is not None else np.zeros(len(a['y_val']), dtype=np.float64)
        return model, value, [], dict(epoch=0, seconds=0., parameters=0, trainable_parameters=0,
            fixed_step_training=fixed_epochs is not None, mean_raw_residual=True,
            query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True,
            trained_neural_model=False, training_steps=0, gradient_evaluations=0,
            initialization_gradients=0, identity_preserves_anchor=True, no_new_lnc_training=True)
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one parameter update is required')
    pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(a['y_train'], dtype=torch.float64, device=device)
    offset = torch.as_tensor(a['anchor_oof'], dtype=torch.float64, device=device)
    query = None if fixed_epochs is not None else torch.as_tensor(
        a['pairs_val'] + a['pair_offset'][:, None], device=device)
    optimizer = optimization.optimizer(model, kind, lr, p['weight_decay'], p['optimizer_parameters'])
    best, selected, state, trace = -np.inf, epochs, None, []
    start = time.monotonic()
    for step in range(epochs + 1):
        fold = max(0, step-1) % len(graphs)
        ids = folds[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual = model(xs, graphs[fold], pairs[:, ids])
        loss = F.binary_cross_entropy_with_logits(offset[ids] + residual.double(), labels[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite residual training loss')
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        diagnostics = optimizer.diagnostics()
        if (not diagnostics['state_finite'] or diagnostics['parameter_updates_min'] != step
                or diagnostics['parameter_updates_max'] != step
                or diagnostics['parameter_gradient_calls_min'] != step+1
                or diagnostics['parameter_gradient_calls_max'] != step+1):
            raise RuntimeError('Invalid optimizer state or unequal per-parameter update budget')
        row = dict(epoch=step, initialization_only=step == 0, loss=float(loss.detach()),
                   omitted_fold=fold, query_count=len(ids), gradient_norm=float(gradient_norm),
                   **diagnostics)
        if step > 0 and (step % p['check_interval'] == 0 or step == epochs) and query is not None:
            value = predict(model, xs, graphs, query, p).cpu().numpy()
            composed = a['anchor_' + str(seed)] + p['strong_weight'] * value
            metrics = common.base.metrics(a['y_val'], composed, a['pairs_val'])
            row['anchor_composed'] = metrics
            criterion = (metrics['auc'] + metrics['ap']) / 2
            if criterion > best:
                best, selected = criterion, step
                state = cpu_copy(model.state_dict())
            print('TRAIN_PRECONDITIONING', kind, lr, seed, step,
                  round(metrics['auc'], 9), round(metrics['ap'], 9), flush=True)
        trace.append(row)
    model.optimization_final_state = cpu_copy(optimizer.state_dict())
    if state is not None:
        model.load_state_dict(state)
    value = None if query is None else predict(model, xs, graphs, query, p).cpu().numpy()
    return model, value, trace, dict(epoch=selected, seconds=time.monotonic()-start,
        parameters=sum(v.numel() for v in model.parameters()),
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None, mean_raw_residual=True,
        query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True,
        trained_neural_model=True, training_steps=epochs, gradient_evaluations=epochs+1,
        initialization_gradients=1, identity_preserves_anchor=False, optimizer_kind=kind,
        matrix_parameter_names=list(optimizer.matrix_names), final_optimizer_diagnostics=diagnostics,
        final_optimizer_state_epoch=epochs, all_device_seeding_used=False)


def run(out, task, device, stage):
    out = Path(out)
    p = common.protocol(out)
    a = common.load(out / 'inputs' / task / 'arrays.npz')
    training = {k: v for k, v in a.items() if k in TRAIN_KEYS}
    ph = common.base.digest(out / 'protocol.json')
    selection = None if stage == 'screen' else nested.base.read_json(out / 'SELECTION.json')
    if selection is not None and selection['protocol_sha256'] != ph:
        raise ValueError('Selection protocol mismatch')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for kind in p['kinds']:
        for lr in p['lrs'] if stage == 'screen' else [selection['chosen'][kind]['lr']]:
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
                torch.save(model.optimization_final_state, dest / 'optimizer_final.pt')
                common.base.write_json(dest / 'trace.json', trace)
                loaded, xs, graphs, _ = setup(training, p, kind, device)
                loaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
                query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
                replay = predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
                chunks = np.concatenate([predict(loaded, xs, graphs, query[:, i:i+800], p).cpu().numpy()
                                         for i in range(0, query.shape[1], 800)])
                for actual in (replay, chunks):
                    np.testing.assert_allclose(value, actual, atol=3e-5, rtol=3e-5)
                scores = score_arrays(a, value, seed, p['strong_weight'])
                np.savez_compressed(dest / 'scores.npz', **scores)
                result = dict(task=task, kind=kind, lr=lr, seed=seed, **detail,
                    protocol_sha256=ph, standalone_is_residual_ranking_not_absolute_predictor=True,
                    replay_max_abs=float(max(abs(value-replay).max(), abs(value-chunks).max())),
                    metrics={n: common.base.metrics(scores['y'], v, scores['pairs']) for n, v in scores.items()
                             if n not in ('y', 'pairs', 'offset', 'offset_composed')},
                    offset_metrics={n: common.base.metrics(scores['y'], scores[n], scores['pairs'])
                                    for n in ('offset', 'offset_composed')},
                    files={n: common.base.digest(dest / n) for n in
                           ('model.pt', 'optimizer_final.pt', 'trace.json', 'scores.npz')})
                common.base.write_json(dest / 'result.json', result)
                print('VERIFIED_PRECONDITIONING', task, kind, lr, seed, detail['epoch'],
                      result['metrics']['anchor_composed'], flush=True)
                del model, loaded, xs, graphs
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'screen', 'select', 'confirm'))
    parser.add_argument('--family', default='eigenbasis_adam')
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
