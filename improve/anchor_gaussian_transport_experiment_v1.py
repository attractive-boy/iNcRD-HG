"""Strong-reference residual fitting for neighbourhood Gaussian transport."""
import argparse
import gc
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from improve import anchor_priority_graph_v1 as shared
from improve import gaussian_transport_graph_v1 as core
from improve import tabular_foundation as base

common, ROOT, nested = shared.common, shared.ROOT, shared.nested
TRAIN_KEYS = shared.TRAIN_KEYS
prepare, setup, predict, select = shared.prepare, shared.setup, shared.predict, shared.select


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def training_snapshot(model, optimizer, steps, device):
    return cpu_copy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), steps=steps,
        optimizer_parameter_names=[name for name, value in model.named_parameters() if value.requires_grad],
        frozen_parameter_names=[name for name, value in model.named_parameters() if not value.requires_grad],
        cpu_rng=torch.random.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if device != 'cpu' else None))


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke', False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a) - TRAIN_KEYS:
        raise ValueError('Fixed Gaussian-transport fitting accepts training arrays only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered Gaussian-transport arm')
    if a['anchor_oof'].shape != a['y_train'].shape or not np.isfinite(a['anchor_oof']).all():
        raise ValueError('Invalid nested reference offset')
    torch.random.default_generator.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
    training = {k: v for k, v in a.items() if k in TRAIN_KEYS}
    model, xs, graphs, folds = setup(training, p, kind, device)
    if core.identity_task(training, p['core_parameters']):
        model.training_final_state = None
        value = None if fixed_epochs is not None else np.zeros(a['pairs_val'].shape[1], dtype=np.float64)
        return model, value, [], dict(epoch=0, seconds=0., parameters=0, trainable_parameters=0,
            fixed_parameters=0, fixed_step_training=fixed_epochs is not None,
            trained_neural_model=False, training_steps=0, gradient_evaluations=0, optimizer_steps=0,
            identity_preserves_anchor=True, no_new_lnc_training=True,
            geometry_diagnostics={},
            query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True)
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one fitting step is required')
    pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(a['y_train'], device=device, dtype=torch.float64)
    offset = torch.as_tensor(a['anchor_oof'], device=device, dtype=torch.float64)
    query = None if fixed_epochs is not None else torch.as_tensor(
        a['pairs_val'] + a['pair_offset'][:, None], device=device)
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad],
                                  lr=lr, weight_decay=p['weight_decay'])
    best, selected, selected_state, trace = -np.inf, epochs, None, []
    start, completed_steps, current_step, operation = time.monotonic(), 0, 0, 'setup'
    try:
        for step in range(1, epochs + 1):
            current_step, operation = step, 'forward'
            fold = (step - 1) % len(graphs)
            ids = folds[fold]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            residual = model(xs, graphs[fold], pairs[:, ids])
            loss = F.binary_cross_entropy_with_logits(offset[ids] + residual.double(), labels[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite Gaussian-transport residual loss')
            operation = 'backward'
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            operation = 'optimizer_step'
            optimizer.step()
            completed_steps = step
            if step % p['check_interval'] == 0 or step == epochs:
                row = dict(epoch=step, loss=float(loss.detach()), gradient_norm=float(gradient))
                if query is not None:
                    operation = 'checkpoint_evaluation'
                    value = predict(model, xs, graphs, query, p).cpu().numpy()
                    metrics = common.base.metrics(a['y_val'],
                        a['anchor_' + str(seed)] + p['strong_weight'] * value, a['pairs_val'])
                    row['anchor_composed'] = metrics
                    criterion = (metrics['auc'] + metrics['ap']) / 2
                    if criterion > best:
                        best, selected = criterion, step
                        selected_state = cpu_copy(model.state_dict())
                    if step % 100 == 0:
                        print('TRAIN_GAUSSIAN_TRANSPORT', kind, lr, seed, step,
                              round(metrics['auc'], 9), round(metrics['ap'], 9), flush=True)
                row['geometry_diagnostics'] = dict(model.geometry_diagnostics)
                trace.append(row)
    except Exception as error:
        # Record an exact partial state if a registered numerical gate stops a run.
        # These fields report only fully returned optimizer steps, never estimates.
        error.training_partial_state = training_snapshot(model, optimizer, completed_steps, device)
        error.training_failure_context = dict(current_step=current_step, completed_steps=completed_steps,
            operation=operation, trace=trace, diagnostics=dict(model.geometry_diagnostics))
        raise
    model.training_final_state = training_snapshot(model, optimizer, epochs, device)
    if selected_state is not None:
        model.load_state_dict(selected_state, strict=True)
    value = None if query is None else predict(model, xs, graphs, query, p).cpu().numpy()
    return model, value, trace, dict(epoch=selected, seconds=time.monotonic() - start,
        parameters=sum(v.numel() for v in model.parameters()),
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_parameters=sum(v.numel() for v in model.parameters() if not v.requires_grad),
        fixed_step_training=fixed_epochs is not None, trained_neural_model=True, training_steps=epochs,
        gradient_evaluations=epochs, optimizer_steps=epochs, final_optimizer_state_epoch=epochs,
        identity_preserves_anchor=False, mean_raw_residual=True,
        query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True,
        all_device_seeding_used=False, geometry_diagnostics=dict(model.geometry_diagnostics))


def run(out, task, device, stage):
    if device != 'cuda:1':
        raise ValueError('Formal execution is restricted to cuda:1')
    out = Path(out)
    p = common.protocol(out)
    if p.get('required_executor') != 'improve.anchor_gaussian_transport_experiment_v1':
        raise ValueError('Dedicated Gaussian-transport executor was not declared')
    if stage not in ('screen', 'confirm') or task not in p['tasks']:
        raise ValueError('Unregistered stage or task')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a = common.load(out / 'inputs' / task / 'arrays.npz')
    ph = common.base.digest(out / 'protocol.json')
    selection = None if stage == 'screen' else base.read_json(out / 'SELECTION.json')
    if selection is not None and selection['protocol_sha256'] != ph:
        raise ValueError('Selection protocol changed')
    for kind in p['kinds']:
        lrs = p['lrs'] if stage == 'screen' else [selection['chosen'][kind]['lr']]
        seeds = [p['screen_seed']] if stage == 'screen' else p['confirm_seeds']
        for lr in lrs:
            for seed in seeds:
                dest = out / 'runs' / task / kind / ('lr' + str(lr)) / ('s' + str(seed))
                if (dest / 'result.json').exists():
                    common.verify(dest, ph)
                    continue
                if dest.exists():
                    raise ValueError('Incomplete run must be reviewed before resuming: ' + str(dest))
                dest.mkdir(parents=True)
                try:
                    model, value, trace, detail = fit(a, p, kind, lr, seed, device)
                except Exception as error:
                    context = getattr(error, 'training_failure_context', {})
                    files = {}
                    if hasattr(error, 'training_partial_state'):
                        torch.save(error.training_partial_state, dest / 'training_partial.pt')
                        files['training_partial.pt'] = base.digest(dest / 'training_partial.pt')
                    base.write_json(dest / 'FAILED.json', base.seal(dict(task=task, kind=kind, lr=lr,
                        seed=seed, protocol_sha256=ph, error=repr(error), traceback=traceback.format_exc(),
                        training_context=context, files=files, completed_formal_fit=False)))
                    raise
                torch.save(model.state_dict(), dest / 'model.pt')
                names = ['model.pt', 'trace.json', 'scores.npz']
                if detail['trained_neural_model']:
                    torch.save(model.training_final_state, dest / 'training_final.pt')
                    names.append('training_final.pt')
                common.base.write_json(dest / 'trace.json', trace)
                training = {k: v for k, v in a.items() if k in TRAIN_KEYS}
                loaded, xs, graphs, _ = setup(training, p, kind, device)
                loaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True), strict=True)
                query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
                replay = predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
                chunks = np.concatenate([predict(loaded, xs, graphs, query[:, i:i + 800], p).cpu().numpy()
                    for i in range(0, query.shape[1], 800)])
                for actual in (replay, chunks):
                    np.testing.assert_allclose(value, actual, atol=3e-5, rtol=3e-5)
                scores = shared.score_arrays(a, value, seed, p['strong_weight'])
                np.savez_compressed(dest / 'scores.npz', **scores)
                result = dict(task=task, kind=kind, lr=lr, seed=seed, **detail,
                    protocol_sha256=ph, standalone_is_residual_ranking_not_absolute_predictor=True,
                    replay_max_abs=float(max(abs(value - replay).max(), abs(value - chunks).max())),
                    metrics={n: common.base.metrics(scores['y'], v, scores['pairs']) for n, v in scores.items()
                             if n not in ('y', 'pairs', 'offset', 'offset_composed')},
                    offset_metrics={n: common.base.metrics(scores['y'], scores[n], scores['pairs'])
                                    for n in ('offset', 'offset_composed')},
                    files={name: common.base.digest(dest / name) for name in names})
                common.base.write_json(dest / 'result.json', result)
                print('VERIFIED_GAUSSIAN_TRANSPORT', task, kind, lr, seed, detail['epoch'],
                      result['metrics']['anchor_composed'], flush=True)
                del model, loaded, xs, graphs
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('screen', 'confirm'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--task', required=True, choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    run(Path(args.out), args.task, args.device, args.stage)
