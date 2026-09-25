"""Strong-reference residual fitting with a two-path self-taught meta loss."""
import argparse
from collections import OrderedDict
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from improve import anchor_priority_graph_v1 as shared
from improve import meta_loss_graph_v1 as core

common, ROOT, nested = shared.common, shared.ROOT, shared.nested
TRAIN_KEYS = shared.TRAIN_KEYS
prepare, setup, predict, select = shared.prepare, shared.setup, shared.predict, shared.select


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke', False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a) - TRAIN_KEYS:
        raise ValueError('Fixed meta-loss fitting accepts training arrays only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered meta-loss arm')
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
            fixed_step_training=fixed_epochs is not None, trained_neural_model=False,
            training_steps=0, gradient_evaluations=0, task_gradient_evaluations=0,
            rescaler_gradient_evaluations=0, task_optimizer_steps=0, rescaler_optimizer_steps=0,
            hypergradient_steps=0, direct_gradient_steps=0, dropout_forward_passes=0,
            identity_preserves_anchor=True, no_new_lnc_training=True,
            query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True)
    c = p['core_parameters']
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one fitting step is required')
    pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(a['y_train'], device=device, dtype=torch.float64)
    offset = torch.as_tensor(a['anchor_oof'], device=device, dtype=torch.float64)
    query = None if fixed_epochs is not None else torch.as_tensor(
        a['pairs_val'] + a['pair_offset'][:, None], device=device)
    fold_ids = [ids.cpu().numpy() for ids in folds]
    memories = {(f, role): core.FeatureMemory((len(ids) + 1) // 2, c['feature_epsilon'])
                for f, ids in enumerate(fold_ids) for role in ('inner', 'meta')}
    rng = np.random.default_rng(seed + p['private_rng_offset'])
    optimizer = torch.optim.Adam(model.rescaler.parameters(), lr=c['rescaler_learning_rate'],
                                 weight_decay=c['rescaler_weight_decay'])
    task_state = {}
    omega = tuple(model.rescaler.parameters())
    best, selected, selected_state, trace = -np.inf, epochs, None, []
    start = time.monotonic()
    for step in range(1, epochs + 1):
        fold = (step - 1) % len(graphs)
        inner_np, meta_np = core.episode_indices(fold_ids[fold], rng)
        inner = torch.as_tensor(inner_np, device=device)
        meta = torch.as_tensor(meta_np, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        theta = OrderedDict(model.backbone.named_parameters())
        risks, logits, losses = core.observed_batch(model.backbone, theta, xs, graphs[fold],
            pairs[:, inner], offset[inner], labels[inner], c['dropout_passes'])
        features = memories[fold, 'inner'].features(logits, losses,
            a['y_train'][inner_np], inner_np)
        wi = core.scaled_weights(model.rescaler, torch.as_tensor(features, device=device),
                                  labels[inner], kind, 'inner', c['weight_multiplier'])
        inner_loss = (wi.double() * risks).mean()
        if not torch.isfinite(inner_loss):
            raise FloatingPointError('Nonfinite inner loss before the task gradient')
        gradients = torch.autograd.grad(inner_loss, tuple(theta.values()), create_graph=True)
        updated, next_state, task_norm = core.differentiable_adamw(theta,
            dict(zip(theta, gradients)), task_state, lr=lr, weight_decay=p['weight_decay'],
            betas=c['adam_betas'], eps=c['adam_epsilon'], clip=c['gradient_clip'])
        evaluation_theta = (OrderedDict((k, v.detach()) for k, v in updated.items())
                            if kind == 'no_meta_gradient' else updated)
        meta_risks, meta_logits, meta_losses = core.observed_batch(model.backbone,
            evaluation_theta, xs, graphs[fold], pairs[:, meta], offset[meta],
            labels[meta], c['dropout_passes'])
        meta_features = memories[fold, 'meta'].features(meta_logits, meta_losses,
            a['y_train'][meta_np], meta_np)
        wm = core.scaled_weights(model.rescaler, torch.as_tensor(meta_features, device=device),
                                  labels[meta], kind, 'meta', c['weight_multiplier'])
        meta_loss = (wm.double() * meta_risks).mean()
        direct_loss = (wm.double() * meta_risks.detach()).mean()
        if not all(torch.isfinite(v) for v in (inner_loss, meta_loss, task_norm)):
            raise FloatingPointError('Nonfinite meta-loss episode')
        # The separate direct derivative is a diagnostic. The actual update is
        # the derivative of the complete meta objective, including its unroll.
        direct = torch.autograd.grad(direct_loss, omega, retain_graph=True)
        total = torch.autograd.grad(meta_loss, omega)
        if any(not torch.isfinite(g).all() for g in total):
            raise FloatingPointError('Nonfinite rescaler gradient')
        hyper = [g - d for g, d in zip(total, direct)]
        for parameter, gradient in zip(omega, total):
            parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_(omega, c['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for name, parameter in model.backbone.named_parameters():
                parameter.copy_(updated[name])
        task_state = core.detached_optimizer_state(next_state)
        row = dict(epoch=step, omitted_fold=fold, inner_count=len(inner_np), meta_count=len(meta_np),
            disjoint_query_batches=True,
            episode_sha256=hashlib.sha256(np.r_[inner_np, -1, meta_np].tobytes()).hexdigest(),
            inner_loss=float(inner_loss.detach()), meta_loss=float(meta_loss.detach()),
            inner_weight_mean=float(wi.detach().mean()), inner_weight_min=float(wi.detach().min()),
            inner_weight_max=float(wi.detach().max()), meta_weight_mean=float(wm.detach().mean()),
            meta_weight_min=float(wm.detach().min()), meta_weight_max=float(wm.detach().max()),
            task_gradient_norm=float(task_norm.detach()), direct_gradient_norm=core.gradient_norm(direct),
            hypergradient_norm=core.gradient_norm(hyper), total_rescaler_gradient_norm=core.gradient_norm(total),
            task_moment_step=task_state['step'], dropout_forwards=2 * c['dropout_passes'])
        if step % p['check_interval'] == 0 or step == epochs:
            if query is not None:
                value = predict(model, xs, graphs, query, p).cpu().numpy()
                metrics = common.base.metrics(a['y_val'],
                    a['anchor_' + str(seed)] + p['strong_weight'] * value, a['pairs_val'])
                row['anchor_composed'] = metrics
                criterion = (metrics['auc'] + metrics['ap']) / 2
                if criterion > best:
                    best, selected = criterion, step
                    selected_state = cpu_copy(model.state_dict())
                if step % 100 == 0:
                    print('TRAIN_META_LOSS', kind, lr, seed, step,
                          round(metrics['auc'], 9), round(metrics['ap'], 9), flush=True)
        trace.append(row)
        del risks, logits, losses, features, gradients, updated, next_state, evaluation_theta
        del meta_risks, meta_logits, meta_losses, meta_features, wi, wm, direct, total, hyper
    model.training_final_state = cpu_copy(dict(task_optimizer=task_state,
        rescaler_optimizer=optimizer.state_dict(), model=model.state_dict(), steps=epochs,
        memories={str(f) + '_' + role: memory.state_dict() for (f, role), memory in memories.items()},
        episode_rng=rng.bit_generator.state, cpu_rng=torch.random.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if device != 'cpu' else None))
    if selected_state is not None:
        model.load_state_dict(selected_state, strict=True)
    value = None if query is None else predict(model, xs, graphs, query, p).cpu().numpy()
    if kind in ('no_meta_gradient', 'uniform') and any(r['hypergradient_norm'] != 0. for r in trace):
        raise ValueError('Disabled meta-gradient path is nonzero')
    if kind in ('no_outer_rescaling', 'uniform') and any(r['direct_gradient_norm'] != 0. for r in trace):
        raise ValueError('Disabled direct rescaling path is nonzero')
    return model, value, trace, dict(epoch=selected, seconds=time.monotonic() - start,
        parameters=sum(v.numel() for v in model.parameters()),
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None, trained_neural_model=True, training_steps=epochs,
        gradient_evaluations=3 * epochs, task_gradient_evaluations=epochs,
        rescaler_gradient_evaluations=2 * epochs, task_optimizer_steps=epochs,
        rescaler_optimizer_steps=epochs, hypergradient_steps=epochs if kind in ('full', 'no_outer_rescaling') else 0,
        direct_gradient_steps=epochs if kind in ('full', 'no_meta_gradient') else 0,
        dropout_forward_passes=2 * c['dropout_passes'] * epochs,
        final_optimizer_state_epoch=epochs, identity_preserves_anchor=False,
        mean_raw_residual=True, query_outcomes_used_for_offset_fitting=False,
        nested_feature_crossfit=True, meta_batch_from_training_only=True,
        inner_and_meta_share_omitted_fold=True, rescaler_used_at_prediction=False,
        all_device_seeding_used=False)


def run(out, task, device, stage):
    out = Path(out)
    p = common.protocol(out)
    if p.get('required_executor') != 'improve.anchor_meta_loss_experiment_v1':
        raise ValueError('Dedicated meta-loss executor was not declared')
    if stage not in ('screen', 'confirm') or task not in p['tasks']:
        raise ValueError('Unregistered stage or task')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a = common.load(out / 'inputs' / task / 'arrays.npz')
    ph = common.base.digest(out / 'protocol.json')
    selection = None if stage == 'screen' else common.base.read_json(out / 'SELECTION.json')
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
                model, value, trace, detail = fit(a, p, kind, lr, seed, device)
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
                print('VERIFIED_META_LOSS', task, kind, lr, seed, detail['epoch'],
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
