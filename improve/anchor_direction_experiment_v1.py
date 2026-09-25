"""Strong-reference residual fitting with projected independent edge directions."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import anchor_priority_graph_v1 as shared
from improve import continuous_direction_graph_v1 as core

common, ROOT, nested = shared.common, shared.ROOT, shared.nested
TRAIN_KEYS = shared.TRAIN_KEYS
prepare, setup, predict, select = shared.prepare, shared.setup, shared.predict, shared.select


def optimizer_for(model, p, lr):
    weights = [v for name, v in model.named_parameters() if v.requires_grad and name != 'theta']
    optimizer = torch.optim.AdamW([
        dict(params=weights, lr=lr, weight_decay=p['weight_decay']),
        dict(params=[model.theta], lr=p['core_parameters']['direction_learning_rate'],
             weight_decay=p['core_parameters']['direction_weight_decay'])])
    return optimizer


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke', False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a) - TRAIN_KEYS:
        raise ValueError('Fixed fitting accepts training inputs only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered direction arm')
    if a['anchor_oof'].shape != a['y_train'].shape or not np.isfinite(a['anchor_oof']).all():
        raise ValueError('Invalid nested reference offset')
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
    model, xs, graphs, folds = setup(a, p, kind, device)
    if core.identity_task(a, p['core_parameters']):
        value = None if fixed_epochs is not None else np.zeros(len(a['pairs_val'][0]), dtype=np.float64)
        return model, value, [], dict(epoch=0, seconds=0., parameters=0, trainable_parameters=0,
            fixed_step_training=fixed_epochs is not None, mean_raw_residual=True,
            query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True,
            trained_neural_model=False, training_steps=0, gradient_evaluations=0,
            identity_preserves_anchor=True, no_new_lnc_training=True,
            direction_projection_steps=0, direction_summary=core.direction_summary(model))
    pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(a['y_train'], device=device)
    offset = torch.as_tensor(a['anchor_oof'], device=device, dtype=torch.float64)
    query = (None if fixed_epochs is not None else
             torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device))
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one fitting step is required')
    optimizer = optimizer_for(model, p, lr)
    best, selected, state, trace = -np.inf, epochs, None, []
    start = time.monotonic()
    theta_gradient_steps = 0
    for step in range(1, epochs + 1):
        fold = (step - 1) % len(graphs)
        ids = folds[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual = model(xs, graphs[fold], pairs[:, ids])
        loss = F.binary_cross_entropy_with_logits(offset[ids] + residual.double(), labels[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite direction residual loss')
        loss.backward()
        theta_gradient_steps += int(model.theta.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        model.project_directions()
        if step % p['check_interval'] == 0 or step == epochs:
            row = dict(epoch=step, loss=float(loss.detach()), direction_summary=core.direction_summary(model))
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
    model._training_final_state = dict(optimizer=optimizer.state_dict(),
        model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, steps=epochs)
    if state is not None:
        model.load_state_dict(state)
    value = None if query is None else predict(model, xs, graphs, query, p).cpu().numpy()
    expected = 0 if kind == 'fixed_direction' else epochs
    if theta_gradient_steps != expected:
        raise ValueError('Independent angle gradients were unexpectedly missing or active')
    summary = core.direction_summary(model)
    if summary['min_angle'] < -1e-6 or summary['max_angle'] > np.pi / 2 + 1e-6:
        raise ValueError('Projected direction bound failed')
    if kind == 'fixed_direction' and summary['mean_abs_angle_change'] != 0.:
        raise ValueError('Frozen direction control changed')
    return model, value, trace, dict(epoch=selected, seconds=time.monotonic() - start,
        parameters=sum(v.numel() for v in model.parameters()),
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None, mean_raw_residual=True,
        query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True,
        trained_neural_model=True, training_steps=epochs, gradient_evaluations=epochs,
        identity_preserves_anchor=False, direction_projection_steps=epochs,
        direction_gradient_steps=theta_gradient_steps, direction_summary=summary,
        direction_learning_rate=p['core_parameters']['direction_learning_rate'],
        direction_weight_decay=p['core_parameters']['direction_weight_decay'])


def run(out, task, device, stage):
    previous = shared.fit
    shared.fit = fit
    try:
        return shared.run(out, task, device, stage)
    finally:
        shared.fit = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('screen', 'confirm'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--task', required=True, choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    run(Path(args.out), args.task, args.device, args.stage)
