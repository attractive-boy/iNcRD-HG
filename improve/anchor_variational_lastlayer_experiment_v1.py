"""Nested-reference likelihood with a variational Gaussian classifier head."""
import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from improve import anchor_priority_graph_v1 as shared
from improve import graph_prediction_precision as precision

common, ROOT, nested = shared.common, shared.ROOT, shared.nested
TRAIN_KEYS = shared.TRAIN_KEYS
library, offset_arrays = shared.library, shared.offset_arrays
score_arrays, select = shared.score_arrays, shared.select
OWN = ('anchor_variational_lastlayer_experiment_v1.py', 'test_anchor_variational_lastlayer_experiment_v1.py',
       'variational_lastlayer_graph_v1.py', 'test_variational_lastlayer_graph_v1.py',
       'sharpness_graph_v1.py', 'overlap_block_models.py',
       'anchor_priority_graph_v1.py', 'test_anchor_priority_graph_v1.py',
       'assess_declared_graph.py', 'graph_prediction_precision.py',
       'residual_graph_experiment.py', 'nested_anchor_inputs_v2.py')


def prepare(out, family, cache, core):
    out, cache, parent_dir = Path(out).resolve(), Path(cache).resolve(), Path(core).resolve()
    if out.exists():
        raise ValueError('Use a new variational classifier study')
    parent = common.protocol(parent_dir)
    if parent['family'] != family or family != 'variational_lastlayer':
        raise ValueError('Unexpected classifier family')
    if nested.protocol(cache)['scope'] != 'historical':
        raise ValueError('Historical screening requires historical OOF offsets')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        method='anchor_priority_'+family+'_v1', family=family, smoke=False,
        implementation_module=parent['implementation_module'],
        tasks=parent['tasks'], kinds=[parent['primary_kind']]+parent['primary_controls'],
        primary_kind=parent['primary_kind'], primary_controls=parent['primary_controls'],
        core_parameters=parent['core_parameters'], screen_seed=666, confirm_seeds=[0, 1],
        epochs=400, check_interval=20, lrs=[.001, .003], weight_decay=.001,
        strong_weight=.1, gnn_weight=.1, lnc_absolute_tolerance=.001,
        parent=str(parent_dir), nested_reference=str(cache), lnc_policy=parent['lnc_policy'],
        identity_records_are_not_neural_fits=True, selection=parent['selection'],
        likelihood='Discriminative VBLL Eq.14: mean logsumexp(class mean + offset + half marginal class variance) minus target class mean, plus Gaussian KL / total training rows. MAP instead uses ordinary cross entropy and the same Gaussian mean prior. Noise variance is zero.',
        training_budget='400 AdamW updates; every update uses the complete cyclically omitted query fold and one backward pass. Encoder decay 0.001; head decay zero because its explicit prior is in the objective. Betas=(0.9,0.999), gradient clipping 5.',
        rng='CPU and cuda:1 seeds are set separately. The variational head consumes no RNG during initialization or training; graph dropout remains common across arms.',
        prediction='For each support graph, integrate sigmoid(evaluation_anchor + mean_class_difference + Gaussian noise of variance equal to the sum of independent class variances) with 64-point Gauss-Hermite. Subtract anchor from the resulting logit and average these residuals over five supports; deploy anchor + 0.1 residual. This is a numerical approximation and fixed support-graph aggregation, not exact Bayesian graph marginalization.',
        numerical_audit='After fitting, compare 64-point inference with a fixed 128-point reference on every evaluation query, without labels; maximum residual error must be <=3e-5. The quadrature order is not selected by metrics.',
        diagnostic='Report the same selected model using its posterior mean only; this is descriptive, has no checkpoint search and is not an additional fit or independent control.',
        local_adaptation=parent['local_adaptation'],
        null='Zero mean and zero variance recover the reference exactly. LncDrug uses an explicit parameter-free identity and never constructs the graph or posterior.',
        gate='Mean MiDrug global ROC AUC and sklearn AP improve with >=2 of 3 seed-wise wins; every LncDrug decline is <=0.001. Full covariance must beat diagonal covariance and MAP on both mean MiDrug metrics. Fixed partitions and frozen outer recheck remain mandatory.',
        previous_candidate_failures_seen=True, previous_outer_results_seen=True,
        primary_and_controls_fixed_before_screening=True, original_test_source_access=False,
        this_candidate_outer_outcomes_used=False,
        innovation_count_policy='At most one variational-weight-learning family after all gates. Covariance variants, quadrature, calibration, fixed composition and identity do not count separately.',
        confirmed_innovations=0, existing_default_replaced=False,
        source_sha256={**parent['source_sha256'], **{n:common.base.digest(ROOT/n) for n in OWN}},
        inputs={str(parent_dir/'protocol.json'):common.base.digest(parent_dir/'protocol.json'),
                str(cache/'protocol.json'):common.base.digest(cache/'protocol.json')})
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT/name, out/'source_snapshot'/name)
    for task in p['tasks']:
        source = shared.prior.INPUTS/task/'arrays.npz'
        a = common.load(source)
        a = {k:v for k,v in a.items() if not k.startswith('offset_')}
        a['anchor_oof'], manifest = offset_arrays(cache, task, 'historical', a)
        dest = out/'inputs'/task/'arrays.npz'
        dest.parent.mkdir(parents=True)
        np.savez_compressed(dest, **a)
        for path in (source, dest, manifest, manifest.parent/'arrays.npz'):
            p['inputs'][str(path)] = common.base.digest(path)
    common.base.write_json(out/'protocol.json', p)
    print('FROZEN_VARIATIONAL_CLASSIFIER', common.base.digest(out/'protocol.json'), flush=True)


def setup(a, p, kind, device):
    if set(a)-TRAIN_KEYS:
        raise ValueError('Variational setup accepts training arrays only')
    return library(p).setup(a, p['core_parameters'], kind, device)


@torch.no_grad()
def predict_components(model, xs, graphs, pairs, p, anchor, quadrature_order=None):
    if anchor.ndim != 1 or len(anchor) != pairs.shape[1] or not torch.isfinite(anchor).all():
        raise ValueError('Prediction requires aligned finite reference logits')
    if not xs:
        zero = anchor.new_zeros(anchor.shape, dtype=torch.float64)
        return dict(residual=zero, mean_residual=zero.clone(), variance=zero.clone())
    order = p['core_parameters']['quadrature_order'] if quadrature_order is None else quadrature_order
    with precision.deterministic_evaluation():
        model.eval()
        values = []
        for graph in graphs:
            mean, variance = model(xs, graph, pairs)
            values.append(library(p).residual_prediction(mean, variance, anchor, order))
        return {name:torch.stack([v[i] for v in values]).mean(0)
                for i,name in enumerate(('residual','mean_residual','variance'))}


@torch.no_grad()
def predict(model, xs, graphs, pairs, p, anchor, quadrature_order=None):
    return predict_components(model, xs, graphs, pairs, p, anchor, quadrature_order)['residual']


def fit(a, p, kind, lr, seed, device, fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke',False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a)-TRAIN_KEYS:
        raise ValueError('Fixed-step fitting accepts training arrays only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered posterior arm')
    if a['anchor_oof'].shape != a['y_train'].shape or not np.isfinite(a['anchor_oof']).all():
        raise ValueError('Invalid training OOF reference')
    if p['core_parameters']['kl_multiplier'] != 1.:
        raise ValueError('Undeclared KL scaling')
    if device != 'cpu':
        torch.cuda.set_device(device)
    torch.random.default_generator.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
    training = {k:v for k,v in a.items() if k in TRAIN_KEYS}
    model, xs, graphs, folds = setup(training, p, kind, device)
    if library(p).identity_task(training, p['core_parameters']):
        value = None if fixed_epochs is not None else np.zeros(len(a['y_val']), dtype=np.float64)
        return model, value, [], dict(epoch=0, seconds=0., parameters=0, trainable_parameters=0,
            fixed_step_training=fixed_epochs is not None, trained_neural_model=False,
            training_steps=0, gradient_evaluations=0, identity_preserves_anchor=True,
            query_outcomes_used_for_offset_fitting=False, nested_feature_crossfit=True, no_new_lnc_training=True)
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one training step is required')
    pairs = torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None], device=device)
    labels = torch.as_tensor(a['y_train'], dtype=torch.long, device=device)
    offset = torch.as_tensor(a['anchor_oof'], dtype=torch.float64, device=device)
    query = None if fixed_epochs is not None else torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None], device=device)
    eval_anchor = None if query is None else torch.as_tensor(a['anchor_'+str(seed)], dtype=torch.float64, device=device)
    groups = [dict(params=[v for name,v in model.named_parameters() if not name.startswith('last.') and v.requires_grad], weight_decay=p['weight_decay']),
              dict(params=[v for v in model.last.parameters() if v.requires_grad], weight_decay=0.)]
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(.9,.999), eps=1e-8)
    best, selected, state, trace = -np.inf, epochs, None, []
    start = time.monotonic()
    for step in range(1,epochs+1):
        fold = (step-1)%len(graphs)
        ids = folds[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mean, variance = model(xs, graphs[fold], pairs[:,ids])
        penalty = model.last.regularizer()
        loss, nll_bound, regularizer = library(p).objective(mean, variance, offset[ids], labels[ids], penalty, len(labels))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite variational objective')
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError('Nonfinite variational gradient')
        optimizer.step()
        row = dict(epoch=step, omitted_fold=fold, query_count=len(ids), loss=float(loss.detach()),
            nll_bound=float(nll_bound.detach()), regularizer_unscaled=float(penalty.detach()),
            regularizer_per_training_row=float(regularizer.detach()), total_training_rows=len(labels),
            regularizer_is_gaussian_kl=kind != 'map', gradient_norm=float(gradient_norm),
            mean_class_variance=float(variance.detach().mean()), max_class_variance=float(variance.detach().max()),
            covariance_offdiagonal_norm=float(model.last.lower.detach().norm()))
        if (step%p['check_interval'] == 0 or step == epochs) and query is not None:
            value = predict(model,xs,graphs,query,p,eval_anchor).cpu().numpy()
            metrics = common.base.metrics(a['y_val'], a['anchor_'+str(seed)]+p['strong_weight']*value, a['pairs_val'])
            row['anchor_composed'] = metrics
            criterion = (metrics['auc']+metrics['ap'])/2
            if criterion > best:
                best, selected = criterion, step
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('TRAIN_VARIATIONAL',kind,lr,seed,step,round(metrics['auc'],9),round(metrics['ap'],9),flush=True)
        trace.append(row)
    if state is not None:
        model.load_state_dict(state)
    value = None if query is None else predict(model,xs,graphs,query,p,eval_anchor).cpu().numpy()
    return model,value,trace,dict(epoch=selected,seconds=time.monotonic()-start,
        parameters=sum(v.numel() for v in model.parameters()),trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None,trained_neural_model=True,training_steps=epochs,
        gradient_evaluations=epochs,identity_preserves_anchor=False,query_outcomes_used_for_offset_fitting=False,
        nested_feature_crossfit=True,covariance_mode=kind,regularizer_is_gaussian_kl=kind != 'map',
        prior_variance=model.last.prior_variance,head_weight_decay=0.,encoder_weight_decay=p['weight_decay'],
        stochastic_last_layer_samples=0,all_device_seeding_used=False)


def run(out,task,device,stage):
    out = Path(out)
    p = common.protocol(out)
    a = common.load(out/'inputs'/task/'arrays.npz')
    training = {k:v for k,v in a.items() if k in TRAIN_KEYS}
    ph = common.base.digest(out/'protocol.json')
    selection = None if stage == 'screen' else nested.base.read_json(out/'SELECTION.json')
    if selection is not None and selection['protocol_sha256'] != ph:
        raise ValueError('Selection protocol mismatch')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for kind in p['kinds']:
        for lr in p['lrs'] if stage == 'screen' else [selection['chosen'][kind]['lr']]:
            for seed in [p['screen_seed']] if stage == 'screen' else p['confirm_seeds']:
                dest = out/'runs'/task/kind/('lr'+str(lr))/('s'+str(seed))
                if (dest/'result.json').exists():
                    common.verify(dest,ph)
                    continue
                if dest.exists() and any(dest.iterdir()):
                    raise ValueError('Preserve partial fit artifacts')
                dest.mkdir(parents=True,exist_ok=True)
                model,value,trace,detail = fit(a,p,kind,lr,seed,device)
                torch.save(model.state_dict(),dest/'model.pt')
                common.base.write_json(dest/'trace.json',trace)
                loaded,xs,graphs,_ = setup(training,p,kind,device)
                loaded.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True))
                pairs = torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=device)
                anchor = torch.as_tensor(a['anchor_'+str(seed)],device=device,dtype=torch.float64)
                components = predict_components(loaded,xs,graphs,pairs,p,anchor)
                replay = predict(loaded,xs,graphs,pairs.flip(1),p,anchor.flip(0)).cpu().numpy()[::-1]
                chunks = np.concatenate([predict(loaded,xs,graphs,pairs[:,i:i+800],p,anchor[i:i+800]).cpu().numpy()
                                         for i in range(0,pairs.shape[1],800)])
                for actual in (components['residual'].cpu().numpy(),replay,chunks):
                    np.testing.assert_allclose(actual,value,atol=3e-5,rtol=3e-5)
                reference = predict(loaded,xs,graphs,pairs,p,anchor,p['core_parameters']['quadrature_reference_order']).cpu().numpy()
                quadrature_error = float(abs(value-reference).max())
                if quadrature_error > p['core_parameters']['quadrature_tolerance']:
                    raise ValueError('Frozen predictive quadrature is insufficiently accurate')
                diagnostics = {k:v.cpu().numpy() for k,v in components.items()}
                np.savez_compressed(dest/'posterior_diagnostics.npz',**diagnostics,reference_residual=reference)
                scores = score_arrays(a,value,seed,p['strong_weight'])
                scores['mean_only_composed'] = a['anchor_'+str(seed)]+p['strong_weight']*diagnostics['mean_residual']
                np.savez_compressed(dest/'scores.npz',**scores)
                result = dict(task=task,kind=kind,lr=lr,seed=seed,**detail,protocol_sha256=ph,
                    standalone_is_residual_ranking_not_absolute_predictor=True,
                    replay_max_abs=float(max(abs(value-replay).max(),abs(value-chunks).max())),
                    quadrature_reference_max_abs=quadrature_error,
                    mean_only_is_same_selected_weights=True,
                    metrics={n:common.base.metrics(scores['y'],v,scores['pairs']) for n,v in scores.items()
                             if n not in ('y','pairs','offset','offset_composed')},
                    offset_metrics={n:common.base.metrics(scores['y'],scores[n],scores['pairs']) for n in ('offset','offset_composed')},
                    files={n:common.base.digest(dest/n) for n in ('model.pt','trace.json','scores.npz','posterior_diagnostics.npz')})
                common.base.write_json(dest/'result.json',result)
                print('VERIFIED_VARIATIONAL',task,kind,lr,seed,detail['epoch'],result['metrics']['anchor_composed'],flush=True)
                del model,loaded,xs,graphs
                gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','screen','select','confirm'))
    parser.add_argument('--family',default='variational_lastlayer')
    parser.add_argument('--core')
    parser.add_argument('--out',required=True)
    parser.add_argument('--cache',default=str(nested.DEFAULT))
    parser.add_argument('--task',choices=('MiDrug','LncDrug'))
    parser.add_argument('--device',default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.command == 'prepare':
        prepare(args.out,args.family,args.cache,args.core)
    elif args.command == 'select':
        select(args.out)
    else:
        run(args.out,args.task,args.device,args.command)
