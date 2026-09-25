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
OWN = ('anchor_stochastic_augmentation_v1.py', 'stochastic_augmentation_graph_v1.py', 'test_stochastic_augmentation_v1.py', 'sharpness_graph_v1.py', 'overlap_block_models.py', 'overlap_block_experiment.py', 'non_tabular_experiment.py', 'anchor_priority_graph_v1.py', 'assess_declared_graph.py', 'graph_prediction_precision.py', 'residual_graph_experiment.py', 'nested_anchor_inputs_v2.py')


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
        likelihood='BCE on OOF-reference-plus-residual, plus 0.1 BPR on same-RNA actual training pairs with latent synthetic negative embeddings and 0.05 symmetric InfoNCE on noisy/reversed node views. The synthetic negative retains its original training-pair reference logit; it is a regularizer, not an observed new association.',
        training_budget='400 encoder AdamW updates and 400 detached-feature denoising score-network updates; two backwards per step. Generator parameters are frozen during differentiable reverse-sampling gradients to encoder features. All arms train the same score network and draw the same private random tensors.',
        local_adaptation=parent['local_adaptation'],
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
    if set(a)-TRAIN_KEYS:
        raise ValueError('Setup accepts training arrays only')
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


def cpu_copy(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def fit(a,p,kind,lr,seed,device,fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke',False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a)-TRAIN_KEYS:
        raise ValueError('Fixed-step fitting accepts training arrays only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered augmentation arm')
    if a['anchor_oof'].shape != a['y_train'].shape or not np.isfinite(a['anchor_oof']).all():
        raise ValueError('Invalid nested training reference')
    if device != 'cpu':
        torch.cuda.set_device(device)
    torch.random.default_generator.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
    training = {k:v for k,v in a.items() if k in TRAIN_KEYS}
    model,xs,graphs,folds = setup(training,p,kind,device)
    if library(p).identity_task(training,p['core_parameters']):
        model.training_final_state = None
        value = None if fixed_epochs is not None else np.zeros(len(a['y_val']),dtype=np.float64)
        return model,value,[],dict(epoch=0,seconds=0.,parameters=0,trainable_parameters=0,
            fixed_step_training=fixed_epochs is not None,trained_neural_model=False,training_steps=0,
            gradient_evaluations=0,encoder_optimizer_steps=0,score_optimizer_steps=0,
            identity_preserves_anchor=True,no_new_lnc_training=True,query_outcomes_used_for_offset_fitting=False)
    c = p['core_parameters']
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one training update is required')
    pairs = torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None],device=device)
    labels = torch.as_tensor(a['y_train'],device=device,dtype=torch.float64)
    offset = torch.as_tensor(a['anchor_oof'],device=device,dtype=torch.float64)
    query = None if fixed_epochs is not None else torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=device)
    pools = library(p).training_triplets(training,device,c['folds'])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed+p['private_rng_offset'])
    encoder_parameters = [v for name,v in model.named_parameters() if not name.startswith('score_network.')]
    optimizer = torch.optim.AdamW(encoder_parameters,lr=lr,weight_decay=p['weight_decay'])
    score_optimizer = torch.optim.AdamW(model.score_network.parameters(),lr=c['score_lr'],weight_decay=c['score_weight_decay'])
    best,selected,state,trace = -np.inf,epochs,None,[]
    start = time.monotonic()
    for step in range(1,epochs+1):
        fold = (step-1)%len(graphs)
        ids = folds[fold]
        model.train()
        h = model.encode(xs,graphs[fold])
        score_optimizer.zero_grad(set_to_none=True)
        time_sample = .00001+.99999*torch.rand(len(h),device=device,generator=generator)
        noise = torch.randn(h.shape,device=device,dtype=h.dtype,generator=generator)
        score_loss = library(p).denoising_loss(model.score_network,h,c,time_sample,noise)
        if not torch.isfinite(score_loss):
            raise FloatingPointError('Nonfinite score-matching loss')
        score_loss.backward()
        score_gradient = torch.nn.utils.clip_grad_norm_(model.score_network.parameters(),5.)
        if not torch.isfinite(score_gradient):
            raise FloatingPointError('Nonfinite score-matching gradient')
        score_optimizer.step()
        score_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        positive,negative = library(p).sample_triplets(pools[fold],c['triplet_batch'],generator)
        rnas = torch.unique(pairs[0,ids])
        drugs = torch.unique(pairs[1,ids])
        nodes = torch.cat((rnas,drugs))
        view_noise = torch.randn((c['reverse_steps']+1,len(nodes),c['width']),device=device,generator=generator)
        hard_noise = torch.randn((c['reverse_steps']+1,2*c['triplet_batch'],c['width']),device=device,generator=generator)
        for parameter in model.score_network.parameters():
            parameter.requires_grad_(False)
        try:
            view1,view2 = library(p).reverse_sample(model.score_network.score,h[nodes],c,view_noise,kind)
            latent = torch.cat((h[pairs[1,positive]],h[pairs[1,negative]]),0)
            _,synthetic = library(p).reverse_sample(model.score_network.score,latent,c,hard_noise,kind,inject=True)
            hard = synthetic[c['triplet_batch']:]
            cl = .5*(library(p).contrastive(view1[:len(rnas)],view2[:len(rnas)],c['contrastive_temperature'])+
                       library(p).contrastive(view1[len(rnas):],view2[len(rnas):],c['contrastive_temperature']))
            residual = model.decode(h[pairs[0,ids]],h[pairs[1,ids]])
            bce = F.binary_cross_entropy_with_logits(offset[ids]+residual.double(),labels[ids])
            pos_score = offset[positive]+model.decode(h[pairs[0,positive]],h[pairs[1,positive]]).double()
            neg_score = offset[negative]+model.decode(h[pairs[0,negative]],hard).double()
            ranking = F.softplus(neg_score-pos_score).mean()
            loss = bce+c['ranking_weight']*ranking+c['contrastive_weight']*cl
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite augmented training objective')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(encoder_parameters,5.)
            if not torch.isfinite(gradient):
                raise FloatingPointError('Nonfinite augmented training gradient')
            optimizer.step()
        finally:
            for parameter in model.score_network.parameters():
                parameter.requires_grad_(True)
        row = dict(epoch=step,omitted_fold=fold,query_count=len(ids),triplet_count=len(positive),
            loss=float(loss.detach()),bce=float(bce.detach()),ranking=float(ranking.detach()),
            contrastive=float(cl.detach()),score_loss=float(score_loss.detach()),
            encoder_gradient_norm=float(gradient),score_gradient_norm=float(score_gradient),
            encoder_optimizer_steps=step,score_optimizer_steps=step,gradient_evaluations=2*step,
            synthetic_displacement=float((hard.detach()-h[pairs[1,negative]].detach()).square().mean().sqrt()),
            same_rna_triplets=bool(torch.equal(pairs[0,positive],pairs[0,negative])))
        if not row['same_rna_triplets']:
            raise ValueError('Triplets mix RNA identities')
        if (step%p['check_interval']==0 or step==epochs) and query is not None:
            value = predict(model,xs,graphs,query,p).cpu().numpy()
            metrics = common.base.metrics(a['y_val'],a['anchor_'+str(seed)]+p['strong_weight']*value,a['pairs_val'])
            row['anchor_composed'] = metrics
            criterion = (metrics['auc']+metrics['ap'])/2
            if criterion > best:
                best,selected = criterion,step
                state = cpu_copy(model.state_dict())
            print('TRAIN_STOCHASTIC_AUGMENTATION',kind,lr,seed,step,round(metrics['auc'],9),round(metrics['ap'],9),flush=True)
        trace.append(row)
    model.training_final_state = cpu_copy(dict(encoder_optimizer=optimizer.state_dict(),score_optimizer=score_optimizer.state_dict(),private_rng=generator.get_state(),epoch=epochs))
    if state is not None:
        model.load_state_dict(state)
    value = None if query is None else predict(model,xs,graphs,query,p).cpu().numpy()
    return model,value,trace,dict(epoch=selected,seconds=time.monotonic()-start,
        parameters=sum(v.numel() for v in model.parameters()),trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None,trained_neural_model=True,training_steps=epochs,
        gradient_evaluations=2*epochs,encoder_optimizer_steps=epochs,score_optimizer_steps=epochs,
        final_optimizer_state_epoch=epochs,identity_preserves_anchor=False,all_device_seeding_used=False,
        mean_raw_residual=True,query_outcomes_used_for_offset_fitting=False,nested_feature_crossfit=True)


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
                if dest.exists() and any(dest.iterdir()):
                    raise ValueError('Preserve partial fitting artifacts')
                dest.mkdir(parents=True, exist_ok=True)
                model, value, trace, detail = fit(a, p, kind, lr, seed, device)
                torch.save(model.state_dict(), dest / 'model.pt')
                torch.save(model.training_final_state, dest / 'training_final.pt')
                common.base.write_json(dest / 'trace.json', trace)
                loaded, xs, graphs, _ = setup({k:v for k,v in a.items() if k in TRAIN_KEYS}, p, kind, device)
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
                    files={n: common.base.digest(dest / n) for n in ('model.pt', 'trace.json', 'scores.npz', 'training_final.pt')})
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
