"""Validation-selected hyperparameters and initialization, with honest exports.

Eight recipes per original/additive arm; then five declared seeds of the
selected recipe. Freeze every decision before fixed-epoch outer refits. Best
single runs and all-seed means are exported separately. Test data is unused.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from improve import rectangle_pilot as previous
from improve import pair_exchange as parent
from improve.dual_pair_tuning import choose_epoch
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import json_digest, sha256, write_json
from improve.network_feature_ablation import score
from improve.pair_exchange_modules import remove_target_pairs, sample_queries
from improve.support_operator_pilot import macro_ap


ARMS = ('pca_query', 'additive')
OWN_FILES = ('additive_search.py', 'test_additive_search.py')


def sources():
    return dict(previous.sources(), **{name: sha256(Path(__file__).with_name(name)) for name in OWN_FILES})


def recipes(base_lr, branch_lr, arm, smoke=False):
    if arm not in ARMS:
        raise ValueError(arm)
    rows = []
    for scale in (.5, 1., 1.5):
        for option in (0, 1):
            rows.append(dict(base_lr=base_lr * scale, branch_lr=branch_lr * scale,
                max_correction=1. + option if arm == 'additive' else 1.,
                weight_decay=option * 1e-4 if arm == 'pca_query' else 0., schedule='constant'))
    for option in (0, 1):
        rows.append(dict(base_lr=base_lr, branch_lr=branch_lr,
            max_correction=1. + option if arm == 'additive' else 1.,
            weight_decay=option * 1e-4 if arm == 'pca_query' else 0., schedule='cosine'))
    if smoke:
        rows = [rows[2], rows[7]]
    return [dict(row, id='c' + str(i)) for i, row in enumerate(rows)]


def learning_factor(schedule, epoch, horizon, minimum=.05):
    if horizon < 1 or not 1 <= epoch <= horizon:
        raise ValueError('Invalid training time')
    if schedule == 'constant':
        return 1.
    if schedule != 'cosine' or not 0 < minimum <= 1:
        raise ValueError('Invalid learning-rate schedule')
    fraction = (epoch - 1) / max(1, horizon - 1)
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * fraction))


def pick_inner(records):
    if not records or any(r['stage'] != 'inner' for r in records):
        raise ValueError('Only inner-validation records may select a model')
    if any(not np.isfinite(r['selection_score']) for r in records):
        raise ValueError('Non-finite selection score')
    return sorted(records, key=lambda r: (-r['selection_score'], r['config_id'], r['seed']))[0]


def prepare(args):
    out, ref_dir = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if (out / 'protocol.json').exists() or (out / 'runs').exists():
        raise ValueError('Preserve the existing experiment')
    ref = previous.load_protocol(ref_dir)
    ref_selection = previous.selection(ref_dir, ref)
    verification = json.loads((ref_dir / 'VALIDATION.json').read_text())
    if not verification['passed'] or verification['smoke']:
        raise ValueError('A verified full reference is required')
    support = json.loads((Path(ref['reference_dir']) / 'protocol.json').read_text())
    grids = {}
    for task in ref['tasks']:
        grids[task] = {}
        for arm in ARMS:
            cfg = ref_selection['chosen'][task][arm]
            base_lr = ref['training'][task]['lr'] * cfg['lr_multiplier']
            branch_lr = base_lr * cfg['branch_lr_multiplier']
            grids[task][arm] = recipes(base_lr, branch_lr, arm, args.smoke)
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=bool(args.smoke),
        reference_dir=str(ref_dir), reference_protocol_sha256=ref['sha256'],
        reference_selection_sha256=ref_selection['sha256'], tasks=ref['tasks'], arms=list(ARMS),
        configurations=grids, tuning_seed=666, seeds=[666, 0] if args.smoke else [666, 0, 1, 2, 3],
        training={t: dict(c, train_epochs=3 if args.smoke else c['train_epochs']) for t, c in ref['training'].items()},
        width=16, depth=2, max_correction=1., query_fraction=.5, ramp_fraction=.2,
        smoothing_window=2 if args.smoke else 5, minimum_lr_fraction=.05,
        selection='Screen eight recipes per arm/task with seed 666. Freeze one recipe per arm by peak trailing-five inner mean(AUC,AP). Run all five declared seeds for that recipe. Select the best seed by the same score, ties by config id then numeric seed. Never select separate AUC/AP checkpoints.',
        reporting='Best validation-selected single run, five-seed mean/std, every attempted recipe, and fixed outer refits are separate outputs. The maximum is not an unbiased mean or evidence of structural novelty.',
        outer_policy='After both parameter and seed decisions are frozen, refit both arms at the union of their selected seeds; fixed epoch counts come from the corresponding inner run. Outer labels are evaluated only after each refit and never select parameters, seeds or epochs.',
        schedule_policy='Cosine time and correction ramp use the original full inner epoch cap even during shorter fixed-epoch outer refits; do not compress the schedule to the selected epoch.',
        graph_policy=ref['graph_policy'], evidence_policy=ref['evidence_policy'],
        optimizer='RMSprop alpha=.99, unweighted query BCE. Original receives LR/weight-decay/schedule recipes; additive receives LR/correction-limit/schedule recipes. Both receive eight distinct recipes and the same seed pool.',
        no_test_evaluation=True, source_sha256=sources(), files={}, outer_source_files={},
        limitations=['Adaptive exploration on an already reused internal split.',
                     'Outer validation was also used by earlier experiments; this is not a new independent test set.',
                     'The initial screening seed participates in the reported all-seed mean.',
                     'A validation-selected best seed is deliberately optimistic on that validation set.',
                     'Task, labels, PCA features, graph schema and existing circRNA extensions are inherited.',
                     'A sampled zero is not experimentally verified drug sensitivity.'])
    out.mkdir(parents=True, exist_ok=True)
    for task in p['tasks']:
        dest = out / 'inputs' / task
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('inner.npz', 'indices.npz', 'train_graph.pth'):
            shutil.copy2(ref_dir / 'inputs' / task / name, dest / name)
        outer = Path(support['reference_dir']) / 'inputs' / task / 'outer.npz'
        digest = sha256(outer)
        if digest != support['audit'][task]['source_archive_sha256']:
            raise ValueError('Original fitting/outer archive changed')
        # Byte-copy only: no outer arrays are parsed during tuning preparation.
        shutil.copy2(outer, dest / 'outer.npz')
        p['outer_source_files'][task] = dict(path=str(outer), sha256=digest)
        for file in dest.iterdir():
            if file.is_file():
                p['files'][str(file.relative_to(out))] = sha256(file)
    snap = out / 'source_snapshot'
    snap.mkdir(exist_ok=True)
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snap / name)
    p['sha256'] = json_digest(p)
    write_json(out / 'protocol.json', p)
    print('FROZEN', out, 'smoke' if args.smoke else '32 recipe runs + 16 seed runs + up to 8 fixed outer refits', flush=True)


def load_protocol(out):
    out = Path(out)
    p = json.loads((out / 'protocol.json').read_text())
    if p['sha256'] != json_digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('Protocol changed')
    if sources() != p['source_sha256']:
        raise ValueError('Frozen source changed')
    for name, digest in p['files'].items():
        if sha256(out / name) != digest:
            raise ValueError('Input changed: ' + name)
    return p


def read_decision(out, filename, p):
    value = json.loads((Path(out) / filename).read_text())
    if value['sha256'] != json_digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('Decision changed')
    if value['protocol_sha256'] != p['sha256']:
        raise ValueError('Decision belongs to another protocol')
    for name, digest in value['result_files_sha256'].items():
        if sha256(Path(out) / name) != digest:
            raise ValueError('A result changed after selection')
    return value


def freeze_decision(out, filename, p, **fields):
    dest = Path(out) / filename
    if dest.exists():
        raise ValueError('Preserve existing decision: ' + filename)
    value = dict(protocol_sha256=p['sha256'], **fields)
    value['sha256'] = json_digest(value)
    write_json(dest, value)
    return value


def directory(out, task, arm, config, seed, stage):
    return Path(out) / 'runs' / stage / task / arm / config / ('s' + str(seed))


def jobs(out, p, phase):
    params = None if phase == 'screen' else read_decision(out, 'PARAMETERS.json', p)
    selected = read_decision(out, 'SELECTION.json', p) if phase == 'outer' else None
    for task in p['tasks']:
        for arm in ARMS:
            if phase == 'screen':
                for cfg in p['configurations'][task][arm]:
                    yield task, arm, cfg, p['tuning_seed'], 'inner', p['training'][task]['train_epochs']
            elif phase == 'seeds':
                for seed in p['seeds']:
                    if seed != p['tuning_seed']:
                        yield task, arm, params['chosen'][task][arm], seed, 'inner', p['training'][task]['train_epochs']
            elif phase == 'outer':
                seeds = sorted({selected['best'][task][a]['seed'] for a in ARMS})
                for seed in seeds:
                    epoch = selected['by_seed'][task][arm][str(seed)]['selected_epoch']
                    yield task, arm, params['chosen'][task][arm], seed, 'outer', epoch
            else:
                raise ValueError(phase)


def build(arrays, graph, task, arm, seed, cfg, p, device):
    return previous.build(arrays, graph, task, arm, seed,
                          dict(p, max_correction=cfg['max_correction']), device)


def train(arrays, graph, task, arm, seed, cfg, p, device, stage, epochs):
    started = time.monotonic()
    model, data, pairs, weights = build(arrays, graph, task, arm, seed, cfg, p, device)
    initial, base_initial = state_digest(model), state_digest(model.base)
    groups = [dict(params=list(model.base.parameters()), lr=cfg['base_lr'])]
    rates = [cfg['base_lr']]
    if model.branch is not None:
        groups.append(dict(params=list(model.branch.parameters()), lr=cfg['branch_lr']))
        rates.append(cfg['branch_lr'])
    optimizer = torch.optim.RMSprop(groups, lr=cfg['base_lr'], alpha=.99, weight_decay=cfg['weight_decay'])
    labels = torch.from_numpy(arrays['y_train']).to(device)
    horizon = p['training'][task]['train_epochs']
    best, state, saved, selected = -np.inf, None, None, None
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        factor = learning_factor(cfg['schedule'], epoch, horizon, p['minimum_lr_fraction'])
        for group, initial_lr in zip(optimizer.param_groups, rates):
            group['lr'] = initial_lr * factor
        optimizer.zero_grad(set_to_none=True)
        query = sample_queries(arrays['y_train'], p['query_fraction'], (seed + 1) * 1000003 + epoch)
        held = arrays['pairs_train'][:, query[arrays['y_train'][query] == 1]]
        edges, step_weights = remove_target_pairs(data.edge_index_dict, weights, task,
            p['training'][task]['rna'], data['drug'].num_nodes, held)
        ramp = min(1., (epoch - 1) / max(1., p['ramp_fraction'] * horizon))
        logits = model(data.x_dict, edges, pairs['train'][:, query], step_weights,
                       pairs['train'], labels, ramp=ramp)
        loss = F.binary_cross_entropy_with_logits(logits, labels[query])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite loss')
        loss.backward()
        optimizer.step()
        row = dict(epoch=epoch, loss=float(loss.detach()), queries=len(query), ramp=ramp,
                   base_lr=optimizer.param_groups[0]['lr'], learning_factor=factor)
        if stage == 'inner':
            outputs = previous.predict(model, data, pairs, weights, labels)
            auc, ap = score(arrays['y_val'], outputs['val'])
            row.update(inner_auc=auc, inner_aupr=ap, criterion=(auc + ap) / 2,
                       correction_mean_abs=float(np.mean(np.abs(outputs['correction']))))
            trace.append(row)
            if epoch >= p['smoothing_window']:
                smooth = float(np.mean([r['criterion'] for r in trace[-p['smoothing_window']:]]))
                if smooth > best:
                    best, selected, saved = smooth, epoch, outputs['val'].copy()
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            # No evaluation of outer labels or queries occurs in the epoch loop.
            trace.append(row)
        if epoch % 150 == 0:
            print('EPOCH', stage, task, arm, cfg['id'], seed, epoch, round(time.monotonic() - started, 1), flush=True)
    if stage == 'outer':
        selected = epochs
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best = None
    model.load_state_dict(state)
    outputs = previous.predict(model, data, pairs, weights, labels)
    restore_error = 0.
    if saved is not None:
        np.testing.assert_allclose(outputs['val'], saved, atol=5e-5, rtol=5e-5)
        restore_error = float(np.max(np.abs(outputs['val'] - saved)))
        outputs['val'] = saved
    auc, ap = score(arrays['y_val'], outputs['val'])
    macro, eligible = macro_ap(arrays['y_val'], outputs['val'], arrays['pairs_val'])
    result = dict(task=task, arm=arm, seed=seed, config=cfg, config_id=cfg['id'], stage=stage,
        auc=auc, aupr=ap, macro_drug_ap=macro, eligible_drugs=eligible,
        selected_epoch=selected, selection_score=best, trained_epochs=epochs,
        initial_state_sha256=initial, initial_base_sha256=base_initial,
        params=sum(v.numel() for v in model.parameters()), restore_max_abs=restore_error,
        elapsed_seconds=time.monotonic() - started, protocol_sha256=p['sha256'], device=device)
    return result, outputs, trace, state


def run(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid worker shard')
    cache = {}
    for i, (task, arm, cfg, seed, stage, epochs) in enumerate(jobs(out, p, args.phase)):
        if i % args.shards != args.shard:
            continue
        dest = directory(out, task, arm, cfg['id'], seed, stage)
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (dest / 'result.json').exists():
                parent.checked_result(dest, p)
                print('SKIP', stage, task, arm, cfg['id'], seed, flush=True)
                continue
            if (task, stage) not in cache:
                cache[task, stage] = parent.load_stage(out, p, task, stage)
                if stage == 'outer':
                    # Audit lineage without using the outer evaluation labels.
                    arrays, _ = cache[task, stage]
                    with np.load(out / 'inputs' / task / 'indices.npz') as ids, np.load(out / 'inputs' / task / 'inner.npz') as inner:
                        for name, index in (('train', ids['fit']), ('val', ids['holdout'])):
                            np.testing.assert_array_equal(arrays['pairs_train'][:, index], inner['pairs_' + name])
                            np.testing.assert_array_equal(arrays['y_train'][index], inner['y_' + name])
            print('START', stage, task, arm, cfg['id'], seed, flush=True)
            arrays, graph = cache[task, stage]
            result, outputs, trace, state = train(arrays, graph, task, arm, seed, cfg, p, args.device, stage, epochs)
            np.savez_compressed(dest / 'scores.npz', **outputs, y_val=arrays['y_val'], pairs_val=arrays['pairs_val'])
            pd.DataFrame(trace).to_csv(dest / 'trace.csv', index=False)
            torch.save(state, dest / 'checkpoint.pt')
            result['artifacts_sha256'] = {n: sha256(dest / n) for n in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
            write_json(dest / 'result.json', result)
            print('DONE', stage, task, arm, cfg['id'], seed, '%.6f/%.6f epoch %d %.1fs' % (
                result['auc'], result['aupr'], result['selected_epoch'], result['elapsed_seconds']), flush=True)


def checked_inner(out, p, task, arm, cfg, seed):
    dest = directory(out, task, arm, cfg['id'], seed, 'inner')
    r = parent.checked_result(dest, p)
    expected = choose_epoch([pd.read_csv(dest / 'trace.csv')], p['smoothing_window'])
    if expected['epochs'] != r['selected_epoch']:
        raise ValueError('Invalid inner checkpoint')
    np.testing.assert_allclose(expected['selection_score'], r['selection_score'], atol=1e-12, rtol=0)
    return r, str((dest / 'result.json').relative_to(out)), sha256(dest / 'result.json')


def parameters(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    if (out / 'PARAMETERS.json').exists():
        print(json.dumps(read_decision(out, 'PARAMETERS.json', p)['chosen'], indent=2), flush=True)
        return
    chosen, hashes = {}, {}
    for task in p['tasks']:
        chosen[task] = {}
        for arm in ARMS:
            records = []
            for cfg in p['configurations'][task][arm]:
                r, name, digest = checked_inner(out, p, task, arm, cfg, p['tuning_seed'])
                records.append(r)
                hashes[name] = digest
            chosen[task][arm] = pick_inner(records)['config']
    freeze_decision(out, 'PARAMETERS.json', p, chosen=chosen, result_files_sha256=hashes)
    print(json.dumps(chosen, indent=2), flush=True)


def select(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    params = read_decision(out, 'PARAMETERS.json', p)
    if (out / 'SELECTION.json').exists():
        print(json.dumps(read_decision(out, 'SELECTION.json', p)['best'], indent=2), flush=True)
        return
    best, by_seed, hashes, preferred = {}, {}, {}, {}
    for task in p['tasks']:
        best[task], by_seed[task] = {}, {}
        for arm in ARMS:
            cfg = params['chosen'][task][arm]
            records = []
            for seed in p['seeds']:
                r, name, digest = checked_inner(out, p, task, arm, cfg, seed)
                records.append(r)
                hashes[name] = digest
            best[task][arm] = pick_inner(records)
            by_seed[task][arm] = {str(r['seed']): r for r in records}
        preferred[task] = max(ARMS, key=lambda a: best[task][a]['selection_score'])
    freeze_decision(out, 'SELECTION.json', p, parameters_sha256=params['sha256'],
        best=best, by_seed=by_seed, preferred_by_inner_validation=preferred, result_files_sha256=hashes)
    print(json.dumps(dict(preferred=preferred, best={t: {a: {k: r[k] for k in ('seed', 'config_id', 'auc', 'aupr', 'selected_epoch')}
        for a, r in arms.items()} for t, arms in best.items()}), indent=2), flush=True)


def report(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    params = read_decision(out, 'PARAMETERS.json', p)
    selected = read_decision(out, 'SELECTION.json', p)
    rows, replay, cache, initial = [], [], {}, {}
    all_jobs = [job for phase in ('screen', 'seeds', 'outer') for job in jobs(out, p, phase)]
    for task, arm, cfg, seed, stage, epochs in all_jobs:
        dest = directory(out, task, arm, cfg['id'], seed, stage)
        r = parent.checked_result(dest, p)
        trace = pd.read_csv(dest / 'trace.csv')
        np.testing.assert_array_equal(trace.epoch, np.arange(1, epochs + 1))
        expected_factor = [learning_factor(cfg['schedule'], e, p['training'][task]['train_epochs'], p['minimum_lr_fraction']) for e in trace.epoch]
        np.testing.assert_allclose(trace.learning_factor, expected_factor, atol=1e-12, rtol=0)
        np.testing.assert_allclose(trace.base_lr, np.asarray(expected_factor) * cfg['base_lr'], atol=1e-12, rtol=0)
        if stage == 'inner':
            checked_inner(out, p, task, arm, cfg, seed)
        elif r['selected_epoch'] != epochs or r['selection_score'] is not None or 'criterion' in trace:
            raise AssertionError('Outer data selected a checkpoint')
        if (task, stage) not in cache:
            cache[task, stage] = parent.load_stage(out, p, task, stage)
        arrays, graph = cache[task, stage]
        with np.load(dest / 'scores.npz') as z:
            np.testing.assert_array_equal(z['y_val'], arrays['y_val'])
            np.testing.assert_array_equal(z['pairs_val'], arrays['pairs_val'])
            scores = {k: z[k].copy() for k in ('val', 'prior', 'correction')}
        auc, ap = score(arrays['y_val'], scores['val'])
        ma, eligible = macro_ap(arrays['y_val'], scores['val'], arrays['pairs_val'])
        np.testing.assert_allclose([auc, ap, ma], [r['auc'], r['aupr'], r['macro_drug_ap']], atol=1e-12, rtol=0)
        if eligible != r['eligible_drugs']:
            raise AssertionError('Macro AP population changed')
        model, data, pairs, weights = build(arrays, graph, task, arm, seed, cfg, p, args.device)
        if state_digest(model) != r['initial_state_sha256']:
            raise AssertionError('Initialization mismatch')
        key = task, seed
        if key in initial and initial[key] != r['initial_base_sha256']:
            raise AssertionError('Unmatched original initialization')
        initial[key] = r['initial_base_sha256']
        model.load_state_dict(torch.load(dest / 'checkpoint.pt', map_location='cpu'))
        actual = previous.predict(model, data, pairs, weights, torch.from_numpy(arrays['y_train']).to(args.device))
        errors = {}
        for name in scores:
            np.testing.assert_allclose(actual[name], scores[name], atol=5e-5, rtol=5e-5)
            errors[name] = float(np.max(np.abs(actual[name] - scores[name])))
        r.update(selected_config=cfg['id'] == params['chosen'][task][arm]['id'],
                 selected_seed=seed == selected['best'][task][arm]['seed'])
        rows.append(r)
        replay.append(dict(task=task, arm=arm, config_id=cfg['id'], seed=seed, stage=stage, max_abs=errors))
        print('REPLAY', stage, task, arm, cfg['id'], seed, max(errors.values()), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'all_results.csv', index=False)
    inner = frame[(frame.stage == 'inner') & frame.selected_config]
    inner.to_csv(out / 'inner_all_seeds.csv', index=False)
    inner[inner.selected_seed].to_csv(out / 'inner_best_single.csv', index=False)
    inner.groupby(['task', 'arm'])[['auc', 'aupr', 'macro_drug_ap']].agg(['mean', 'std']).to_csv(out / 'inner_mean_std.csv')
    outer = frame[frame.stage == 'outer']
    outer.to_csv(out / 'outer_all_refits.csv', index=False)
    outer[outer.selected_seed].to_csv(out / 'outer_selected_single.csv', index=False)
    comparisons = []
    for task in p['tasks']:
        for stage in ('inner', 'outer'):
            block = inner if stage == 'inner' else outer
            a = block[(block.task == task) & (block.arm == 'additive')].set_index('seed').sort_index()
            b = block[(block.task == task) & (block.arm == 'pca_query')].set_index('seed').loc[a.index]
            da, dp, dm = a.auc - b.auc, a.aupr - b.aupr, a.macro_drug_ap - b.macro_drug_ap
            comparisons.append(dict(task=task, stage=stage, seeds=len(a), delta_auc=float(da.mean()),
                delta_aupr=float(dp.mean()), delta_macro_ap=float(dm.mean()), positive_auc_seeds=int((da > 0).sum()),
                positive_ap_seeds=int((dp > 0).sum()), scope='all five declared seeds' if stage == 'inner' else 'union of validation-selected seeds; not a random-seed average'))
    pd.DataFrame(comparisons).to_csv(out / 'paired_comparisons.csv', index=False)
    write_json(out / 'VALIDATION.json', dict(passed=True, smoke=p['smoke'], runs=len(rows),
        inner_runs=int((frame.stage == 'inner').sum()), outer_runs=int((frame.stage == 'outer').sum()),
        checkpoint_replays=len(replay), protocol_sha256=p['sha256'], parameters_sha256=params['sha256'],
        selection_sha256=selected['sha256'], no_test_evaluation=True, replay=replay))
    print('BEST INNER', inner[inner.selected_seed][['task', 'arm', 'seed', 'auc', 'aupr']].to_string(index=False), flush=True)
    print('SELECTED OUTER', outer[outer.selected_seed][['task', 'arm', 'seed', 'auc', 'aupr']].to_string(index=False), flush=True)


def export(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    selected = read_decision(out, 'SELECTION.json', p)
    validation = json.loads((out / 'VALIDATION.json').read_text())
    if not validation['passed'] or validation['smoke']:
        raise ValueError('Export requires a verified full experiment')
    for task in p['tasks']:
        for arm in ARMS:
            pick = selected['best'][task][arm]
            dest = out / 'best' / task / arm
            if dest.exists():
                raise ValueError('Preserve the existing export')
            dest.mkdir(parents=True)
            artifacts, metrics = {}, {}
            for stage in ('inner', 'outer'):
                source = directory(out, task, arm, pick['config_id'], pick['seed'], stage)
                result = parent.checked_result(source, p)
                metrics[stage] = {k: result[k] for k in ('auc', 'aupr', 'macro_drug_ap', 'selected_epoch')}
                for name in ('checkpoint.pt', 'scores.npz'):
                    target = dest / (stage + '_' + name)
                    shutil.copy2(source / name, target)
                    artifacts[target.name] = sha256(target)
            write_json(dest / 'configuration.json', dict(task=task, arm=arm, seed=pick['seed'],
                configuration=pick['config'], width=p['width'], depth=p['depth'],
                protocol_sha256=p['sha256'], selection_sha256=selected['sha256'],
                metrics=metrics, artifacts_sha256=artifacts,
                interpretation='One seed selected using inner validation. Five-seed means are separately available in inner_mean_std.csv. Outer results did not select this export.'))
    print('EXPORTED', out / 'best', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'parameters', 'select', 'report', 'export'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--out-dir', required=True)
        if name == 'prepare':
            cmd.add_argument('--reference-dir', default='improve/results/rectangle_completion_20260910')
            cmd.add_argument('--smoke', action='store_true')
        if name in ('run', 'report'):
            cmd.add_argument('--device', default='cuda:0')
        if name == 'run':
            cmd.add_argument('--phase', choices=('screen', 'seeds', 'outer'), required=True)
            cmd.add_argument('--shard', type=int, default=0)
            cmd.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
