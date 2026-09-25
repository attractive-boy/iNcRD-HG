"""Task-episodic adaptation of the official TabICLv2 checkpoint.

prepare -> inner (two freezing policies, three seeds, three saved steps) ->
select (including step zero) -> outer fixed-step refit -> verify -> report.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from improve import tabular_foundation as base
from improve import tabicl_adaptation_modules as kernels


DEFAULT_OUT = 'improve/results/tabicl_adaptation_20260912'
OWN = ('tabicl_adaptation.py', 'tabicl_adaptation_modules.py', 'test_tabicl_adaptation.py', 'tabular_foundation.py')
METRICS = base.METRICS


def sources():
    return {name: base.digest(Path(__file__).with_name(name)) for name in OWN}


def recipes(p):
    return [dict(id='frozen', mode='frozen', steps=0)] + [dict(id=f'{mode}_{step:03d}', mode=mode, steps=step)
            for mode in p['modes'] for step in p['milestones']]


def directory(out, phase, task, recipe, seed):
    return Path(out)/phase/task/recipe['id']/f's{seed}'


def prepare(args):
    out, parent = Path(args.out_dir).resolve(), Path(args.parent_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    tp = base.protocol(parent)
    ts = base.selection(parent, tp)
    v = base.read_json(parent/'VALIDATION.json')
    if tp['smoke'] or not v['passed'] or not v['all_estimators_replayed'] or v['protocol_sha256'] != tp['sha256']:
        raise ValueError('Need the verified formal TabICLv2 reference')
    seeds = [666] if args.smoke else list(tp['seeds'])
    files = {name: base.digest(parent/name) for name in ('protocol.json', 'SELECTION.json', 'VALIDATION.json')}
    baseline = {phase: {} for phase in ('inner', 'outer')}
    for phase in baseline:
        for task in tp['tasks']:
            baseline[phase][task] = {}
            for seed in seeds:
                relative = f"{phase}/{task}/{ts['chosen'][task]['config']['id']}/s{seed}"
                r = base.checked_result(parent/relative, tp)
                if (r['phase'], r['task'], r['seed'], r['config']) != (phase, task, seed, ts['chosen'][task]['config']):
                    raise ValueError('Wrong frozen baseline identity')
                baseline[phase][task][str(seed)] = relative
                for name in ('result.json', 'scores.npz'):
                    files[relative+'/'+name] = base.digest(parent/relative/name)
    loader = base.estimator(dict(n_estimators=2), 666, tp, 'cpu')
    loader._resolve_device()
    loader._load_model()
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        tasks=tp['tasks'], seeds=seeds, modes=['icl', 'all'], milestones=[1, 2] if args.smoke else [8, 24, 48],
        horizon_steps=2 if args.smoke else 48, learning_rate=1e-5, weight_decay=.01,
        chunk_size=128 if args.smoke else 2048, training_members=2,
        inference_members=2 if args.smoke else 32, source_sha256=sources(), parent_dir=str(parent),
        parent_protocol_sha256=tp['sha256'], reference_files=files, baseline=baseline,
        input_configs={task:ts['chosen'][task]['config'] for task in tp['tasks']},
        original_parameter_sha256=base.parameter_digest(loader), original_group_sha256=kernels.group_digests(loader.model_),
        original_parameters=sum(x.numel() for x in loader.model_.parameters()),
        training='Use official TabICL meta-batch construction: fresh shuffled fitting-only chunks, stratified 80/20 context/query split, two none/power feature/class permutation views, query cross-entropy, AdamW, 10% warmup then cosine schedule, FP32 and gradient norm clipping at 1. Freeze column and row encoders for icl; update all weights for all.',
        selection='For each task choose mode and saved step by mean inner (AUC+AP)/2 over all seeds, including unchanged frozen TabICLv2. Exact ties prefer fewer steps then recipe ID. All saved steps use 32 inference members in the formal experiment.',
        outer='Freeze both task selections before loading outer arrays; start again from official weights and train only the frozen step count. Preserve the original 48-step LR horizon, with no outer validation during updates.',
        primary='Both tasks must improve mean AUC and AP over the selected frozen TabICLv2 baseline, with at least 2/3 positive seed-label differences per metric.',
        no_test_evaluation=True, historical_outer_single_model_results_already_seen=True,
        smoke_selection='Fixed icl final step for LncDrug and all final step for MiDrug to exercise both outer refit paths; never interpret smoke metrics.' if args.smoke else None,
        limitations=['History-reused validation, not independent testing.',
                    'Three adaptation random seeds share the same pretrained checkpoint and data split.',
                    'The fitting-row graph features use the original five-fold masking; training pseudo-queries are fitting data, not new external labels.',
                    'This is an adaptation of the official episodic training mechanism; the public high-level trainer is not used.']))
    del loader
    out.mkdir(parents=True, exist_ok=True)
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name), out/'source_snapshot'/name)
    base.write_json(out/'protocol.json', p)
    print('FROZEN', out, flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out)/'protocol.json'))
    if p['source_sha256'] != sources():
        raise ValueError('Frozen adaptation source changed')
    tp = base.protocol(p['parent_dir'])
    if tp['sha256'] != p['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    for name, expected in p['reference_files'].items():
        if base.digest(Path(p['parent_dir'])/name) != expected:
            raise ValueError('Frozen baseline artifact changed')
    return p


def checked_result(dest, p):
    r = base.read_json(Path(dest)/'result.json')
    if r['protocol_sha256'] != p['sha256']:
        raise ValueError('Adaptation result belongs to another protocol')
    for name, expected in r['artifacts_sha256'].items():
        if base.digest(Path(dest)/name) != expected:
            raise ValueError('Adaptation artifact changed')
    return r


def selection(out, p):
    chosen = base.check_seal(base.read_json(Path(out)/'SELECTION.json'))
    if chosen['protocol_sha256'] != p['sha256'] or set(chosen['chosen']) != set(p['tasks']):
        raise ValueError('Wrong adaptation selection')
    for name, expected in chosen['artifacts_sha256'].items():
        if base.digest(Path(out)/name) != expected:
            raise ValueError('Frozen inner selection input changed')
    if any(x['recipe'] not in recipes(p) for x in chosen['chosen'].values()):
        raise ValueError('Unregistered adaptation recipe')
    return chosen


def features(out, p, task, phase):
    if phase not in ('inner', 'outer') or task not in p['tasks']:
        raise ValueError('Unknown feature task/phase')
    if phase == 'outer':
        selection(out, p)  # Before the parent loader can open outer arrays.
    tp = base.read_json(Path(p['parent_dir'])/'protocol.json')
    a = base.load_features(p['parent_dir'], tp, task, phase)
    if p['smoke']:
        for part, budget in (('train', 128), ('val', 40)):
            keep = base.balanced_subset(a['y_'+part], budget, 20260912)
            for prefix in ('raw', 'x', 'y'):
                a[prefix+'_'+part] = a[prefix+'_'+part][keep]
            a['pairs_'+part] = a['pairs_'+part][:, keep]
    base.validate_arrays(a)
    return a


def input_view(p, task, a):
    prefix = 'raw' if p['input_configs'][task]['input_view'] == 'raw' else 'x'
    return a[prefix+'_train'], a[prefix+'_val']


def new_estimator(p, task, seed, a, device):
    tp = base.read_json(Path(p['parent_dir'])/'protocol.json')
    clf = base.estimator(dict(n_estimators=p['inference_members']), seed, tp, device)
    x, _ = input_view(p, task, a)
    base.fit_context(clf, x, a['y_train'])
    if base.parameter_digest(clf) != p['original_parameter_sha256']:
        raise ValueError('Adaptation did not start from the official weights')
    return clf


def baseline_result(out, p, task, phase, seed, a, device=None):
    if p['smoke']:
        dest = Path(out)/'baseline'/phase/task/f's{seed}'
        if not (dest/'result.json').exists():
            if device is None:
                raise ValueError('Smoke baseline must be computed before selection/report')
            clf = new_estimator(p, task, seed, a, device)
            _, query = input_view(p, task, a)
            score = base.query_scores(clf, query)
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/'scores.npz', val=score, y_val=a['y_val'], pairs_val=a['pairs_val'])
            base.write_json(dest/'result.json', dict(protocol_sha256=p['sha256'], task=task, phase=phase, seed=seed,
                **base.metrics(a['y_val'], score, a['pairs_val']), artifacts_sha256={'scores.npz':base.digest(dest/'scores.npz')}))
            del clf
            gc.collect()
            torch.cuda.empty_cache()
        r = checked_result(dest, p)
    else:
        dest = Path(p['parent_dir'])/p['baseline'][phase][task][str(seed)]
        r = base.read_json(dest/'result.json')
    with np.load(dest/'scores.npz', allow_pickle=False) as z:
        if set(z.files) != {'val', 'y_val', 'pairs_val'}:
            raise ValueError('Baseline must contain validation-only arrays')
        np.testing.assert_array_equal(z['y_val'], a['y_val'])
        np.testing.assert_array_equal(z['pairs_val'], a['pairs_val'])
        score = z['val'].copy()
    for key, value in base.metrics(a['y_val'], score, a['pairs_val']).items():
        if value is not None:
            np.testing.assert_allclose(r[key], value, atol=1e-12, rtol=0)
    return dict(task=task, phase=phase, seed=seed, recipe_id='frozen', mode='frozen', steps=0,
                **base.metrics(a['y_val'], score, a['pairs_val'])), score


def snapshot(out, p, task, phase, recipe, seed, a, clf, trace, started, device):
    dest = directory(out, phase, task, recipe, seed)
    parameter_sha = base.parameter_digest(clf)
    groups = kernels.group_digests(clf.model_)
    if parameter_sha == p['original_parameter_sha256']:
        raise ValueError('Adapted checkpoint still has the original parameters')
    if recipe['mode'] == 'icl' and any(groups[key] != p['original_group_sha256'][key] for key in ('col_embedder', 'row_interactor')):
        raise ValueError('A frozen encoder changed during adaptation')
    if (dest/'result.json').exists():
        r = checked_result(dest, p)
        if r['parameter_sha256'] != parameter_sha:
            raise ValueError('Partial trajectory rerun produced different weights')
        return
    _, query = input_view(p, task, a)
    clf.model_.eval()
    torch.cuda.synchronize(device)
    prediction_started = time.monotonic()
    score = base.query_scores(clf, query)
    torch.cuda.synchronize(device)
    prediction_seconds = time.monotonic()-prediction_started
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'scores.npz', val=score, y_val=a['y_val'], pairs_val=a['pairs_val'])
    # Embedded weights are essential: the estimator's source model_path still
    # names the untouched official checkpoint, not this adapted state.
    clf.save(dest/'fitted.pkl', save_model_weights=True, save_training_data=True, save_kv_cache=False)
    base.write_json(dest/'trace.json', trace)
    r = dict(protocol_sha256=p['sha256'], task=task, phase=phase, seed=seed,
        recipe_id=recipe['id'], mode=recipe['mode'], steps=recipe['steps'], recipe=recipe,
        inference_members=p['inference_members'], parameter_sha256=parameter_sha, group_sha256=groups,
        feature_manifest_sha256=a['feature_manifest_sha256'], context_rows=len(a['y_train']),
        trainable_parameters=sum(x.numel() for x in clf.model_.parameters() if x.requires_grad),
        elapsed_seconds=time.monotonic()-started, checkpoint_inference_seconds=prediction_seconds,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device), reused_frozen=False,
        **base.metrics(a['y_val'], score, a['pairs_val']),
        artifacts_sha256={name:base.digest(dest/name) for name in ('scores.npz', 'fitted.pkl', 'trace.json')})
    base.write_json(dest/'result.json', r)
    print('SAVED', phase, task, recipe['id'], seed, r['auc'], r['ap'], flush=True)


def train_trajectory(out, p, task, phase, mode, seed, steps, a, device):
    targets = [dict(id=f'{mode}_{step:03d}', mode=mode, steps=step) for step in steps]
    if all((directory(out, phase, task, cfg, seed)/'result.json').exists() for cfg in targets):
        for cfg in targets:
            checked_result(directory(out, phase, task, cfg, seed), p)
        print('ALREADY_DONE', phase, task, mode, seed, flush=True)
        return
    clf = new_estimator(p, task, seed, a, device)
    kernels.set_trainable(clf.model_, mode)
    optimizer = torch.optim.AdamW([x for x in clf.model_.parameters() if x.requires_grad],
                                  lr=p['learning_rate'], weight_decay=p['weight_decay'])
    fitting, _ = input_view(p, task, a)
    batches = kernels.meta_batches(fitting, a['y_train'], seed=seed,
                                   chunk_size=p['chunk_size'], n_estimators=p['training_members'])
    trace, started = [], time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, max(steps)+1):
        lr = kernels.learning_rate(step-1, p['horizon_steps'], p['learning_rate'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        batch, epoch = next(batches)
        row = kernels.update(clf.model_, batch, optimizer, device)
        trace.append(dict(step=step, epoch=epoch, learning_rate=lr, **row))
        if step == 1 or step % 8 == 0:
            print('STEP', phase, task, mode, seed, step, 'loss', round(row['loss'], 6), flush=True)
        if step in steps:
            recipe = dict(id=f'{mode}_{step:03d}', mode=mode, steps=step)
            with kernels.inference_rng_guard(device):
                snapshot(out, p, task, phase, recipe, seed, a, clf, trace, started, device)
    del clf, optimizer, batches, batch
    gc.collect()
    torch.cuda.empty_cache()


def run(args):
    out, p = Path(args.out_dir).resolve(), protocol(args.out_dir)
    task, phase, device = args.task, args.phase, args.device
    if task not in p['tasks']:
        raise ValueError('Choose a registered task')
    chosen = selection(out, p)['chosen'][task]['recipe'] if phase == 'outer' else None
    torch.cuda.set_device(device)
    with (out/f'{phase}_{task}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a = features(out, p, task, phase)
        for seed in p['seeds']:
            baseline, score = baseline_result(out, p, task, phase, seed, a, device)
            if phase == 'outer' and chosen['steps'] == 0:
                dest = directory(out, phase, task, chosen, seed)
                if (dest/'result.json').exists():
                    checked_result(dest, p)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest/'scores.npz', val=score, y_val=a['y_val'], pairs_val=a['pairs_val'])
                base.write_json(dest/'trace.json', [])
                base.write_json(dest/'result.json', dict(baseline, protocol_sha256=p['sha256'],
                    recipe=chosen, reused_frozen=True, parameter_sha256=p['original_parameter_sha256'],
                    group_sha256=p['original_group_sha256'], context_rows=len(a['y_train']),
                    feature_manifest_sha256=a['feature_manifest_sha256'], inference_members=p['inference_members'],
                    artifacts_sha256={name:base.digest(dest/name) for name in ('scores.npz', 'trace.json')}))
                print('REUSED_FROZEN', task, seed, flush=True)
            else:
                modes = p['modes'] if phase == 'inner' else [chosen['mode']]
                for mode in modes:
                    steps = p['milestones'] if phase == 'inner' else [chosen['steps']]
                    train_trajectory(out, p, task, phase, mode, seed, steps, a, device)


def choose(rows, candidates, seeds):
    if len(rows) != len(candidates)*len(seeds):
        raise ValueError('Incomplete adaptation selection grid')
    choices = []
    for cfg in candidates:
        group = [r for r in rows if r['recipe_id'] == cfg['id']]
        if len(group) != len(seeds) or {r['seed'] for r in group} != set(seeds):
            raise ValueError('Missing or duplicate adaptation seed')
        values = {key:float(np.mean([r[key] for r in group])) if all(r[key] is not None for r in group) else None
                  for key in (*METRICS, 'criterion')}
        if not all(np.isfinite(values[key]) for key in ('auc', 'ap', 'criterion')):
            raise ValueError('Nonfinite inner selection values')
        choices.append(dict(recipe=cfg, **values))
    return min(choices, key=lambda r:(-r['criterion'], r['recipe']['steps'], r['recipe']['id']))


def inner_rows(out, p):
    rows, hashes, chosen = [], {}, {}
    for task in p['tasks']:
        a = features(out, p, task, 'inner')
        group = []
        for cfg in recipes(p):
            for seed in p['seeds']:
                if cfg['steps'] == 0:
                    r, _ = baseline_result(out, p, task, 'inner', seed, a)
                    if p['smoke']:
                        path = out/'baseline'/'inner'/task/f's{seed}'/'result.json'
                        hashes[str(path.relative_to(out))] = base.digest(path)
                else:
                    dest = directory(out, 'inner', task, cfg, seed)
                    r = checked_result(dest, p)
                    if (r['phase'], r['task'], r['seed'], r['recipe']) != ('inner', task, seed, cfg):
                        raise ValueError('Wrong inner checkpoint identity')
                    hashes[str((dest/'result.json').relative_to(out))] = base.digest(dest/'result.json')
                group.append({key:r[key] for key in ('task', 'phase', 'seed', 'recipe_id', 'mode', 'steps', *METRICS, 'criterion')})
        selected = choose(group, recipes(p), p['seeds'])
        if p['smoke']:
            wanted = 'icl' if task == 'LncDrug' else 'all'
            cfg = next(c for c in recipes(p) if c['mode'] == wanted and c['steps'] == p['milestones'][-1])
            selected = choose([r for r in group if r['recipe_id'] == cfg['id']], [cfg], p['seeds'])
        chosen[task] = selected
        rows.extend(group)
    return rows, hashes, chosen


def select(out):
    p = protocol(out)
    if (out/'SELECTION.json').exists():
        selection(out, p)
        return
    rows, hashes, chosen = inner_rows(out, p)
    pd.DataFrame(rows).to_csv(out/'inner_results.csv', index=False)
    pd.DataFrame(rows).groupby(['task', 'recipe_id', 'mode', 'steps'])[list(METRICS)+['criterion']].agg(['mean', 'std']).to_csv(out/'inner_summary.csv')
    hashes.update({name:base.digest(out/name) for name in ('inner_results.csv', 'inner_summary.csv')})
    base.write_json(out/'SELECTION.json', base.seal(dict(protocol_sha256=p['sha256'], chosen=chosen,
        created_at_utc=datetime.now(timezone.utc).isoformat(), artifacts_sha256=hashes)))
    print('FROZEN_SELECTION', chosen, flush=True)


def expected_jobs(p, chosen, task):
    for cfg in recipes(p):
        if cfg['steps']:
            for seed in p['seeds']:
                yield 'inner', cfg, seed
    for seed in p['seeds']:
        yield 'outer', chosen['chosen'][task]['recipe'], seed


def verify_task(args):
    from tabicl import TabICLClassifier
    out, p = Path(args.out_dir).resolve(), protocol(args.out_dir)
    chosen = selection(out, p)
    task, device = args.task, args.device
    if task not in p['tasks']:
        raise ValueError('Unknown verification task')
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data, checked = {}, []
    for phase, cfg, seed in expected_jobs(p, chosen, task):
        if phase not in data:
            data[phase] = features(out, p, task, phase)
        a = data[phase]
        dest = directory(out, phase, task, cfg, seed)
        r = checked_result(dest, p)
        if (r['task'], r['phase'], r['seed'], r['recipe']) != (task, phase, seed, cfg):
            raise ValueError('Saved adaptation checkpoint identity differs')
        if r['feature_manifest_sha256'] != a['feature_manifest_sha256'] or r['context_rows'] != len(a['y_train']):
            raise ValueError('Wrong graph feature context')
        with np.load(dest/'scores.npz', allow_pickle=False) as z:
            if set(z.files) != {'val', 'y_val', 'pairs_val'}:
                raise ValueError('Unexpected score arrays')
            np.testing.assert_array_equal(z['y_val'], a['y_val'])
            np.testing.assert_array_equal(z['pairs_val'], a['pairs_val'])
            score = z['val'].copy()
        for key, value in base.metrics(a['y_val'], score, a['pairs_val']).items():
            if value is not None:
                np.testing.assert_allclose(r[key], value, atol=1e-12, rtol=0)
        trace = base.read_json(dest/'trace.json')
        if len(trace) != cfg['steps']:
            raise ValueError('Saved training prefix has wrong length')
        for i, row in enumerate(trace):
            if row['step'] != i+1 or row['learning_rate'] != kernels.learning_rate(i, p['horizon_steps'], p['learning_rate']):
                raise ValueError('Outer or inner optimizer schedule changed')
            if not np.isfinite(row['loss']) or not np.isfinite(row['grad_norm']):
                raise ValueError('Nonfinite training history')
        if cfg['steps'] == 0:
            _, replay = baseline_result(out, p, task, phase, seed, a)
            if not r['reused_frozen']:
                raise ValueError('Step zero must identify baseline reuse')
        else:
            if r['reused_frozen'] or r['parameter_sha256'] == p['original_parameter_sha256']:
                raise ValueError('Adapted state did not change weights')
            clf = TabICLClassifier.load(dest/'fitted.pkl', device=device)
            if clf.n_estimators != p['inference_members'] or clf.random_state != seed or clf.allow_auto_download:
                raise ValueError('Inference settings differ from the protocol')
            if base.parameter_digest(clf) != r['parameter_sha256'] or kernels.group_digests(clf.model_) != r['group_sha256']:
                raise ValueError('Saved estimator lost its adapted weights')
            if cfg['mode'] == 'icl' and any(r['group_sha256'][key] != p['original_group_sha256'][key] for key in ('col_embedder', 'row_interactor')):
                raise ValueError('Frozen encoders changed')
            if kernels.set_trainable(clf.model_, cfg['mode']) != r['trainable_parameters']:
                raise ValueError('Wrong trainable parameter count')
            clf.model_.eval()
            fitting, query = input_view(p, task, a)
            np.testing.assert_array_equal(clf.ensemble_generator_.y_, a['y_train'])
            expected_context = clf.ensemble_generator_.unique_filter_.transform(clf.X_encoder_.transform(fitting))
            np.testing.assert_allclose(clf.ensemble_generator_.X_, expected_context, atol=1e-7, rtol=1e-7)
            replay = base.query_scores(clf, query)
            if p['smoke']:
                permuted = base.query_scores(clf, query[::-1])[::-1]
                split = np.concatenate([base.query_scores(clf, q) for q in np.array_split(query, 2)])
                np.testing.assert_allclose(permuted, replay, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(split, replay, atol=1e-5, rtol=1e-5)
            del clf
            gc.collect()
            torch.cuda.empty_cache()
        np.testing.assert_allclose(replay, score, atol=2e-6, rtol=2e-6)
        checked.append(dict(path=str(dest.relative_to(out)), result_sha256=base.digest(dest/'result.json'),
                            adapted=bool(cfg['steps']), max_abs=float(np.max(np.abs(replay-score)))))
        print('REPLAYED', phase, task, cfg['id'], seed, checked[-1]['max_abs'], flush=True)
    base.write_json(out/f'VERIFY_{task}.json', base.seal(dict(passed=True, task=task, protocol_sha256=p['sha256'],
        selection_sha256=chosen['sha256'], replays=checked)))


def report(out):
    p = protocol(out)
    chosen = selection(out, p)
    rows, _, recomputed = inner_rows(out, p)
    if recomputed != chosen['chosen']:
        raise ValueError('Cannot reproduce internal selection')
    pd.testing.assert_frame_equal(pd.read_csv(out/'inner_results.csv'), pd.DataFrame(rows), check_dtype=False, atol=1e-12, rtol=0)
    replayed, results = [], []
    for task in p['tasks']:
        check = base.check_seal(base.read_json(out/f'VERIFY_{task}.json'))
        if not check['passed'] or (check['protocol_sha256'], check['selection_sha256']) != (p['sha256'], chosen['sha256']):
            raise ValueError('Verification belongs to another experiment')
        expected = {str(directory(out, phase, task, cfg, seed).relative_to(out)) for phase, cfg, seed in expected_jobs(p, chosen, task)}
        if len(check['replays']) != len(expected) or {r['path'] for r in check['replays']} != expected:
            raise ValueError('Not all adapted states were replayed')
        for item in check['replays']:
            if base.digest(out/item['path']/'result.json') != item['result_sha256']:
                raise ValueError('Verified checkpoint result changed')
        replayed.extend(check['replays'])
        a = features(out, p, task, 'outer')
        for seed in p['seeds']:
            old, _ = baseline_result(out, p, task, 'outer', seed, a)
            new = checked_result(directory(out, 'outer', task, chosen['chosen'][task]['recipe'], seed), p)
            for method, r in (('tabicl_frozen', old), ('tabicl_adapted_selection', new)):
                results.append(dict(task=task, seed=seed, method=method, **{key:r[key] for key in METRICS}))
    frame = pd.DataFrame(results)
    frame.to_csv(out/'outer_results.csv', index=False)
    summary = frame.groupby(['task', 'method'])[list(METRICS)].agg(['mean', 'std'])
    summary.to_csv(out/'summary.csv')
    deltas, gates = [], {}
    for task in p['tasks']:
        old = frame[(frame.task == task) & (frame.method == 'tabicl_frozen')].set_index('seed')
        new = frame[(frame.task == task) & (frame.method == 'tabicl_adapted_selection')].set_index('seed')
        d = new[list(METRICS)]-old[list(METRICS)]
        means = {key:float(d[key].mean()) if d[key].notna().any() else None for key in METRICS}
        counts = {key:int((d[key] > 0).sum()) for key in ('auc', 'ap')}
        gates[task] = dict(mean_deltas=means, positive_seed_labels=counts,
                          passed=not p['smoke'] and all(means[key] > 0 and counts[key] >= 2 for key in ('auc', 'ap')))
        deltas.extend(dict(task=task, seed=int(seed), **r.to_dict()) for seed, r in d.iterrows())
    pd.DataFrame(deltas).to_csv(out/'paired_deltas.csv', index=False)
    lines = ['# 官方 TabICLv2：任务内适配', '',
        '**工程短流程，固定配置覆盖训练/重放路径，不用于性能结论。**' if p['smoke'] else '同一历史外层验证、seed 666/0/1 的均值。模式与步数只在内部验证选择；外层固定步数重训后评估。', '',
        '| 任务 | 方案 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for method, name in (('tabicl_frozen', '原 TabICLv2'), ('tabicl_adapted_selection', '内部选定适配')):
            r = summary.loc[(task, method)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f |' % (task, name, *(r[(key, 'mean')] for key in METRICS)))
    lines += ['', '## 冻结选择', '', '| 任务 | 模式 | 更新步数 | 内部 AUC | 内部 AP |', '|---|---|---:|---:|---:|']
    for task, value in chosen['chosen'].items():
        lines.append('| %s | %s | %d | %.6f | %.6f |' % (task, value['recipe']['mode'], value['recipe']['steps'], value['auc'], value['ap']))
    lines += ['', '## 相对原 TabICLv2 的变化', '', '| 任务 | ΔAUC | 正向种子标签 | ΔAP | 正向种子标签 |', '|---|---:|---:|---:|---:|']
    for task, value in gates.items():
        lines.append('| %s | %+.6f | %d/%d | %+.6f | %d/%d |' % (task, value['mean_deltas']['auc'], value['positive_seed_labels']['auc'], len(p['seeds']), value['mean_deltas']['ap'], value['positive_seed_labels']['ap'], len(p['seeds'])))
    adapted_count = sum(r['adapted'] for r in replayed)
    outer_runs = sum(value['recipe']['steps'] > 0 for value in chosen['chosen'].values())*len(p['seeds'])
    inner_runs = len(p['tasks'])*len(p['modes'])*len(p['seeds'])
    joint = all(value['passed'] for value in gates.values())
    lines += ['', '## 核验与适用范围', '',
        '- 双任务 AUC/AP 共同提高的预设判定：`%s`。每项均值增量为正，且至少 2/3 个种子标签提高。' % joint,
        '- %d 条内部适配轨迹、%d 次外层固定步数适配；%d 个实际更新的模型状态全部重载并重放。中间保存状态不重复计为独立训练轨迹。' % (inner_runs, outer_runs, adapted_count),
        '- icl 冻结列编码器与行编码器，all 更新全部参数。官方元批次由拟合行分出上下文与伪查询，只有上下文标签输入模型，伪查询标签仅用于交叉熵。',
        '- 继续使用原有 36 维五折图特征及任务对应输入变换。每个实际评估状态的推理成员数相同；正式均为 32。',
        '- 外层从官方权重重新开始，使用内部冻结的更新步数和原完整学习率日程，更新期间不评价外层。零步选择直接复用既有预测。',
        '- 拟合状态自包含适配后的模型权重。核对原始权重、输入/标签/配对、更新轨迹、冻结编码器、参数哈希及重放分数；重新计算内部选择。',
        '- 使用官方 meta-batch 构造与分类目标，自写训练调度以固定选参边界；不是直接调用官方高层 fine-tune 类。原包、环境和旧实验源码均保留。',
        '- 三个随机种子共享官方预训练检查点和数据划分，表征适配与推理随机性，不是独立预训练或独立数据重复。',
        '- 该历史验证多次参与开发，此前外层单模型结果已知。本轮没有读取测试数组，不据此声称独立泛化或统计显著性。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    base.write_json(out/'OUTCOME.json', dict(protocol_sha256=p['sha256'], smoke=p['smoke'], gates=gates, joint=joint, scope_complete=True))
    base.write_json(out/'VALIDATION.json', dict(passed=True, protocol_sha256=p['sha256'], smoke=p['smoke'],
        selection_replayed=True, all_adapted_estimators_replayed=True, baseline_predictions_aligned=True,
        no_test_evaluation=True, adapted_model_states=adapted_count, inner_training_trajectories=inner_runs,
        outer_training_trajectories=outer_runs, replays=replayed))
    print(summary.to_string(), flush=True)
    print('VERIFIED', adapted_count, 'adapted states; joint outcome:', joint, flush=True)


def pipeline(args):
    out, p = Path(args.out_dir).resolve(), protocol(args.out_dir)
    devices = [d.strip() for d in args.devices.split(',') if d.strip()]
    if not devices:
        raise ValueError('At least one device is required')
    for phase in ('inner', 'outer', 'verify'):
        for start in range(0, len(p['tasks']), len(devices)):
            workers = []
            for task, device in zip(p['tasks'][start:start+len(devices)], devices):
                command = [sys.executable, '-m', 'improve.tabicl_adaptation', 'verify' if phase == 'verify' else 'run',
                           '--out-dir', str(out), '--task', task, '--device', device]
                if phase != 'verify':
                    command += ['--phase', phase]
                workers.append(subprocess.Popen(command))
            codes = [worker.wait() for worker in workers]
            if any(code != 0 for code in codes):
                raise RuntimeError(f'Adaptation {phase} worker failed: {codes}')
        print('STAGE_DONE', phase, flush=True)
        if phase == 'inner':
            select(out)
    report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'select', 'verify', 'report', 'pipeline'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--parent-dir', default=base.DEFAULT_OUT)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--task', choices=('LncDrug', 'MiDrug'))
    parser.add_argument('--phase', choices=('inner', 'outer'), default='inner')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'run':
        run(args)
    elif args.command == 'verify':
        verify_task(args)
    elif args.command == 'pipeline':
        pipeline(args)
    else:
        {'select': select, 'report': report}[args.command](Path(args.out_dir).resolve())


if __name__ == '__main__':
    main()
