"""Graph-support matching for frozen TabICLv2, with paired context controls.

prepare -> rebuild inner views -> inner -> select -> rebuild outer views ->
outer -> replay every fitted context -> report. All prediction components and
the seven-candidate selection rule are fixed before evaluation.
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

from improve import tabular_foundation as base
from improve import tabicl_graph_context_modules as kernels


DEFAULT_OUT = 'improve/results/tabicl_graph_context_20260912'
DEFAULT_PARENT = 'improve/results/tabicl_20260912'
GRAPH_PYTHON = '/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python'
OWN = ('tabicl_graph_context.py', 'tabicl_graph_context_modules.py',
       'tabicl_graph_context_features.py', 'test_tabicl_graph_context.py')
KINDS = ('full', 'fold0', 'fold1', 'fold2', 'fold3', 'fold4')


def sources():
    return {name:base.digest(Path(__file__).with_name(name)) for name in OWN}


def prepare(args):
    out, parent = Path(args.out_dir).resolve(), Path(args.parent_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    tp = base.protocol(parent)
    ts = base.selection(parent, tp)
    validation = base.read_json(parent/'VALIDATION.json')
    if tp['smoke'] or not validation['passed'] or not validation['all_estimators_replayed']:
        raise ValueError('Need the verified formal TabICLv2 reference')
    if validation['protocol_sha256'] != tp['sha256']:
        raise ValueError('Wrong parent validation protocol')
    feature_root = Path(tp['reference_dir'])
    fp = base.check_seal(base.read_json(feature_root/'protocol.json'))
    graph_root = Path(fp['reference_dir'])
    gp = base.check_seal(base.read_json(graph_root/'protocol.json'))
    seeds = [666] if args.smoke else list(tp['seeds'])
    references = {name:base.digest(parent/name) for name in ('protocol.json', 'SELECTION.json', 'VALIDATION.json')}
    baseline, parameter_hashes = {}, set()
    for phase in ('inner', 'outer'):
        baseline[phase] = {}
        for task in tp['tasks']:
            baseline[phase][task] = {}
            for seed in seeds:
                relative = '%s/%s/%s/s%d' % (phase, task, ts['chosen'][task]['config']['id'], seed)
                r = base.checked_result(parent/relative, tp)
                if (r['phase'], r['task'], r['seed'], r['config']) != (phase, task, seed, ts['chosen'][task]['config']):
                    raise ValueError('Wrong baseline identity')
                parameter_hashes.add(r['parameter_sha256'])
                baseline[phase][task][str(seed)] = relative
                for name in ('result.json', 'scores.npz', 'fitted.pkl'):
                    references[relative+'/'+name] = base.digest(parent/relative/name)
    if len(parameter_hashes) != 1:
        raise ValueError('Baseline estimators do not share the official parameters')
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        tasks=tp['tasks'], seeds=seeds, recipes=kernels.recipes(), kinds=list(KINDS), folds=5,
        parent_dir=str(parent), parent_protocol_sha256=tp['sha256'], reference_files=references,
        baseline=baseline, input_configs={task:ts['chosen'][task]['config'] for task in tp['tasks']},
        feature_reference_dir=str(feature_root), feature_reference_protocol_sha256=fp['sha256'],
        graph_reference_dir=str(graph_root), graph_reference_protocol_sha256=gp['sha256'],
        graph_python=args.graph_python, graph_python_sha256=base.digest(args.graph_python),
        source_sha256=sources(), parameter_sha256=next(iter(parameter_hashes)),
        inference_members=2 if args.smoke else 32, no_test_evaluation=True,
        replay_atol=1e-6, replay_rtol=1e-6, query_invariance_atol=1e-5, query_invariance_rtol=1e-5,
        mechanism='Reconstruct each evaluation query on all five original fitting-fold support graphs. Compare the original all-fold context against per-fold contexts encoded on the same support graph. No pretrained parameter updates.',
        controls='For every per-fold context, predict both full-support and same-fold-support queries with exactly the same fitted estimator. This controls context sample count and model count. The full-context query bag uses identical preprocessing permutations across graph views.',
        recipes_note='Seven candidates: the unchanged baseline plus each of query_bag, fold_full and fold_match at probability weights 0.5 and 1.0; remaining weight is the unchanged baseline. All three standalone components are also reported, including the control.',
        selection='Use all three seed labels to maximize mean inner (AUC+AP)/2. Exact ties prefer lower new-component weight then recipe ID. Freeze both tasks before any own outer feature or array load. Every outer component is predeclared, never choose among outer scores.',
        input_policy='Reuse 36 original graph features and the previously selected input view. The existing quantile transform, where used, remains fitted on all OOF fitting rows; official per-estimator preprocessing is fitted only on that estimator context.',
        graph_policy='Target support consists only of fitting positives outside the context fold. Context-fold positive edges and all evaluation target edges are absent in both directions. Original other-task/circRNA side information remains under the prior single-task protocol. Do not pool task caches.',
        inference_policy='FP32, TF32 off, original batch size 4, identical seed per graph view, no KV cache. Concatenate query views into a single prediction call; validate query permutation and splitting in the smoke pipeline.',
        gate='Both tasks must improve mean AUC and AP over selected frozen TabICLv2, with at least 2/3 positive seed-label differences for each metric. Macro drug AP is reported separately.',
        smoke_policy='Use 32 balanced context rows per fold and 40 evaluation rows from full rebuilt graphs, one seed and two inference members; all paths are exercised, and no smoke metric is performance evidence.',
        limitations=['Historical inner and outer validation have repeatedly been used for development; no independent test inference.',
            'All inference seeds and graph views share a single pretrained checkpoint; neither is an independent pretrained model or data split.',
            'The graph-view ensemble costs additional inference and graph feature reconstruction.',
            'This is a local support-consistency hypothesis, not a reproduction of TFMLinker learnable cross-graph prototypes or its pretrained GNN.',
            'The unchanged baseline and its past outer results were known before this trial.']))
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name), out/'source_snapshot'/name)
    base.write_json(out/'protocol.json', p)
    print('FROZEN', out, flush=True)


def header(out):
    """Safe in both Python environments; does not import TabICL or open NPZs."""
    p = base.check_seal(base.read_json(Path(out)/'protocol.json'))
    if sources() != p['source_sha256']:
        raise ValueError('Frozen graph-context source changed')
    if base.digest(p['graph_python']) != p['graph_python_sha256']:
        raise ValueError('Historical feature Python changed')
    for name, expected in p['reference_files'].items():
        if base.digest(Path(p['parent_dir'])/name) != expected:
            raise ValueError('Frozen baseline artifact changed')
    return p


def protocol(out):
    p = header(out)
    tp = base.protocol(p['parent_dir'])
    if tp['sha256'] != p['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    return p


def selection(out, p):
    s = base.check_seal(base.read_json(Path(out)/'SELECTION.json'))
    if s['protocol_sha256'] != p['sha256'] or set(s['chosen']) != set(p['tasks']):
        raise ValueError('Wrong graph-context selection')
    for name, expected in s['artifacts_sha256'].items():
        if base.digest(Path(out)/name) != expected:
            raise ValueError('An inner selection artifact changed')
    if any(v['recipe'] not in p['recipes'] for v in s['chosen'].values()):
        raise ValueError('An unregistered recipe was selected')
    return s


def feature_manifest(out, p, task, phase):
    if phase == 'outer':
        selection(out, p)
    dest = Path(out)/'features'/phase/task
    m = base.read_json(dest/'manifest.json')
    if (m['protocol_sha256'], m['phase'], m['task']) != (p['sha256'], phase, task):
        raise ValueError('Wrong graph feature cache identity')
    original = Path(p['feature_reference_dir'])/'features'/phase/task/'manifest.json'
    if m['reference_manifest_sha256'] != base.digest(original):
        raise ValueError('Graph feature reference changed')
    for name, expected in m['files'].items():
        if base.digest(dest/name) != expected:
            raise ValueError('Matched graph features changed')
    return m


def features(out, p, task, phase):
    if phase not in ('inner', 'outer') or task not in p['tasks']:
        raise ValueError('Unknown feature task/phase')
    if phase == 'outer':
        selection(out, p)  # Must precede parent load_features and every np.load.
    m = feature_manifest(out, p, task, phase)
    tp = base.read_json(Path(p['parent_dir'])/'protocol.json')
    a = base.load_features(p['parent_dir'], tp, task, phase)
    with np.load(Path(out)/'features'/phase/task/'arrays.npz', allow_pickle=False) as archive:
        if set(archive.files) != {'raw_query_views', 'x_query_views', 'fold_ids', 'pairs_val'}:
            raise ValueError('Unexpected graph feature arrays')
        np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
        for key in ('raw_query_views', 'x_query_views', 'fold_ids'):
            a[key] = archive[key].copy()
    kernels.support_plan(a['pairs_train'], a['y_train'], a['fold_ids'], a['pairs_val'])
    for key in ('raw_query_views', 'x_query_views'):
        if a[key].shape != (5, len(a['y_val']), 36) or not np.isfinite(a[key]).all():
            raise ValueError('Expected five finite query-feature views')
    if p['smoke']:
        keep = np.sort(np.concatenate([np.flatnonzero(a['fold_ids'] == fold)[
            base.balanced_subset(a['y_train'][a['fold_ids'] == fold], 32, 20260912+fold)] for fold in range(5)]))
        for prefix in ('raw', 'x', 'y'):
            a[prefix+'_train'] = a[prefix+'_train'][keep]
        a['pairs_train'] = a['pairs_train'][:, keep]
        a['fold_ids'] = a['fold_ids'][keep]
        keep = base.balanced_subset(a['y_val'], 40, 20260912)
        for prefix in ('raw', 'x', 'y'):
            a[prefix+'_val'] = a[prefix+'_val'][keep]
        a['pairs_val'] = a['pairs_val'][:, keep]
        for key in ('raw_query_views', 'x_query_views'):
            a[key] = a[key][:, keep]
    base.validate_arrays(a)
    a['graph_manifest_sha256'] = base.digest(Path(out)/'features'/phase/task/'manifest.json')
    return a


def view(p, task, a):
    prefix = 'raw' if p['input_configs'][task]['input_view'] == 'raw' else 'x'
    return a[prefix+'_train'], a[prefix+'_val'], a[prefix+'_query_views']


def directory(out, phase, task, seed, kind):
    return Path(out)/phase/task/('s%d' % seed)/kind


def checked_result(dest, p):
    r = base.read_json(Path(dest)/'result.json')
    if r['protocol_sha256'] != p['sha256']:
        raise ValueError('Wrong graph-context result protocol')
    for name, expected in r['artifacts_sha256'].items():
        if base.digest(Path(dest)/name) != expected:
            raise ValueError('Graph-context inference artifact changed')
    return r


def context_and_queries(p, task, a, kind):
    x, full, query_views = view(p, task, a)
    if kind == 'full':
        return x, a['y_train'], np.concatenate([full[None], query_views], axis=0)
    if kind not in KINDS:
        raise ValueError('Unknown context kind')
    fold = int(kind[-1])
    keep = a['fold_ids'] == fold
    return x[keep], a['y_train'][keep], np.stack([full, query_views[fold]])


def baseline_score(p, phase, task, seed, a):
    if p['smoke']:
        raise ValueError('Smoke baseline must be computed from its own context')
    dest = Path(p['parent_dir'])/p['baseline'][phase][task][str(seed)]
    with np.load(dest/'scores.npz', allow_pickle=False) as archive:
        if set(archive.files) != {'val', 'y_val', 'pairs_val'}:
            raise ValueError('Baseline must contain validation-only arrays')
        np.testing.assert_array_equal(archive['y_val'], a['y_val'])
        np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
        score = archive['val'].copy()
    r = base.read_json(dest/'result.json')
    for key, value in base.metrics(a['y_val'], score, a['pairs_val']).items():
        if value is not None:
            np.testing.assert_allclose(r[key], value, atol=1e-12, rtol=0)
    return score


def run_job(out, p, phase, task, seed, kind, a, device):
    import torch
    dest = directory(out, phase, task, seed, kind)
    if (dest/'result.json').exists():
        checked_result(dest, p)
        return
    x, y, queries = context_and_queries(p, task, a, kind)
    tp = base.read_json(Path(p['parent_dir'])/'protocol.json')
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    clf = base.fit_context(base.estimator(dict(n_estimators=p['inference_members']), seed, tp, device), x, y)
    if base.parameter_digest(clf) != p['parameter_sha256']:
        raise ValueError('Official pretrained parameters changed')
    score = base.query_scores(clf, queries.reshape(-1, x.shape[1])).reshape(queries.shape[:2])
    if base.parameter_digest(clf) != p['parameter_sha256']:
        raise ValueError('Inference updated model parameters')
    baseline_error = None
    if kind == 'full' and not p['smoke']:
        original = baseline_score(p, phase, task, seed, a)
        np.testing.assert_allclose(score[0], original, atol=1e-6, rtol=1e-6)
        baseline_error = float(np.max(np.abs(score[0]-original)))
    dest.mkdir(parents=True, exist_ok=True)
    clf.save(dest/'fitted.pkl', save_model_weights=False, save_training_data=True, save_kv_cache=False)
    np.savez_compressed(dest/'scores.npz', views=score, y_val=a['y_val'], pairs_val=a['pairs_val'])
    r = dict(protocol_sha256=p['sha256'], phase=phase, task=task, seed=seed, kind=kind,
        context_rows=len(y), n_query_views=len(queries), inference_members=p['inference_members'],
        parameter_sha256=p['parameter_sha256'], graph_manifest_sha256=a['graph_manifest_sha256'],
        baseline_replay_max_abs=baseline_error, elapsed_seconds=time.monotonic()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device), pretrained_parameter_updates=0,
        artifacts_sha256={name:base.digest(dest/name) for name in ('scores.npz', 'fitted.pkl')})
    base.write_json(dest/'result.json', r)
    print('SAVED', phase, task, seed, kind, 'context', len(y), 'seconds', round(r['elapsed_seconds'], 1), flush=True)
    del clf
    gc.collect()
    torch.cuda.empty_cache()


def components(out, p, phase, task, seed, a):
    views, hashes = {}, {}
    for kind in KINDS:
        dest = directory(out, phase, task, seed, kind)
        r = checked_result(dest, p)
        if (r['phase'], r['task'], r['seed'], r['kind']) != (phase, task, seed, kind):
            raise ValueError('Wrong context-state identity')
        if r['graph_manifest_sha256'] != a['graph_manifest_sha256']:
            raise ValueError('Context state used different graph features')
        with np.load(dest/'scores.npz', allow_pickle=False) as archive:
            if set(archive.files) != {'views', 'y_val', 'pairs_val'}:
                raise ValueError('Unexpected prediction arrays')
            np.testing.assert_array_equal(archive['y_val'], a['y_val'])
            np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
            views[kind] = archive['views'].copy()
        expected = 6 if kind == 'full' else 2
        if views[kind].shape != (expected, len(a['y_val'])) or not np.isfinite(views[kind]).all():
            raise ValueError('Wrong query view count')
        hashes[str((dest/'result.json').relative_to(out))] = base.digest(dest/'result.json')
    full = views['full'][0] if p['smoke'] else baseline_score(p, phase, task, seed, a)
    result = dict(query_bag=kernels.average_views(views['full'][1:], len(full)),
                  fold_full=kernels.average_views(np.stack([views[k][0] for k in KINDS[1:]]), len(full)),
                  fold_match=kernels.average_views(np.stack([views[k][1] for k in KINDS[1:]]), len(full)))
    return full, result, hashes


def run(args):
    import torch
    out = Path(args.out_dir).resolve()
    p = protocol(out)
    if args.phase == 'outer':
        selection(out, p)
    if args.task not in p['tasks']:
        raise ValueError('Choose a registered task')
    torch.cuda.set_device(args.device)
    with (out/('%s_%s.lock' % (args.phase, args.task))).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a = features(out, p, args.task, args.phase)
        for seed in p['seeds']:
            for kind in KINDS:
                run_job(out, p, args.phase, args.task, seed, kind, a, args.device)
            full, values, _ = components(out, p, args.phase, args.task, seed, a)
            print('COMPONENTS', args.phase, args.task, seed,
                  {key:base.metrics(a['y_val'], val, a['pairs_val']) for key, val in values.items()}, flush=True)


def inner_rows(out, p):
    rows, hashes, chosen = [], {}, {}
    for task in p['tasks']:
        a = features(out, p, task, 'inner')
        group = []
        for seed in p['seeds']:
            full, values, state_hashes = components(out, p, 'inner', task, seed, a)
            hashes.update(state_hashes)
            for cfg in p['recipes']:
                score = kernels.combine(full, values, cfg)
                group.append(dict(task=task, seed=seed, recipe_id=cfg['id'],
                    **base.metrics(a['y_val'], score, a['pairs_val'])))
        chosen[task] = kernels.choose(group, p['recipes'], p['seeds'])
        rows.extend(group)
        dest = out/'features'/'inner'/task
        for name in ('manifest.json', 'AUDIT.json'):
            hashes[str((dest/name).relative_to(out))] = base.digest(dest/name)
    return rows, hashes, chosen


def select(out):
    import pandas as pd
    p = protocol(out)
    if (out/'SELECTION.json').exists():
        selection(out, p)
        return
    rows, hashes, chosen = inner_rows(out, p)
    pd.DataFrame(rows).to_csv(out/'inner_results.csv', index=False)
    pd.DataFrame(rows).groupby(['task', 'recipe_id'])[list(kernels.METRICS)].agg(['mean', 'std']).to_csv(out/'inner_summary.csv')
    hashes.update({name:base.digest(out/name) for name in ('inner_results.csv', 'inner_summary.csv')})
    base.write_json(out/'SELECTION.json', base.seal(dict(protocol_sha256=p['sha256'], chosen=chosen,
        artifacts_sha256=hashes, created_at_utc=datetime.now(timezone.utc).isoformat())))
    print('FROZEN_SELECTION', chosen, flush=True)


def verify_task(args):
    import torch
    from tabicl import TabICLClassifier
    out = Path(args.out_dir).resolve()
    p = protocol(out)
    selection(out, p)
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    replays = []
    for phase in ('inner', 'outer'):
        a = features(out, p, args.task, phase)
        for seed in p['seeds']:
            for kind in KINDS:
                dest = directory(out, phase, args.task, seed, kind)
                r = checked_result(dest, p)
                x, y, query = context_and_queries(p, args.task, a, kind)
                clf = TabICLClassifier.load(dest/'fitted.pkl', device=args.device)
                if clf.n_estimators != p['inference_members'] or clf.random_state != seed or clf.allow_auto_download:
                    raise ValueError('Saved estimator settings changed')
                if base.parameter_digest(clf) != p['parameter_sha256'] or clf.n_samples_in_ != len(y):
                    raise ValueError('Saved parameters or context row count changed')
                np.testing.assert_array_equal(clf.ensemble_generator_.y_, y)
                expected = clf.ensemble_generator_.unique_filter_.transform(clf.X_encoder_.transform(x))
                np.testing.assert_allclose(clf.ensemble_generator_.X_, expected, atol=1e-7, rtol=1e-7)
                flat = query.reshape(-1, x.shape[1])
                replay = base.query_scores(clf, flat).reshape(query.shape[:2])
                with np.load(dest/'scores.npz', allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive['y_val'], a['y_val'])
                    np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
                    saved = archive['views'].copy()
                np.testing.assert_allclose(replay, saved, atol=1e-6, rtol=1e-6)
                entry = dict(path=str(dest.relative_to(out)), result_sha256=base.digest(dest/'result.json'),
                             max_abs=float(np.max(np.abs(replay-saved))))
                if kind == 'full' and not p['smoke']:
                    original = baseline_score(p, phase, args.task, seed, a)
                    np.testing.assert_allclose(replay[0], original, atol=1e-6, rtol=1e-6)
                    entry['baseline_max_abs'] = float(np.max(np.abs(replay[0]-original)))
                if p['smoke']:
                    reverse = base.query_scores(clf, flat[::-1])[::-1]
                    split = np.concatenate([base.query_scores(clf, part) for part in np.array_split(flat, 3)])
                    # Match the original TabICLv2 smoke's FP32 query-invariance
                    # tolerance; same-shape saved-state replay stays at 1e-6.
                    np.testing.assert_allclose(reverse, replay.ravel(),
                        atol=p['query_invariance_atol'], rtol=p['query_invariance_rtol'])
                    np.testing.assert_allclose(split, replay.ravel(),
                        atol=p['query_invariance_atol'], rtol=p['query_invariance_rtol'])
                    entry.update(query_permutation_max_abs=float(np.max(np.abs(reverse-replay.ravel()))),
                                 query_splitting_max_abs=float(np.max(np.abs(split-replay.ravel()))))
                replays.append(entry)
                print('REPLAYED', phase, args.task, seed, kind, entry['max_abs'], flush=True)
                del clf
                gc.collect()
                torch.cuda.empty_cache()
    base.write_json(out/('VERIFY_%s.json' % args.task), dict(protocol_sha256=p['sha256'],
        task=args.task, passed=True, replays=replays))


def report(out):
    import pandas as pd
    p = protocol(out)
    s = selection(out, p)
    _, _, recomputed = inner_rows(out, p)
    if recomputed != s['chosen']:
        raise ValueError('Inner selection could not be reproduced')
    replays = []
    expected_paths = {str(directory(out, phase, task, seed, kind).relative_to(out))
        for phase in ('inner', 'outer') for task in p['tasks'] for seed in p['seeds'] for kind in KINDS}
    for task in p['tasks']:
        v = base.read_json(out/('VERIFY_%s.json' % task))
        if not v['passed'] or v['task'] != task or v['protocol_sha256'] != p['sha256']:
            raise ValueError('Missing per-task replay verification')
        for item in v['replays']:
            if base.digest(out/item['path']/'result.json') != item['result_sha256']:
                raise ValueError('Replayed result changed')
        replays.extend(v['replays'])
    if len(replays) != len(expected_paths) or {item['path'] for item in replays} != expected_paths:
        raise ValueError('Not every estimator was replayed exactly once')
    rows, differences, gates = [], [], {}
    for task in p['tasks']:
        a = features(out, p, task, 'outer')
        for seed in p['seeds']:
            full, values, _ = components(out, p, 'outer', task, seed, a)
            selected = kernels.combine(full, values, s['chosen'][task]['recipe'])
            for method, score in dict(frozen=full, selected=selected, **values).items():
                rows.append(dict(task=task, seed=seed, method=method,
                    **base.metrics(a['y_val'], score, a['pairs_val'])))
    frame = pd.DataFrame(rows)
    for method in ('selected', *kernels.ARMS):
        gates[method] = {}
        for task in p['tasks']:
            old = frame[(frame.task == task) & (frame.method == 'frozen')].set_index('seed')
            new = frame[(frame.task == task) & (frame.method == method)].set_index('seed')
            delta = new[list(base.METRICS)]-old[list(base.METRICS)]
            means = {key:float(delta[key].mean()) for key in base.METRICS}
            positive = {key:int((delta[key] > 0).sum()) for key in ('auc', 'ap')}
            gates[method][task] = dict(mean_deltas=means, positive_seed_labels=positive,
                passed=all(means[key] > 0 and positive[key] >= (2 if len(p['seeds']) == 3 else 1) for key in ('auc', 'ap')))
            for seed, row in delta.iterrows():
                differences.append(dict(task=task, method=method, seed=int(seed), **row.to_dict()))
    joint = {method:all(item['passed'] for item in panel.values()) for method, panel in gates.items()}
    frame.to_csv(out/'outer_results.csv', index=False)
    pd.DataFrame(differences).to_csv(out/'paired_deltas.csv', index=False)
    summary = frame.groupby(['task', 'method'])[list(base.METRICS)].agg(['mean', 'std'])
    summary.to_csv(out/'summary.csv')
    labels = dict(frozen='原 TabICLv2', selected='内部选定机制及权重', query_bag='完整上下文 + 遮边查询集成',
                  fold_full='对应折上下文 + 完整图查询（对照）', fold_match='对应折上下文 + 同图查询')
    lines = ['# TabICLv2：支持图与上下文匹配', '',
             '**工程短流程，不作性能结论。**' if p['smoke'] else '以下为同一历史外层验证、seed 666/0/1 的均值。两任务机制及概率权重仅由内部数据确定。', '',
             '| 任务 | 方案 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for method in labels:
            r = summary.loc[(task, method)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f |' % (task, labels[method],
                r[('auc','mean')], r[('ap','mean')], r[('macro_drug_ap','mean')]))
    lines += ['', '## 内部冻结选择', '', '| 任务 | 配置 | 内部 AUC | 内部 AP |', '|---|---|---:|---:|']
    for task in p['tasks']:
        c = s['chosen'][task]
        lines.append('| %s | %s | %.6f | %.6f |' % (task, c['recipe']['id'], c['auc'], c['ap']))
    lines += ['', '## 相对原模型的选定方案差值', '',
              '| 任务 | ΔAUC | 正向种子标签 | ΔAP | 正向种子标签 |', '|---|---:|---:|---:|---:|']
    for task in p['tasks']:
        g = gates['selected'][task]
        lines.append('| %s | %+.6f | %d/%d | %+.6f | %d/%d |' % (task, g['mean_deltas']['auc'],
            g['positive_seed_labels']['auc'], len(p['seeds']), g['mean_deltas']['ap'], g['positive_seed_labels']['ap'], len(p['seeds'])))
    lines += ['', '## 核验与边界', '',
        '- 选定方案的双任务 AUC/AP 预设判定：`%s`。' % joint['selected'],
        '- %d 个拟合上下文状态全部重载；所有状态共享同一官方检查点，没有梯度更新。' % len(replays),
        '- 四份特征缓存各重建五个支持图，并核对原有逐折上下文及完整查询特征；每个查询的目标正边及反向边均不可见。',
        '- 五个对应折上下文模型对完整图查询和同图查询分别评分，形成相同上下文数及相同模型数的对照。原分位数输入变换仍只由拟合行得到。',
        '- 完整上下文的完整图查询分数与原基线逐行校验；每个新拟合状态核对参数哈希、上下文、标签、查询配对及所有预测视图。',
        '- 控制项及三种独立机制均预先声明；表中内部选定方案没有按外层指标改选。',
        '- 图支持采样与 TabICL 推理成员是不同层级；五个视图均使用同一个种子的预处理置换，仍有额外计算成本。',
        '- 本轮是支持一致性的本地假设验证，不是 TFMLinker 的可学习跨图原型或预训练 GNN 复现。',
        '- 历史验证被反复用于开发；三个种子不构成独立划分或独立预训练重复。没有加载测试数组。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    base.write_json(out/'OUTCOME.json', dict(protocol_sha256=p['sha256'], smoke=p['smoke'],
        gates=gates, joint=joint, scope_complete=True))
    base.write_json(out/'VALIDATION.json', dict(protocol_sha256=p['sha256'], smoke=p['smoke'], passed=True,
        all_estimators_replayed=True, estimator_states=len(replays), selection_replayed=True,
        baseline_predictions_aligned=True, no_test_evaluation=True, replays=replays))
    print(summary.to_string(), flush=True)
    print('VERIFIED', len(replays), 'context states; selected joint outcome:', joint['selected'], flush=True)


def workers(commands, label):
    processes = [subprocess.Popen(command) for command in commands]
    while any(process.poll() is None for process in processes):
        time.sleep(1)
    if any(process.returncode for process in processes):
        raise RuntimeError('Failed workers in '+label)
    print('STAGE_DONE', label, flush=True)


def pipeline(out, devices):
    p = protocol(out)
    for phase in ('inner', 'outer'):
        if phase == 'outer':
            select(out)
        workers([[p['graph_python'], '-u', '-m', 'improve.tabicl_graph_context_features',
                  '--out-dir', str(out), '--phase', phase, '--task', task] for task in p['tasks']], 'features_'+phase)
        workers([[sys.executable, '-u', '-m', 'improve.tabicl_graph_context', 'run', '--out-dir', str(out),
                  '--phase', phase, '--task', task, '--device', devices[i % len(devices)]]
                 for i, task in enumerate(p['tasks'])], phase)
    workers([[sys.executable, '-u', '-m', 'improve.tabicl_graph_context', 'verify', '--out-dir', str(out),
              '--task', task, '--device', devices[i % len(devices)]] for i, task in enumerate(p['tasks'])], 'verify')
    report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'select', 'verify', 'report', 'pipeline'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--parent-dir', default=DEFAULT_PARENT)
    parser.add_argument('--graph-python', default=GRAPH_PYTHON)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--phase', choices=('inner', 'outer'), default='inner')
    parser.add_argument('--task', choices=('LncDrug', 'MiDrug'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'run':
        run(args)
    elif args.command == 'select':
        select(out)
    elif args.command == 'verify':
        verify_task(args)
    elif args.command == 'report':
        report(out)
    else:
        pipeline(out, args.devices.split(','))


if __name__ == '__main__':
    main()
