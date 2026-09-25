"""MiDrug-specific drug memory around the verified graph-context TabICLv2.

prepare -> cache inner -> run inner seeds -> select -> cache/run outer ->
evaluate -> replay -> report. Historical protocols and artifacts are read-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from improve import tabicl_graph_context as reference
from improve import tabicl_graph_context_modules as reference_kernels
from improve import tabicl_midrug_memory_modules as kernels


DEFAULT_OUT = 'improve/results/tabicl_midrug_memory_20260912'
DEFAULT_REFERENCE = 'improve/results/tabicl_graph_context_20260912'
OWN = ('tabicl_midrug_memory.py', 'tabicl_midrug_memory_modules.py',
       'test_tabicl_midrug_memory.py')
COMPONENTS = (*kernels.COMPONENTS, 'global_side')


def sources():
    return {name:base.digest(Path(__file__).with_name(name)) for name in OWN}


def prepare(args):
    out, ref = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    rp = reference.protocol(ref)
    rs = reference.selection(ref, rp)
    validation = base.read_json(ref/'VALIDATION.json')
    if rp['smoke'] or not validation['passed'] or not validation['all_estimators_replayed']:
        raise ValueError('Need the verified formal graph-context reference')
    if validation['protocol_sha256'] != rp['sha256']:
        raise ValueError('Reference validation belongs to another protocol')
    seeds = [666] if args.smoke else list(rp['seeds'])
    files = [ref/name for name in ('protocol.json', 'SELECTION.json', 'VALIDATION.json',
                                   'SELECTED_PREDICTIONS.json')]
    for phase in ('inner', 'outer'):
        for task in rp['tasks']:
            for seed in seeds:
                for kind in reference.KINDS:
                    dest = reference.directory(ref, phase, task, seed, kind)
                    files += [dest/name for name in ('result.json', 'scores.npz')]
                dest = Path(rp['parent_dir'])/rp['baseline'][phase][task][str(seed)]
                files += [dest/name for name in ('result.json', 'scores.npz')]
            for root in (ref, Path(rp['feature_reference_dir'])):
                dest = root/'features'/phase/task
                manifest = base.read_json(dest/'manifest.json')
                files += [dest/'manifest.json', *[dest/name for name in manifest['files']]]
        files.append(Path(rp['graph_reference_dir'])/'inputs'/'MiDrug'/(phase+'.npz'))
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        reference_dir=str(ref), reference_protocol_sha256=rp['sha256'],
        reference_files={str(path):base.digest(path) for path in sorted(set(files))},
        baseline_recipes={t:rs['chosen'][t]['recipe'] for t in rp['tasks']},
        source_sha256=sources(), seeds=seeds, tasks=list(rp['tasks']), recipes=kernels.recipes(),
        components=list(COMPONENTS), clusters=3 if args.smoke else 8,
        context_size=64 if args.smoke else 2048, reservoir_size=16 if args.smoke else 256,
        side_dim=8, route_topk=2, route_temperature=.1, routing_seed=20260912,
        inference_members=2 if args.smoke else 32, parameter_sha256=rp['parameter_sha256'],
        no_test_evaluation=True, replay_atol=1e-6, replay_rtol=1e-6,
        query_invariance_atol=1e-5, query_invariance_rtol=1e-5,
        architecture='Protected global graph-context TabICLv2 plus drug-routed local contexts. Eight drug-attribute prototypes retrieve 1792 local fitting rows and a shared 256-row label-free reservoir. Each query uses two experts, evaluated on five query support graphs. A bounded logit residual optionally uses router similarity and graph-view agreement.',
        protection='LncDrug always returns its previously selected graph-context scores bit for bit. Its gate is fixed to zero before any new selection. No performance-based switch or shared parameter update is permitted.',
        inputs='Reuse 36 verified graph features. Drug-side experts add the first eight existing RNA PCA coordinates and eight drug PCA coordinates fitted from the same available entity attributes. PCA and routing use no target labels. No additional biological dataset.',
        context_policy='Only fitting rows enter an expert. Preserve the original out-of-fold graph features and reconstruct no target labels. Queries use the five already verified masked support views. The local contexts mix the original graph folds, as in the reference full-context query-bag branch.',
        selection='For MiDrug only, test two local input views, uniform/consensus bounded residuals and alpha 0.05/0.10/0.20/0.35, plus identity. A candidate needs positive seed-mean inner AUC and AP and at least two positive seed differences for each. Among eligible candidates maximize mean (AUC+AP)/2; ties prefer lower alpha then ID. Freeze before any own outer NPZ load.',
        controls='Global context with the same side inputs is an unselected feature-only control. Report same-alpha global-side, no-side and no-consensus controls on outer data, without selecting them from their outer results. All three components are computed on all registered seeds and phases.',
        gate='MiDrug must improve mean AUC and AP over the current graph-context reference, with at least two positive seed differences per metric. All LncDrug scores must be exactly equal. Macro drug AP is also reported.',
        limitations=['Both historical partitions have repeatedly participated in development; this is not independent test evidence.',
            'The official pretrained checkpoint is unchanged; this is a new task-level memory architecture, not a newly pretrained foundation model or a global novelty claim.',
            'Three inference seeds share one checkpoint and one data split.',
            'Drug and RNA attributes retain the reference transductive feature policy.',
            'The LncDrug guarantee comes from a fixed identity path. It is not evidence that the new branch improves both tasks.',
            'Additional expert contexts and query graph views increase inference cost.']))
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name), out/'source_snapshot'/name)
    base.write_json(out/'protocol.json', p)
    print('FROZEN', out, flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out)/'protocol.json'))
    if sources() != p['source_sha256']:
        raise ValueError('Frozen memory implementation changed; use a new experiment')
    for path, expected in p['reference_files'].items():
        if base.digest(path) != expected:
            raise ValueError('Reference artifact changed: '+path)
    rp = reference.protocol(p['reference_dir'])
    if rp['sha256'] != p['reference_protocol_sha256']:
        raise ValueError('Reference protocol changed')
    return p


def selection(out, p):
    s = base.check_seal(base.read_json(Path(out)/'SELECTION.json'))
    if s['protocol_sha256'] != p['sha256'] or s['chosen']['MiDrug']['recipe'] not in p['recipes']:
        raise ValueError('Invalid memory selection')
    if s['chosen']['LncDrug'] != dict(path='identity', reference_recipe=p['baseline_recipes']['LncDrug']):
        raise ValueError('The protected relation was modified')
    for name, expected in s['artifacts_sha256'].items():
        if base.digest(Path(out)/name) != expected:
            raise ValueError('Frozen selection artifact changed')
    return s


def data(out, p, phase, task='MiDrug'):
    if phase not in ('inner', 'outer') or task not in p['tasks']:
        raise ValueError('Invalid task or phase')
    if phase == 'outer':
        selection(out, p)  # Precedes every outer archive or reference feature load.
    rp = base.read_json(Path(p['reference_dir'])/'protocol.json')
    full = reference.features(p['reference_dir'], rp, task, phase)
    a = {key:value for key, value in full.items()}
    a['_reference_a'] = full
    a['_query_indices'] = np.arange(len(a['y_val']))
    if task == 'MiDrug':
        graph_file = Path(rp['graph_reference_dir'])/'inputs'/task/(phase+'.npz')
        with np.load(graph_file, allow_pickle=False) as archive:
            if any('test' in k.lower() for k in archive.files):
                raise ValueError('Test arrays are forbidden')
            a['drug_features'] = archive['x_drug'].copy()
            a['rna_features'] = archive['pca_x_miRNA'].copy()
            for key in ('pairs_train', 'y_train', 'pairs_val', 'y_val'):
                np.testing.assert_array_equal(archive[key], a[key])
    if p['smoke']:
        for part, budget in (('train', 256), ('val', 40)):
            keep = base.balanced_subset(a['y_'+part], budget, 20260912)
            for prefix in ('raw', 'x', 'y'):
                a[prefix+'_'+part] = a[prefix+'_'+part][keep]
            a['pairs_'+part] = a['pairs_'+part][:, keep]
            if part == 'val':
                a['_query_indices'] = keep
                a['raw_query_views'] = a['raw_query_views'][:, keep]
                a['x_query_views'] = a['x_query_views'][:, keep]
            else:
                a['fold_ids'] = a['fold_ids'][keep]
    base.validate_arrays(a)
    return a


def baseline_score(p, phase, task, seed, a):
    rp = base.read_json(Path(p['reference_dir'])/'protocol.json')
    original, components, _ = reference.components(Path(p['reference_dir']), rp,
        phase, task, seed, a['_reference_a'])
    score = reference_kernels.combine(original, components, p['baseline_recipes'][task])
    return score[a['_query_indices']]


def cache(args):
    out = Path(args.out_dir).resolve()
    p = protocol(out)
    a = data(out, p, args.phase)
    dest = out/'memory'/args.phase
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest/'manifest.json').exists():
            memory(out, p, args.phase, a)
            return
        state = kernels.fit_memory(a['drug_features'], a['rna_features'], a['pairs_train'],
            clusters=p['clusters'], context_size=p['context_size'], reservoir_size=p['reservoir_size'],
            side_dim=p['side_dim'], seed=p['routing_seed'])
        for context in state['contexts']:
            if set(np.unique(a['y_train'][context])) != {0, 1}:
                raise ValueError('A label-free retrieved context lacks a class')
        weights, reliability = kernels.route(state, a['pairs_val'], topk=p['route_topk'],
                                             temperature=p['route_temperature'])
        np.savez_compressed(dest/'state.npz', **state)
        np.savez_compressed(dest/'routing.npz', weights=weights, reliability=reliability,
                            pairs_val=a['pairs_val'])
        base.write_json(dest/'manifest.json', dict(protocol_sha256=p['sha256'], phase=args.phase,
            query_labels_used=False, fitting_labels_used_for_routing=False,
            context_rows=[len(c) for c in state['contexts']],
            routed_query_rows=[int((weights[:, g] > 0).sum()) for g in range(p['clusters'])],
            features_with_side=36+2*p['side_dim'],
            files={name:base.digest(dest/name) for name in ('state.npz', 'routing.npz')}))
    print('MEMORY', args.phase, 'contexts', state['contexts'].shape, flush=True)


def memory(out, p, phase, a):
    if phase == 'outer':
        selection(out, p)
    dest = Path(out)/'memory'/phase
    m = base.read_json(dest/'manifest.json')
    if (m['protocol_sha256'], m['phase']) != (p['sha256'], phase):
        raise ValueError('Wrong memory identity')
    for name, expected in m['files'].items():
        if base.digest(dest/name) != expected:
            raise ValueError('Memory changed')
    with np.load(dest/'state.npz', allow_pickle=False) as archive:
        state = {key:archive[key].copy() for key in archive.files}
    weights, reliability = kernels.route(state, a['pairs_val'], topk=p['route_topk'],
                                         temperature=p['route_temperature'])
    with np.load(dest/'routing.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
        np.testing.assert_array_equal(archive['weights'], weights)
        np.testing.assert_array_equal(archive['reliability'], reliability)
    return state, weights, reliability


def kinds(p):
    return [(component, group) for component in COMPONENTS
            for group in ([-1] if component == 'global_side' else range(p['clusters']))]


def directory(out, phase, seed, component, group):
    return Path(out)/phase/('s%d' % seed)/component/('full' if group < 0 else 'g%d' % group)


def context_and_queries(a, state, weights, component, group):
    x, pairs = a['raw_train'], a['pairs_train']
    views = a['raw_query_views']
    context = np.arange(len(x)) if component == 'global_side' else state['contexts'][group]
    query = np.arange(views.shape[1]) if component == 'global_side' else np.flatnonzero(weights[:, group] > 0)
    x = x[context]
    values = views[:, query]
    if component in ('drug_side', 'global_side'):
        x = kernels.append_side(x, pairs[:, context], state)
        values = np.stack([kernels.append_side(v, a['pairs_val'][:, query], state) for v in values])
    if component not in COMPONENTS or not len(query):
        raise ValueError('Unknown or unused context')
    return x, a['y_train'][context], values, query, context


def checked_result(dest, p):
    r = base.read_json(Path(dest)/'result.json')
    if r['protocol_sha256'] != p['sha256']:
        raise ValueError('Wrong expert result protocol')
    for name, expected in r['artifacts_sha256'].items():
        if base.digest(Path(dest)/name) != expected:
            raise ValueError('Expert artifact changed')
    return r


def run(args):
    import torch
    out = Path(args.out_dir).resolve()
    p = protocol(out)
    if args.seed not in p['seeds']:
        raise ValueError('Unregistered inference seed')
    a = data(out, p, args.phase)
    state, weights, _ = memory(out, p, args.phase, a)
    rp = base.read_json(Path(p['reference_dir'])/'protocol.json')
    tp = base.read_json(Path(rp['parent_dir'])/'protocol.json')
    torch.cuda.set_device(args.device)
    torch.empty(0, device=args.device)
    with (out/('%s_s%d.lock' % (args.phase, args.seed))).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for component, group in kinds(p):
            dest = directory(out, args.phase, args.seed, component, group)
            if (dest/'result.json').exists():
                checked_result(dest, p)
                continue
            x, y, queries, query, context = context_and_queries(a, state, weights, component, group)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(args.device)
            clf = base.fit_context(base.estimator(dict(n_estimators=p['inference_members']),
                args.seed, tp, args.device), x, y)
            if base.parameter_digest(clf) != p['parameter_sha256']:
                raise ValueError('Pretrained parameters changed')
            scores = base.query_scores(clf, queries.reshape(-1, x.shape[1])).reshape(queries.shape[:2])
            dest.mkdir(parents=True, exist_ok=True)
            clf.save(dest/'fitted.pkl', save_model_weights=False, save_training_data=True, save_kv_cache=False)
            np.savez_compressed(dest/'scores.npz', views=scores, query_indices=query, context_indices=context,
                                y_val=a['y_val'][query], pairs_val=a['pairs_val'][:, query])
            r = dict(protocol_sha256=p['sha256'], phase=args.phase, seed=args.seed,
                component=component, group=group, input_features=x.shape[1], context_rows=len(x),
                query_rows=len(query), parameter_sha256=base.parameter_digest(clf),
                memory_manifest_sha256=base.digest(out/'memory'/args.phase/'manifest.json'),
                elapsed_seconds=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device),
                artifacts_sha256={name:base.digest(dest/name) for name in ('scores.npz', 'fitted.pkl')})
            base.write_json(dest/'result.json', r)
            print('SAVED', args.phase, args.seed, component, group, 'seconds', round(r['elapsed_seconds'], 1), flush=True)
            del clf
            gc.collect()
            torch.cuda.empty_cache()


def components(out, p, phase, seed, a):
    state, weights, reliability = memory(out, p, phase, a)
    values = {name:np.zeros((5, len(a['y_val'])), dtype=np.float64) for name in COMPONENTS}
    hashes = {}
    for component, group in kinds(p):
        dest = directory(out, phase, seed, component, group)
        r = checked_result(dest, p)
        if (r['phase'], r['seed'], r['component'], r['group']) != (phase, seed, component, group):
            raise ValueError('Wrong expert identity')
        if r['memory_manifest_sha256'] != base.digest(Path(out)/'memory'/phase/'manifest.json'):
            raise ValueError('Wrong expert memory')
        _, _, _, query, context = context_and_queries(a, state, weights, component, group)
        with np.load(dest/'scores.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['query_indices'], query)
            np.testing.assert_array_equal(archive['context_indices'], context)
            np.testing.assert_array_equal(archive['y_val'], a['y_val'][query])
            np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'][:, query])
            scores = archive['views'].copy()
        if scores.shape != (5, len(query)) or not np.isfinite(scores).all():
            raise ValueError('Invalid saved expert probabilities')
        factor = 1. if group < 0 else weights[query, group][None]
        values[component][:, query] += factor*scores
        hashes[str((dest/'result.json').relative_to(out))] = base.digest(dest/'result.json')
    for score in values.values():
        kernels.probabilities(score.ravel())
    return values, reliability, hashes


def inner_selection(out, p):
    a = data(out, p, 'inner')
    rows, hashes = [], {}
    for seed in p['seeds']:
        values, reliability, files = components(out, p, 'inner', seed, a)
        hashes.update(files)
        baseline = baseline_score(p, 'inner', 'MiDrug', seed, a)
        for cfg in p['recipes']:
            pred = kernels.protected_residual('MiDrug', baseline, values.get(cfg['component']), reliability, cfg)
            rows.append(dict(seed=seed, recipe_id=cfg['id'], **base.metrics(a['y_val'], pred, a['pairs_val'])))
    chosen, summaries = kernels.choose(rows, p['recipes'], p['seeds'])
    for name in ('manifest.json', 'state.npz', 'routing.npz'):
        relative = 'memory/inner/'+name
        hashes[relative] = base.digest(Path(out)/relative)
    return rows, chosen, summaries, hashes


def select(out):
    import pandas as pd
    out = Path(out).resolve()
    p = protocol(out)
    if (out/'SELECTION.json').exists():
        selection(out, p)
        return
    rows, chosen, summaries, hashes = inner_selection(out, p)
    pd.DataFrame(rows).to_csv(out/'inner_results.csv', index=False)
    base.write_json(out/'inner_summary.json', summaries)
    hashes.update({name:base.digest(out/name) for name in ('inner_results.csv', 'inner_summary.json')})
    base.write_json(out/'SELECTION.json', base.seal(dict(protocol_sha256=p['sha256'],
        chosen=dict(MiDrug=chosen, LncDrug=dict(path='identity', reference_recipe=p['baseline_recipes']['LncDrug'])),
        artifacts_sha256=hashes, created_at_utc=datetime.now(timezone.utc).isoformat())))
    print('FROZEN_SELECTION', chosen, flush=True)


def evaluate(out):
    import pandas as pd
    out = Path(out).resolve()
    p = protocol(out)
    s = selection(out, p)
    cfg = s['chosen']['MiDrug']['recipe']
    rows, differences, exports = [], [], []
    protected_exact = True
    for task in p['tasks']:
        a = data(out, p, 'outer', task)
        for seed in p['seeds']:
            baseline = baseline_score(p, 'outer', task, seed, a)
            if task == 'MiDrug':
                values, reliability, _ = components(out, p, 'outer', seed, a)
                pred = kernels.protected_residual(task, baseline, values.get(cfg['component']), reliability, cfg)
                methods = dict(reference=baseline, selected=pred)
                # These are frozen-checkpoint diagnostic substitutions, not
                # additional outer-selected recipes.
                methods['global_side_control'] = kernels.protected_residual(task, baseline,
                    values['global_side'], np.ones_like(reliability), cfg)
                if cfg['id'] != 'frozen':
                    no_side = next(c for c in p['recipes'] if c['component'] == 'drug_graph'
                        and c['gate'] == cfg['gate'] and c['alpha'] == cfg['alpha'])
                    uniform = next(c for c in p['recipes'] if c['component'] == cfg['component']
                        and c['gate'] == 'uniform' and c['alpha'] == cfg['alpha'])
                    methods['no_side_control'] = kernels.protected_residual(task, baseline,
                        values['drug_graph'], reliability, no_side)
                    methods['uniform_control'] = kernels.protected_residual(task, baseline,
                        values[cfg['component']], reliability, uniform)
                for component in COMPONENTS:
                    methods[component+'_expert'] = values[component].mean(0)
            else:
                pred = kernels.protected_residual(task, baseline, None, None, cfg)
                protected_exact &= np.array_equal(pred, baseline)
                methods = dict(reference=baseline, selected=pred)
            for method, score in methods.items():
                rows.append(dict(task=task, seed=seed, method=method,
                    **base.metrics(a['y_val'], score, a['pairs_val'])))
            dest = out/'selected'/task/('s%d' % seed)
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/'scores.npz', val=pred, reference=baseline,
                                y_val=a['y_val'], pairs_val=a['pairs_val'])
            exports.append(dict(task=task, seed=seed, path=str((dest/'scores.npz').relative_to(out)),
                                sha256=base.digest(dest/'scores.npz')))
    frame = pd.DataFrame(rows)
    gates = {}
    for task in p['tasks']:
        old = frame[(frame.task == task) & (frame.method == 'reference')].set_index('seed')
        new = frame[(frame.task == task) & (frame.method == 'selected')].set_index('seed')
        delta = new[list(base.METRICS)]-old[list(base.METRICS)]
        means = {key:float(delta[key].mean()) for key in base.METRICS}
        positives = {key:int((delta[key] > 0).sum()) for key in ('auc', 'ap')}
        passed = (protected_exact and all(v == 0. for v in means.values())) if task == 'LncDrug' else all(
            means[key] > 0 and positives[key] >= min(2, len(p['seeds'])) for key in ('auc', 'ap'))
        gates[task] = dict(mean_deltas=means, positive_seeds=positives, passed=passed)
        for seed, row in delta.iterrows():
            differences.append(dict(task=task, seed=int(seed), **row.to_dict()))
    frame.to_csv(out/'outer_results.csv', index=False)
    pd.DataFrame(differences).to_csv(out/'paired_deltas.csv', index=False)
    frame.groupby(['task', 'method'])[list(base.METRICS)].agg(['mean', 'std']).to_csv(out/'summary.csv')
    base.write_json(out/'OUTCOME.json', dict(protocol_sha256=p['sha256'], smoke=p['smoke'],
        gates=gates, lncdrug_predictions_bitwise_equal=bool(protected_exact),
        passed=all(v['passed'] for v in gates.values()), no_test_evaluation=True))
    base.write_json(out/'SELECTED_PREDICTIONS.json', dict(protocol_sha256=p['sha256'],
        selection_sha256=base.digest(out/'SELECTION.json'), exported=exports))
    print('OUTCOME', gates, 'lncdrug_exact', protected_exact, flush=True)


def verify(args):
    import torch
    from tabicl import TabICLClassifier
    out = Path(args.out_dir).resolve()
    p = protocol(out)
    selection(out, p)
    if args.seed not in p['seeds']:
        raise ValueError('Unregistered replay seed')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checks = []
    for phase in ('inner', 'outer'):
        a = data(out, p, phase)
        state, weights, _ = memory(out, p, phase, a)
        for component, group in kinds(p):
            dest = directory(out, phase, args.seed, component, group)
            r = checked_result(dest, p)
            x, y, queries, _, _ = context_and_queries(a, state, weights, component, group)
            clf = TabICLClassifier.load(dest/'fitted.pkl', device=args.device)
            if (clf.n_estimators, clf.random_state, clf.n_samples_in_, clf.allow_auto_download) != (
                    p['inference_members'], args.seed, len(y), False):
                raise ValueError('Saved expert settings changed')
            if base.parameter_digest(clf) != p['parameter_sha256']:
                raise ValueError('Saved expert pretrained weights changed')
            np.testing.assert_array_equal(clf.ensemble_generator_.y_, y)
            encoded = clf.ensemble_generator_.unique_filter_.transform(clf.X_encoder_.transform(x))
            np.testing.assert_allclose(clf.ensemble_generator_.X_, encoded, atol=1e-7, rtol=1e-7)
            flat = queries.reshape(-1, x.shape[1])
            replay = base.query_scores(clf, flat).reshape(queries.shape[:2])
            with np.load(dest/'scores.npz', allow_pickle=False) as archive:
                saved = archive['views'].copy()
            np.testing.assert_allclose(replay, saved, atol=p['replay_atol'], rtol=p['replay_rtol'])
            check = dict(path=str(dest.relative_to(out)), result_sha256=base.digest(dest/'result.json'),
                         max_abs=float(np.max(np.abs(replay-saved))))
            if p['smoke']:
                reverse = base.query_scores(clf, flat[::-1])[::-1]
                split = np.concatenate([base.query_scores(clf, v) for v in np.array_split(flat, 3)])
                for value in (reverse, split):
                    np.testing.assert_allclose(value, replay.ravel(),
                        atol=p['query_invariance_atol'], rtol=p['query_invariance_rtol'])
                check.update(query_permutation_max_abs=float(np.max(np.abs(reverse-replay.ravel()))),
                             query_split_max_abs=float(np.max(np.abs(split-replay.ravel()))))
            checks.append(check)
            print('REPLAYED', phase, args.seed, component, group, check['max_abs'], flush=True)
            del clf
            gc.collect()
            torch.cuda.empty_cache()
    base.write_json(out/('VERIFY_s%d.json' % args.seed), dict(protocol_sha256=p['sha256'],
        seed=args.seed, passed=True, replays=checks))


def report(out):
    import pandas as pd
    out = Path(out).resolve()
    p = protocol(out)
    s = selection(out, p)
    _, chosen, _, _ = inner_selection(out, p)
    if chosen != s['chosen']['MiDrug']:
        raise ValueError('Inner selection did not replay')
    replays = []
    for seed in p['seeds']:
        v = base.read_json(out/('VERIFY_s%d.json' % seed))
        if v['protocol_sha256'] != p['sha256'] or v['seed'] != seed or not v['passed']:
            raise ValueError('Missing seed replay')
        for item in v['replays']:
            if item['result_sha256'] != base.digest(out/item['path']/'result.json'):
                raise ValueError('Replayed expert changed')
        replays.extend(v['replays'])
    expected = {str(directory(out, phase, seed, component, group).relative_to(out))
        for phase in ('inner', 'outer') for seed in p['seeds'] for component, group in kinds(p)}
    if len(replays) != len(expected) or {v['path'] for v in replays} != expected:
        raise ValueError('Not all experts were replayed exactly once')
    exports = base.read_json(out/'SELECTED_PREDICTIONS.json')
    if exports['protocol_sha256'] != p['sha256'] or exports['selection_sha256'] != base.digest(out/'SELECTION.json'):
        raise ValueError('Wrong prediction export provenance')
    for item in exports['exported']:
        if base.digest(out/item['path']) != item['sha256']:
            raise ValueError('Exported predictions changed')
    # Recompute every aggregate, metric and protected output from saved experts.
    old = {name:base.digest(out/name) for name in ('outer_results.csv', 'paired_deltas.csv', 'summary.csv', 'OUTCOME.json')}
    evaluate(out)
    if old != {name:base.digest(out/name) for name in old}:
        raise ValueError('Outer evaluation did not replay')
    outcome = base.read_json(out/'OUTCOME.json')
    base.write_json(out/'VALIDATION.json', dict(protocol_sha256=p['sha256'], passed=True,
        all_estimators_replayed=True, estimator_states=len(replays), replays=replays,
        selection_replayed=True, final_predictions_recomputed=True,
        lncdrug_predictions_bitwise_equal=outcome['lncdrug_predictions_bitwise_equal'],
        no_test_evaluation=True))
    frame = pd.read_csv(out/'outer_results.csv')
    summary = frame.groupby(['task', 'method'])[list(base.METRICS)].mean()
    lines = ['# MiDrug 药物记忆 TabICL', '',
        '**工程短流程，不作性能结论。**' if p['smoke'] else '同一历史外层验证、seed 666/0/1 均值；本轮选择只用内部验证。', '',
        'MiDrug 内部选定：`%s`。LncDrug 固定保留原图上下文路径。' % chosen['recipe']['id'], '',
        '| 任务 | 方案 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for method in ('reference', 'selected'):
            row = summary.loc[(task, method)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f |' %
                (task, method, row.auc, row.ap, row.macro_drug_ap))
    lines += ['', '通过预设目标：**%s**。LncDrug 预测逐项完全相同：**%s**。' %
              (outcome['passed'], outcome['lncdrug_predictions_bitwise_equal']), '',
              '全部 %d 个局部/全局专家状态已重载重放，最大误差 %.3g；内部选择和最终预测已重新计算。' %
              (len(replays), max(v['max_abs'] for v in replays)), '',
              '结构对照（使用相同的内部选定修正强度，不能由此重新选择外层模型）：', '',
              '| MiDrug 方案 | AUC | AP | 按药物宏 AP |', '|---|---:|---:|---:|']
    for (task, method), row in summary.iterrows():
        if task == 'MiDrug' and method not in ('reference', 'selected'):
            lines.append('| %s | %.6f | %.6f | %.6f |' % (method, row.auc, row.ap, row.macro_drug_ap))
    lines += ['', '注意：专家单独成绩和固定参数消融用于分析，未用外层结果改选。', '',
        '当前结果属于重复使用的历史验证上的开发证据，没有打开测试数组。三个推理种子共享同一官方权重和数据划分，不能当作独立泛化重复。LncDrug 的不下降来自恒等路径，不说明新分支对它有效。', '',
        '见 `protocol.json`、`SELECTION.json`、`paired_deltas.csv`、`VALIDATION.json` 和 `selected/`。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print('VERIFIED', len(replays), 'experts; goal passed', outcome['passed'], flush=True)


class MidrugMemoryPredictor:
    """Load a fitted specialist around externally supplied reference scores.

    The reference graph-context predictor remains responsible for the stable
    scores and the five graph query views. Entity IDs must use the saved graph
    indexing. LncDrug needs only its stable scores; specialists are not loaded.
    """
    def __init__(self, out, seed=666, device='cuda:0'):
        self.out = Path(out).resolve()
        self.p = protocol(self.out)
        self.selection = selection(self.out, self.p)
        if seed not in self.p['seeds']:
            raise ValueError('Unknown saved seed')
        self.seed, self.device = seed, device

    def predict(self, task, baseline, *, graph_views=None, pairs=None):
        cfg = self.selection['chosen']['MiDrug']['recipe']
        if task == 'LncDrug' or cfg['id'] == 'frozen':
            return kernels.protected_residual(task, baseline, None, None, cfg)
        import torch
        from tabicl import TabICLClassifier
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        baseline = kernels.probabilities(baseline)
        graph_views = np.asarray(graph_views)
        if graph_views.shape != (5, len(baseline), 36) or not np.isfinite(graph_views).all():
            raise ValueError('Expected five aligned 36-column graph query views')
        dest = self.out/'memory'/'outer'
        manifest = base.read_json(dest/'manifest.json')
        for name, expected in manifest['files'].items():
            if base.digest(dest/name) != expected:
                raise ValueError('Saved inference memory changed')
        with np.load(dest/'state.npz', allow_pickle=False) as archive:
            state = {key:archive[key].copy() for key in archive.files}
        weights, reliability = kernels.route(state, pairs, topk=self.p['route_topk'],
            temperature=self.p['route_temperature'])
        if len(weights) != len(baseline):
            raise ValueError('Pairs and probabilities are not aligned')
        values = np.zeros((5, len(baseline)))
        for group in range(self.p['clusters']):
            query = np.flatnonzero(weights[:, group] > 0)
            if not len(query):
                continue
            views = graph_views[:, query]
            if cfg['component'] == 'drug_side':
                views = np.stack([kernels.append_side(v, np.asarray(pairs)[:, query], state) for v in views])
            path = directory(self.out, 'outer', self.seed, cfg['component'], group)
            checked_result(path, self.p)
            clf = TabICLClassifier.load(path/'fitted.pkl', device=self.device)
            score = base.query_scores(clf, views.reshape(-1, views.shape[-1])).reshape(5, len(query))
            values[:, query] += weights[query, group][None]*score
            del clf
            gc.collect()
            if str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()
        return kernels.protected_residual(task, baseline, values, reliability, cfg)


def parallel_seeds(out, p, command, devices, phase=None):
    def worker(device, seeds):
        for seed in seeds:
            label = '%s_%s_s%d' % (command, phase or 'both', seed)
            argv = [sys.executable, '-m', 'improve.tabicl_midrug_memory', command,
                    '--out-dir', str(out), '--seed', str(seed), '--device', device]
            if phase is not None:
                argv += ['--phase', phase]
            print('START', label, device, flush=True)
            with (Path(out)/(label+'.log')).open('a') as stream:
                subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True)
            print('DONE', label, flush=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, device, p['seeds'][i::len(devices)]) for i, device in enumerate(devices)]
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'cache', 'run', 'select', 'evaluate', 'verify', 'report', 'pipeline'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--reference-dir', default=DEFAULT_REFERENCE)
    parser.add_argument('--phase', choices=('inner', 'outer'), default='inner')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command in ('prepare', 'cache', 'run', 'verify'):
        globals()[args.command](args)
    elif args.command == 'pipeline':
        out = Path(args.out_dir).resolve()
        p = protocol(out)
        for phase in ('inner', 'outer'):
            args.phase = phase
            cache(args)
            parallel_seeds(out, p, 'run', args.devices.split(','), phase)
            if phase == 'inner':
                select(out)
        evaluate(out)
        parallel_seeds(out, p, 'verify', args.devices.split(','))
        report(out)
    else:
        globals()[args.command](Path(args.out_dir).resolve())


if __name__ == '__main__':
    main()
