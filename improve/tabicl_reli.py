"""Frozen-weight RELI-TabICL validation with a shared two-task selection gate.

prepare -> run inner -> verify inner -> select -> (if eligible) run/verify outer.
The original test arrays are never loaded. A failed inner gate stops the trial.
"""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from improve import tabular_foundation as base
from improve import tabicl_graph_context as graph
from improve import tabicl_graph_context_modules as graph_math
from improve import tabicl_reli_modules as reli


DEFAULT_OUT = 'improve/results/tabicl_reli_20260912'
DEFAULT_INPUTS = 'improve/results/tabicl_reli_inputs_20260912'
DEFAULT_PARENT = 'improve/results/tabicl_graph_context_20260912'
OWN = ('tabicl_reli.py', 'tabicl_reli_modules.py', 'tabicl_reli_inputs.py',
       'tabicl_reli_mature_refresh.py', 'test_tabicl_reli.py', 'tabular_foundation.py',
       'tabicl_graph_context.py', 'tabicl_graph_context_modules.py')
METRICS = ('auc', 'ap', 'macro_drug_ap')


def sources():
    return {name: base.digest(Path(__file__).with_name(name)) for name in OWN}


def prepare(args):
    import tabicl
    out, inputs, parent = map(lambda p: Path(p).resolve(), (args.out_dir, args.inputs_dir, args.parent_dir))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh experiment directory')
    gp = graph.protocol(parent)
    selection = graph.selection(parent, gp)
    for task, expected in [('LncDrug', 'query_bag_050'), ('MiDrug', 'fold_match_050')]:
        if selection['chosen'][task]['recipe']['id'] != expected:
            raise ValueError('Reference graph-context recipe differs')
    required = ['inputs.json', 'MiDrug_sequences.json', 'LncDrug_sequences.json',
                'rna_LncDrug.npz', 'rna_MiDrug.npz', 'drug.npz', 'kernels.npz',
                'rna_LncDrug.json', 'rna_MiDrug.json', 'drug.json', 'kernels.json']
    feature_sha = {name: base.digest(inputs/name) for name in required}
    kernel_meta = base.read_json(inputs/'kernels.json')
    if kernel_meta['output_sha256'] != feature_sha['kernels.npz']:
        raise ValueError('Kernel artifact differs from its source manifest')
    for name, expected in kernel_meta['files'].items():
        if base.digest(inputs/name) != expected:
            raise ValueError('Raw entity evidence changed')
    installed = Path(tabicl.__file__).parent/'_model'
    implementation = {str(installed/name): base.digest(installed/name)
                      for name in ['learning.py', 'encoders.py', 'layers.py', 'attention.py', 'kv_cache.py']}
    p = base.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        tasks=['LncDrug', 'MiDrug'], seeds=[666] if args.smoke else [666, 0, 1],
        inference_members=2 if args.smoke else 32, recipes=reli.recipes(),
        parent_dir=str(parent), parent_protocol_sha256=gp['sha256'],
        parent_selection_sha256=base.digest(parent/'SELECTION.json'),
        input_configs=gp['input_configs'], graph_recipes={t: selection['chosen'][t]['recipe'] for t in gp['tasks']},
        inputs_dir=str(inputs), inputs_sha256=feature_sha, source_sha256=sources(), installed_source_sha256=implementation,
        parameter_sha256=gp['parameter_sha256'],
        mechanism='Frozen RNA-FM local sites and atom-centered Morgan evidence. Shared local/pooled endpoint comparison forms a class-blind relation bias only in the final ICL query-to-support attention. Prefix and support representations are unchanged.',
        parameter_updates=0,
        shared_parameters='One mode and strength for both tasks; strengths 0.03125, 0.125, 0.5, 2.0; pooled comparison is a matched-input control. All neural weights and random projections fixed.',
        input_scope='Entity-only unsupervised transductive attributes. No association labels in sequence/atom kernels. Missing entities retained. Long RNA sites sample the full transcript span with a fixed budget; not all bases are encoded.',
        zero_control='Apply the reference graph-context composition to unmodified official predictions; also compare against original TabICLv2. Every nonzero candidate modifies both tasks and all their reference composition branches.',
        selection='Only local candidates may advance. For each LncDrug metric, mean candidate must be >= both the original TabICLv2 and graph-context mean. MiDrug mean AUC and AP must exceed graph-context, each with at least 2/3 positive paired seed differences. Choose eligible local candidate by mean MiDrug (AUC+AP)/2, exact ties prefer smaller strength. Otherwise select zero and stop.',
        outer_policy='Before outer loading require own frozen eligible inner selection and full inner replay. Evaluate only zero, selected local and pooled at the same strength; do not reselect on outer.',
        replay_atol=1e-6, replay_rtol=1e-6, query_invariance_atol=5e-5, query_invariance_rtol=1e-5,
        query_effect_invariance_atol=1e-5, query_effect_invariance_rtol=1e-5,
        query_numerics='Before metric inspection, the smoke official unmodified model itself changed by 1.623e-5 across query shapes. Allow 5e-5 absolute cross-shape roundoff, and separately require the adapter-induced probability change to remain invariant to 1e-5. Saved-state replay remains 1e-6.',
        native_icl_atol=1e-5, native_icl_rtol=1e-5, no_test_evaluation=True,
        limitations=['First fixed-matcher prototype; the proposed learned interaction projection is not trained here.',
                     'Historical inner/outer split; seed labels are inference randomness, not independent generalization replicates.',
                     'No RNA-family, scaffold, or cross-RNA weight-transfer benchmark in this first improvement gate.',
                     'No guarantee of non-degradation under future distribution shifts.']))
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name), out/'source_snapshot'/name)
    base.write_json(out/'protocol.json', p)
    print('PROTOCOL_FROZEN', out, p['sha256'], flush=True)


def protocol(out):
    out = Path(out)
    p = base.check_seal(base.read_json(out/'protocol.json'))
    if p['source_sha256'] != sources():
        raise ValueError('Frozen RELI source changed; keep artifacts and start a new trial')
    if base.digest(Path(p['parent_dir'])/'SELECTION.json') != p['parent_selection_sha256']:
        raise ValueError('Reference selection changed')
    for path, expected in p['installed_source_sha256'].items():
        if base.digest(path) != expected:
            raise ValueError('Installed TabICL implementation changed')
    for name, expected in p['inputs_sha256'].items():
        if base.digest(Path(p['inputs_dir'])/name) != expected:
            raise ValueError('Frozen entity evidence changed: '+name)
    return p


def chosen(out, p):
    s = base.check_seal(base.read_json(Path(out)/'SELECTION.json'))
    if s['protocol_sha256'] != p['sha256']:
        raise ValueError('Selection from another protocol')
    if not s['eligible'] or s['recipe']['mode'] != 'local':
        raise ValueError('No eligible local candidate; outer evaluation is prohibited')
    if base.digest(Path(out)/'REPLAY_inner.json') != s['replay_sha256']:
        raise ValueError('Inner replay record changed')
    for name, expected in s['results_sha256'].items():
        if base.digest(Path(out)/name) != expected:
            raise ValueError('Inner result changed after selection')
    return s


def phase_recipes(out, p, phase):
    if phase == 'inner':
        return p['recipes']
    selected = chosen(out, p)['recipe']
    return [p['recipes'][0], selected,
            next(r for r in p['recipes'] if r['mode'] == 'pooled' and r['strength'] == selected['strength'])]


def features(out, p, task, phase):
    if phase not in ('inner', 'outer') or task not in p['tasks']:
        raise ValueError('Unknown task or phase')
    if phase == 'outer':
        chosen(out, p)
    gp = graph.protocol(p['parent_dir'])
    if gp['sha256'] != p['parent_protocol_sha256']:
        raise ValueError('Reference protocol changed')
    data = graph.features(p['parent_dir'], gp, task, phase)
    if p['smoke']:
        keep = np.sort(np.concatenate([
            np.flatnonzero(data['fold_ids'] == f)[base.balanced_subset(data['y_train'][data['fold_ids'] == f], 32, 20260912+f)]
            for f in range(5)]))
        for prefix in ('raw', 'x', 'y'):
            data[prefix+'_train'] = data[prefix+'_train'][keep]
        data['pairs_train'], data['fold_ids'] = data['pairs_train'][:, keep], data['fold_ids'][keep]
        keep = base.balanced_subset(data['y_val'], 40, 20260912)
        for prefix in ('raw', 'x', 'y'):
            data[prefix+'_val'] = data[prefix+'_val'][keep]
        data['pairs_val'] = data['pairs_val'][:, keep]
        for prefix in ('raw', 'x'):
            data[prefix+'_query_views'] = data[prefix+'_query_views'][:, keep]
    base.validate_arrays(data)
    return data


def kinds(task):
    return ['full'] if task == 'LncDrug' else ['full']+['fold'+str(i) for i in range(5)]


def job_inputs(p, task, data, kind):
    prefix = 'raw' if p['input_configs'][task]['input_view'] == 'raw' else 'x'
    x, full, views = data[prefix+'_train'], data[prefix+'_val'], data[prefix+'_query_views']
    if kind == 'full':
        keep = np.arange(len(x))
        queries = np.concatenate([full[None], views]) if task == 'LncDrug' else full[None]
    else:
        fold = int(kind[-1])
        keep = np.flatnonzero(data['fold_ids'] == fold)
        queries = views[fold:fold+1]
    return x[keep], data['y_train'][keep], data['pairs_train'][:, keep], queries


def load_kernels(p):
    with np.load(Path(p['inputs_dir'])/'kernels.npz', allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files if key.endswith(('_local', '_pooled', '_present', '_names'))}


def directory(out, phase, task, seed, kind):
    return Path(out)/phase/task/('s'+str(seed))/kind


def checked_result(dest, p):
    result = base.read_json(Path(dest)/'result.json')
    if result['protocol_sha256'] != p['sha256']:
        raise ValueError('Result belongs to another trial')
    for name, expected in result['files'].items():
        if base.digest(Path(dest)/name) != expected:
            raise ValueError('Saved RELI state changed')
    return result


def run(args):
    out = Path(args.out_dir)
    p = protocol(out)
    configs = phase_recipes(out, p, args.phase)
    data = features(out, p, args.task, args.phase)
    kernels = load_kernels(p)
    gp = base.read_json(Path(p['parent_dir'])/'protocol.json')
    tp = base.read_json(Path(gp['parent_dir'])/'protocol.json')
    for seed in p['seeds']:
        for kind in kinds(args.task):
            dest = directory(out, args.phase, args.task, seed, kind)
            if (dest/'result.json').exists():
                checked_result(dest, p)
                continue
            started = time.monotonic()
            x, labels, support_pairs, queries = job_inputs(p, args.task, data, kind)
            query_pairs = np.tile(data['pairs_val'], (1, len(queries)))
            biases = {mode: reli.pair_bias(kernels, args.task, query_pairs, support_pairs, mode)
                      for mode in ('local', 'pooled')}
            torch.cuda.set_device(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            clf = base.fit_context(base.estimator(dict(n_estimators=p['inference_members']), seed, tp, args.device), x, labels)
            if base.parameter_digest(clf) != p['parameter_sha256']:
                raise ValueError('Model does not have the original pretrained parameters')
            print('RUN', args.phase, args.task, seed, kind, 'support', len(x), 'query_views', len(queries), flush=True)
            scores, numerical = reli.predict_all(clf, queries.reshape(-1, 36), biases, configs)
            if base.parameter_digest(clf) != p['parameter_sha256']:
                raise ValueError('Frozen pretrained parameters changed')
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/'scores.npz', scores=np.stack([scores[c['id']].reshape(queries.shape[:2]) for c in configs]),
                                recipe_ids=np.array([c['id'] for c in configs]), y_val=data['y_val'], pairs_val=data['pairs_val'])
            clf.save(dest/'fitted.pkl', save_model_weights=False, save_training_data=True, save_kv_cache=False)
            base.write_json(dest/'result.json', dict(protocol_sha256=p['sha256'], task=args.task, phase=args.phase,
                seed=seed, kind=kind, recipes=configs, context_rows=len(x), query_views=len(queries),
                numerical=numerical, parameter_sha256=p['parameter_sha256'], graph_manifest_sha256=data['graph_manifest_sha256'],
                elapsed_seconds=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device),
                files={name: base.digest(dest/name) for name in ['scores.npz', 'fitted.pkl']}))
            print('SAVED', args.task, seed, kind, 'seconds', round(time.monotonic()-started, 1), flush=True)
            del clf, scores, biases
            gc.collect()
            torch.cuda.empty_cache()


def task_scores(out, p, phase, task, seed, data):
    configs = phase_recipes(out, p, phase)
    views = {}
    for kind in kinds(task):
        dest = directory(out, phase, task, seed, kind)
        checked_result(dest, p)
        with np.load(dest/'scores.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['recipe_ids'], [c['id'] for c in configs])
            np.testing.assert_array_equal(archive['pairs_val'], data['pairs_val'])
            np.testing.assert_array_equal(archive['y_val'], data['y_val'])
            views[kind] = archive['scores'].copy()
    original = views['full'][:, 0]
    if task == 'LncDrug':
        component = views['full'][:, 1:].mean(1)
    else:
        component = np.stack([views[kind][:, 0] for kind in kinds(task)[1:]]).mean(0)
    scores = .5*original+.5*component
    return {cfg['id']: values for cfg, values in zip(configs, scores)}, original[0]


def verify(args):
    from tabicl import TabICLClassifier
    out = Path(args.out_dir)
    p = protocol(out)
    configs = phase_recipes(out, p, args.phase)
    kernels = load_kernels(p)
    records = []
    for task in p['tasks']:
        data = features(out, p, task, args.phase)
        for seed in p['seeds']:
            for kind in kinds(task):
                dest = directory(out, args.phase, task, seed, kind)
                result = checked_result(dest, p)
                x, labels, support_pairs, queries = job_inputs(p, task, data, kind)
                query_pairs = np.tile(data['pairs_val'], (1, len(queries)))
                biases = {mode: reli.pair_bias(kernels, task, query_pairs, support_pairs, mode) for mode in ('local', 'pooled')}
                clf = TabICLClassifier.load(dest/'fitted.pkl', device=args.device)
                # Loading can reset the global Torch thread count; keep the original FP32 policy.
                torch.set_num_threads(4)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                scores, numerical = reli.predict_all(clf, queries.reshape(-1, 36), biases, configs)
                with np.load(dest/'scores.npz', allow_pickle=False) as archive:
                    expected = archive['scores']
                actual = np.stack([scores[c['id']].reshape(queries.shape[:2]) for c in configs])
                np.testing.assert_allclose(actual, expected, atol=p['replay_atol'], rtol=p['replay_rtol'])
                if base.parameter_digest(clf) != p['parameter_sha256']:
                    raise ValueError('Reloaded pretrained parameters changed')
                record = dict(task=task, seed=seed, kind=kind, replay_max_abs=float(np.max(np.abs(actual-expected))),
                              result_sha256=base.digest(dest/'result.json'), numerical=numerical)
                if p['smoke']:
                    rng = np.random.default_rng(13)
                    order = rng.permutation(len(query_pairs.T))
                    perm_scores, _ = reli.predict_all(clf, queries.reshape(-1, 36)[order],
                        {mode: values[order] for mode, values in biases.items()}, configs)
                    maximum = 0.
                    effect_maximum = 0.
                    for cfg in configs:
                        np.testing.assert_allclose(perm_scores[cfg['id']], scores[cfg['id']][order],
                            atol=p['query_invariance_atol'], rtol=p['query_invariance_rtol'])
                        maximum = max(maximum, float(np.max(np.abs(perm_scores[cfg['id']]-scores[cfg['id']][order]))))
                        original_effect = (scores[cfg['id']]-scores['zero'])[order]
                        permuted_effect = perm_scores[cfg['id']]-perm_scores['zero']
                        np.testing.assert_allclose(permuted_effect, original_effect,
                            atol=p['query_effect_invariance_atol'], rtol=p['query_effect_invariance_rtol'])
                        effect_maximum = max(effect_maximum, float(np.max(np.abs(permuted_effect-original_effect))))
                    first = order[:max(1, len(order)//3)]
                    subset, _ = reli.predict_all(clf, queries.reshape(-1, 36)[first],
                        {mode: values[first] for mode, values in biases.items()}, configs)
                    for cfg in configs:
                        np.testing.assert_allclose(subset[cfg['id']], scores[cfg['id']][first],
                            atol=p['query_invariance_atol'], rtol=p['query_invariance_rtol'])
                        maximum = max(maximum, float(np.max(np.abs(subset[cfg['id']]-scores[cfg['id']][first]))))
                        original_effect = (scores[cfg['id']]-scores['zero'])[first]
                        subset_effect = subset[cfg['id']]-subset['zero']
                        np.testing.assert_allclose(subset_effect, original_effect,
                            atol=p['query_effect_invariance_atol'], rtol=p['query_effect_invariance_rtol'])
                        effect_maximum = max(effect_maximum, float(np.max(np.abs(subset_effect-original_effect))))
                    record['query_invariance_max_abs'] = maximum
                    record['query_effect_invariance_max_abs'] = effect_maximum
                records.append(record)
                print('REPLAYED', task, seed, kind, record['replay_max_abs'], flush=True)
                del clf, scores, biases
                gc.collect()
                torch.cuda.empty_cache()
    base.write_json(out/('REPLAY_'+args.phase+'.json'), dict(protocol_sha256=p['sha256'], passed=True,
        phase=args.phase, all_saved_states_replayed=True, states=len(records), records=records,
        no_test_evaluation=True))


def metric_table(out, p, phase):
    rows, results = [], {}
    gp = base.read_json(Path(p['parent_dir'])/'protocol.json')
    for task in p['tasks']:
        data = features(out, p, task, phase)
        for seed in p['seeds']:
            scores, original = task_scores(out, p, phase, task, seed, data)
            for key, value in [('original_tabicl', original), *scores.items()]:
                rows.append(dict(task=task, seed=seed, recipe=key, **base.metrics(data['y_val'], value, data['pairs_val'])))
            if not p['smoke']:
                parent_original, parent_components, _ = graph.components(p['parent_dir'], gp, phase, task, seed, data)
                parent_graph = graph_math.combine(parent_original, parent_components, p['graph_recipes'][task])
                for name, own, reference in [('original', original, parent_original), ('graph', scores['zero'], parent_graph)]:
                    np.testing.assert_allclose(own, reference, atol=1e-5, rtol=1e-5)
                    results[f'{task}/s{seed}/{name}'] = float(np.max(np.abs(own-reference)))
            for kind in kinds(task):
                path = directory(out, phase, task, seed, kind)/'result.json'
                results[str(path.relative_to(out))] = base.digest(path)
    table = pd.DataFrame(rows)
    table.to_csv(Path(out)/(phase+'_metrics.csv'), index=False)
    return table, results


def select(args):
    out = Path(args.out_dir)
    p = protocol(out)
    replay = base.read_json(out/'REPLAY_inner.json')
    if replay['protocol_sha256'] != p['sha256'] or not replay['all_saved_states_replayed']:
        raise ValueError('Complete inner replay before selection')
    table, records = metric_table(out, p, 'inner')
    means = table.groupby(['task', 'recipe'])[list(METRICS)].mean()
    candidates = []
    for cfg in [c for c in p['recipes'] if c['mode'] == 'local']:
        failures, deltas = [], {}
        for task in p['tasks']:
            deltas[task] = {}
            for metric in METRICS:
                value = float(means.loc[(task, cfg['id']), metric])
                original = float(means.loc[(task, 'original_tabicl'), metric])
                reference = float(means.loc[(task, 'zero'), metric])
                deltas[task][metric] = dict(vs_graph=value-reference, vs_original=value-original)
                if task == 'LncDrug' and (not np.isfinite(value) or value < max(original, reference)):
                    failures.append(task+':'+metric)
                if task == 'MiDrug' and metric in ('auc', 'ap'):
                    candidate_values = table[(table.task == task) & (table.recipe == cfg['id'])].sort_values('seed')[metric].to_numpy()
                    reference_values = table[(table.task == task) & (table.recipe == 'zero')].sort_values('seed')[metric].to_numpy()
                    needed = 1 if p['smoke'] else 2
                    if not value > reference or int(np.sum(candidate_values > reference_values)) < needed:
                        failures.append(task+':'+metric)
        candidates.append(dict(recipe=cfg, eligible=not failures, failed_conditions=failures, deltas=deltas,
                               criterion=float(means.loc[('MiDrug', cfg['id']), ['auc', 'ap']].mean())))
    eligible = [r for r in candidates if r['eligible'] and not p['smoke']]
    best = sorted(eligible, key=lambda r: (-r['criterion'], r['recipe']['strength']))[0] if eligible else None
    choice = base.seal(dict(protocol_sha256=p['sha256'], eligible=bool(best),
        recipe=best['recipe'] if best else p['recipes'][0], candidates=candidates,
        replay_sha256=base.digest(out/'REPLAY_inner.json'),
        results_sha256={k: v for k, v in records.items() if k.endswith('result.json')},
        reference_alignment_max_abs={k: v for k, v in records.items() if not k.endswith('result.json')},
        no_test_evaluation=True, smoke=p['smoke']))
    base.write_json(out/'SELECTION.json', choice)
    print('SELECTED', choice['recipe'], 'eligible', choice['eligible'], flush=True)
    for candidate in candidates:
        print(candidate['recipe']['id'], candidate['failed_conditions'], candidate['deltas'], flush=True)


def report(args):
    out = Path(args.out_dir)
    p = protocol(out)
    selection = base.read_json(out/'SELECTION.json')
    table, _ = metric_table(out, p, args.phase)
    means = table.groupby(['task', 'recipe'])[list(METRICS)].mean()
    text = ['# RELI-TabICL 共享局部匹配原型验证', '',
            '工程短流程；不可解释其指标。' if p['smoke'] else '本轮以相同历史内部划分、三个推理种子、每种子 32 个成员验证共享原型。', '',
            '所有神经网络权重冻结。新改动作用于最后一层 ICL 的查询到支持注意力；两任务使用同一组强度候选。',
            'pooled 与 local 使用完全相同的 RNA-FM 局部表示、药物原子表示和支持案例；差异是先池化还是先局部匹配。', '',
            '| 任务 | 方案 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for (task, recipe), row in means.iterrows():
        text.append(f"| {task} | {recipe} | {row.auc:.6f} | {row.ap:.6f} | {row.macro_drug_ap:.6f} |")
    text += ['', '内部选择：`'+selection['recipe']['id']+'`；满足共同条件：'+str(selection['eligible'])+'。',
             'zero 是未改动的图上下文组合，original_tabicl 是原始 TabICLv2。LncDrug 三项指标均对比两个参考的较强者；MiDrug 要求 AUC/AP 提升且至少两个种子方向为正。', '']
    for candidate in selection['candidates']:
        text.append('- `'+candidate['recipe']['id']+'` 未满足项：'+(', '.join(candidate['failed_conditions']) or '无')+'。')
    text += ['', '完整数字见 `'+args.phase+'_metrics.csv`，逐候选判定与差值见 `SELECTION.json`，保存状态重放见 `REPLAY_'+args.phase+'.json`。',
             '', '这不是完整可学习匹配架构的最终评价。当前原型没有训练新的投影，也没有进行家族/药物骨架留出或跨 RNA 参数迁移；推理种子不等于独立数据重复。',
             '原始测试数组未读取。没有合格内部候选时停止，不依据外层成绩调整方案。', '']
    (out/('REPORT.md' if args.phase == 'inner' else 'REPORT_outer.md')).write_text('\n'.join(text))
    base.write_json(out/'STATUS.json', dict(protocol_sha256=p['sha256'],
        status='inner_passed' if selection['eligible'] else 'stopped_after_inner_validation',
        selected=selection['recipe'], inner_eligible=selection['eligible'],
        outer_evaluated=(out/'REPLAY_outer.json').exists(), no_test_evaluation=True,
        neural_parameter_updates=0, full_learned_matcher_evaluated=False))
    print('REPORT_SAVED', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'verify', 'select', 'report'])
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--inputs-dir', default=DEFAULT_INPUTS)
    parser.add_argument('--parent-dir', default=DEFAULT_PARENT)
    parser.add_argument('--task', choices=['LncDrug', 'MiDrug'])
    parser.add_argument('--phase', choices=['inner', 'outer'], default='inner')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'run' and args.task is None:
        parser.error('--task required')
    {'prepare': prepare, 'run': run, 'verify': verify, 'select': select, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
