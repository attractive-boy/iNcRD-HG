"""Replay official TabICLv2 estimators and compare predeclared evaluation panels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from improve import tabular_foundation as experiment


def baseline(out, p, task, seed, a):
    ref = Path(p['reference_dir'])/p['baseline'][task][str(seed)]
    r = experiment.read_json(ref/'result.json')
    with np.load(ref/'scores.npz', allow_pickle=False) as saved:
        y, pairs, scores = saved['y_val'], saved['pairs_val'], saved['val']
    for key, value in experiment.metrics(y, scores, pairs).items():
        if value is not None:
            np.testing.assert_allclose(r[key], value, atol=1e-12, rtol=0)
    if p['smoke']:
        by_pair = {tuple(pair): index for index, pair in enumerate(pairs.T)}
        keep = [by_pair[tuple(pair)] for pair in a['pairs_val'].T]
        y, pairs, scores = y[keep], pairs[:, keep], scores[keep]
    np.testing.assert_array_equal(y, a['y_val'])
    np.testing.assert_array_equal(pairs, a['pairs_val'])
    return dict(task=task, method='tabm', seed=seed,
                **experiment.metrics(y, scores, pairs), elapsed_seconds=None)


def compare(frame, tasks, seeds):
    gates, deltas = {}, []
    for method in ('tabicl_default', 'tabicl_selected'):
        gates[method] = {}
        for task in tasks:
            old = frame[(frame.task == task) & (frame.method == 'tabm')].set_index('seed')
            new = frame[(frame.task == task) & (frame.method == method)].set_index('seed')
            if set(old.index) != set(seeds) or set(new.index) != set(seeds):
                raise ValueError('Incomplete comparison panel')
            difference = new[list(experiment.METRICS)]-old[list(experiment.METRICS)]
            counts = {key:int((difference[key] > 0).sum()) for key in ('auc', 'ap')}
            means = {key:float(difference[key].mean()) for key in experiment.METRICS}
            gates[method][task] = dict(mean_deltas=means, positive_seed_labels=counts,
                passed=all(means[key] > 0 and counts[key] >= (2 if len(seeds) == 3 else 1) for key in ('auc', 'ap')))
            for seed, row in difference.iterrows():
                deltas.append(dict(task=task, method=method, seed=int(seed), **row.to_dict()))
    joint = {method:all(v['passed'] for v in task_values.values()) for method, task_values in gates.items()}
    return gates, joint, pd.DataFrame(deltas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=experiment.DEFAULT_OUT)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    p = experiment.protocol(out)
    chosen = experiment.selection(out, p)
    import torch
    from tabicl import TabICLClassifier
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache, results, replays, inner = {}, {}, [], {}
    for phase in ('inner', 'outer'):
        for task, cfg, seed in experiment.jobs(out, p, phase):
            key = (phase, task)
            if key not in cache:
                cache[key] = experiment.load_features(out, p, task, phase)
            a = cache[key]
            dest = experiment.directory(out, phase, task, cfg['id'], seed)
            r = experiment.checked_result(dest, p)
            if (r['phase'], r['task'], r['config'], r['seed']) != (phase, task, cfg, seed):
                raise ValueError('Inference result identity mismatch')
            if r['feature_manifest_sha256'] != a['feature_manifest_sha256']:
                raise ValueError('Inference used different features')
            with np.load(dest/'scores.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['y_val'], a['y_val'])
                np.testing.assert_array_equal(saved['pairs_val'], a['pairs_val'])
                scores = saved['val'].copy()
            for name, value in experiment.metrics(a['y_val'], scores, a['pairs_val']).items():
                if value is not None:
                    np.testing.assert_allclose(r[name], value, atol=1e-12, rtol=0)
            model = TabICLClassifier.load(dest/'fitted.pkl', device=args.device)
            if model.n_estimators != cfg['n_estimators'] or model.random_state != seed or model.allow_auto_download:
                raise ValueError('Saved estimator settings differ')
            if experiment.parameter_digest(model) != r['parameter_sha256']:
                raise ValueError('Saved estimator parameters differ')
            if model.n_samples_in_ != len(a['y_train']):
                raise ValueError('Wrong number of context rows')
            np.testing.assert_array_equal(model.ensemble_generator_.y_, a['y_train'])
            prefix = 'raw' if cfg['input_view'] == 'raw' else 'x'
            expected_context = model.ensemble_generator_.unique_filter_.transform(model.X_encoder_.transform(a[prefix+'_train']))
            np.testing.assert_allclose(model.ensemble_generator_.X_, expected_context, atol=1e-7, rtol=1e-7)
            replay = experiment.query_scores(model, a[prefix+'_val'])
            np.testing.assert_allclose(replay, scores, atol=p['replay_atol'], rtol=p['replay_rtol'])
            entry = dict(path=str(dest.relative_to(out)), replay_max_abs=float(np.max(np.abs(replay-scores))))
            # On engineering smoke data, verify that query order/batches cannot supply labels or context.
            if p['smoke']:
                query = a[prefix+'_val']
                permuted = experiment.query_scores(model, query[::-1])[::-1]
                split = np.concatenate([experiment.query_scores(model, piece) for piece in np.array_split(query, 2)])
                np.testing.assert_allclose(permuted, replay, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(split, replay, atol=1e-5, rtol=1e-5)
                entry.update(query_permutation_max_abs=float(np.max(np.abs(permuted-replay))),
                             query_batch_max_abs=float(np.max(np.abs(split-replay))))
            replays.append(entry)
            results[(phase, task, cfg['id'], seed)] = r
            if phase == 'inner':
                inner.setdefault(task, []).append(r)
            print('REPLAYED', phase, task, cfg['id'], seed, entry['replay_max_abs'], flush=True)
            del model
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
    for task in p['tasks']:
        if experiment.select_recipe(p['configurations'], inner[task], p['seeds']) != chosen['chosen'][task]:
            raise ValueError('Cannot reproduce inner selection')
    rows = []
    for task in p['tasks']:
        for seed in p['seeds']:
            a = cache[('outer', task)]
            rows.append(baseline(out, p, task, seed, a))
            for method, cfg_id in [('tabicl_default', p['default_id']),
                                   ('tabicl_selected', chosen['chosen'][task]['config']['id'])]:
                r = results[('outer', task, cfg_id, seed)]
                rows.append(dict(task=task, method=method, seed=seed, config_id=cfg_id,
                                 **{key:r[key] for key in (*experiment.METRICS, 'elapsed_seconds', 'peak_cuda_bytes')}))
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'outer_results.csv', index=False)
    pd.DataFrame([r for r in results.values() if r['phase'] == 'inner']).drop(columns=['artifacts_sha256']).to_csv(out/'inner_results.csv', index=False)
    summary = frame.groupby(['task', 'method'])[list(experiment.METRICS)].agg(['mean', 'std'])
    summary.to_csv(out/'summary.csv')
    gates, joint, differences = compare(frame, p['tasks'], p['seeds'])
    differences.to_csv(out/'paired_deltas.csv', index=False)
    names = dict(tabm='原 TabM', tabicl_default='TabICLv2 默认', tabicl_selected='TabICLv2 内部选参')
    lines = ['# 官方 TabICLv2 与 TabM：固定图特征比较', '',
        '**工程短流程，不用于性能结论。**' if p['smoke'] else '以下是同一历史外层验证、seed 666/0/1 的均值。TabICL 只改变推理随机化，所有运行共享同一官方预训练权重。', '',
        '| 任务 | 方案 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for method in names:
            r = summary.loc[(task, method)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f |' %
                         (task, names[method], r[('auc','mean')], r[('ap','mean')], r[('macro_drug_ap','mean')]))
    lines += ['', '## 相对 TabM 的变化', '',
              '| 任务 | 方案 | ΔAUC | 正向种子标签 | ΔAP | 正向种子标签 | Δ宏 AP |',
              '|---|---|---:|---:|---:|---:|---:|']
    for method, tasks in gates.items():
        for task, value in tasks.items():
            m, c = value['mean_deltas'], value['positive_seed_labels']
            lines.append('| %s | %s | %+.6f | %d/%d | %+.6f | %d/%d | %+.6f |' %
                (task, names[method], m['auc'], c['auc'], len(p['seeds']), m['ap'], c['ap'], len(p['seeds']), m['macro_drug_ap']))
    lines += ['', '## 冻结选择与核验', '', '| 任务 | 输入视图 | 推理成员数 | 内部均值 (AUC+AP)/2 |', '|---|---|---:|---:|']
    for task, value in chosen['chosen'].items():
        cfg = value['config']
        lines.append('| %s | %s | %d | %.6f |' % (task, cfg['input_view'], cfg['n_estimators'], value['mean_selection_score']))
    lines += ['', '- 预设双任务判定：`%s`。默认与内部选参面板分开判断，不根据外层结果改选配置。' % json.dumps(joint),
        '- %d 个实际推理作业全部保存并重放；若默认与所选配置相同，两面板引用同一次预测，不重复计作独立作业。' % len(replays),
        '- 各作业核对输入标签、配对、上下文行、预训练参数、官方权重、源码与环境；重新计算指标和内部选择。',
        '- 原 TabM 使用之前已经重放验证的缓存预测，本轮重算其指标并逐行对齐；没有重新训练 TabM，也不重复计算已有检查点数量。',
        '- TabM 在前序同预算 12 配置调参中仍选择原配置。当前比较不包含等额合成预训练算力；这是现成预测器的性能比较。',
        '- 输入仍是已验证的五折图特征，目标边及反向边按原协议移除；官方预处理仅在拟合行上拟合，没有测试集评估。',
        '- TabICL 为官方分类 v2 权重，%d 种输入/成员配置全部完成内部 %d 个种子比较；模型参数始终不更新。' % (len(p['configurations']), len(p['seeds'])),
        '- 官方默认成员数为 8；固定使用 4 个成员一批、FP32、关闭 FA3/缓存，均为资源设置。短流程确认 FP32 查询分批差异低于 1e-6。',
        '- 三个种子不是三个独立预训练模型，也不是独立数据划分；与 TabM 的同号种子配对只用于记账，不能当作共同随机扰动。',
        '- 历史验证多次参与开发，结论只适用于当前开发划分，不能据此宣称统计显著或跨划分泛化。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    experiment.write_json(out/'OUTCOME.json', dict(protocol_sha256=p['sha256'], smoke=p['smoke'],
        gates=gates, joint=joint, scope_complete=True))
    experiment.write_json(out/'VALIDATION.json', dict(passed=True, protocol_sha256=p['sha256'],
        smoke=p['smoke'], completed_inferences=len(replays), all_estimators_replayed=True,
        selection_replayed=True, baseline_predictions_aligned=True, no_test_evaluation=True, replays=replays))
    print(summary.to_string(), flush=True)
    print('VERIFIED', len(replays), 'estimators; joint outcome:', joint, flush=True)


if __name__ == '__main__':
    main()
