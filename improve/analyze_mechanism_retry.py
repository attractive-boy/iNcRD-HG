"""Compare the completed PLE tuning and subsequent AP-objective trial."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from improve import backbone_replacement as parent
from improve import global_ap
from improve import numeric_embedding as original
from improve import numeric_embedding_tuning as tuning


def load_verified(module, root):
    p = module.protocol(root)
    v = json.loads((root/'VALIDATION.json').read_text())
    if p['smoke'] or not v['passed'] or not v['all_checkpoints_replayed'] or v['protocol_sha256'] != p['sha256']:
        raise ValueError('Only a verified formal experiment can be compared')
    return p, v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ple-dir', default='improve/results/numeric_embedding_tuning_20260912')
    parser.add_argument('--ap-dir', default='improve/results/global_ap_20260912')
    args = parser.parse_args()
    ple_root, ap_root = Path(args.ple_dir), Path(args.ap_dir)
    ple_p, ple_v = load_verified(tuning, ple_root)
    ap_p, ap_v = load_verified(global_ap, ap_root)
    if ple_p['reference_protocol_sha256'] != ap_p['reference_protocol_sha256']:
        raise ValueError('The studies use different reference inputs')
    repeated = []
    for stage in ('screen', 'confirm', 'outer'):
        for task, arm, cfg, seed, epochs in tuning.jobs(ple_root, ple_p, stage):
            if arm != 'tabm':
                continue
            a = parent.directory(ple_root, stage, task, arm, cfg['id'], seed)
            b = parent.directory(ap_root, stage, task, arm, cfg['id'], seed)
            left, right = parent.checked_result(a, ple_p), parent.checked_result(b, ap_p)
            for key in ('config', 'seed', 'selected_epoch', 'trained_epochs'):
                if left[key] != right[key]:
                    raise ValueError('Repeated scalar control changed: '+key)
            with np.load(a/'scores.npz') as old, np.load(b/'scores.npz') as new:
                np.testing.assert_array_equal(old['y_val'], new['y_val'])
                np.testing.assert_array_equal(old['pairs_val'], new['pairs_val'])
                np.testing.assert_allclose(old['val'], new['val'], atol=2e-6, rtol=2e-6)
                error = float(np.max(np.abs(old['val']-new['val'])))
            repeated.append(dict(stage=stage, task=task, config_id=cfg['id'], seed=seed, max_abs=error))
    reference = Path(ple_p['reference_dir'])
    old_p, _ = load_verified(original, reference)
    rows, reference_arrays = [], {}
    for task, arm, cfg, seed, _ in original.jobs(reference, old_p, 'outer'):
        if arm != 'tabm':
            continue
        dest = parent.directory(reference, 'outer', task, arm, cfg['id'], seed)
        r = parent.checked_result(dest, old_p)
        with np.load(dest/'scores.npz') as saved:
            reference_arrays[(task, seed)] = (saved['y_val'].copy(), saved['pairs_val'].copy())
        rows.append(dict(task=task, method='original_tabm', seed=seed,
                         **{k: r[k] for k in ('auc', 'ap', 'macro_drug_ap')}))
    for module, root, p in ((tuning, ple_root, ple_p), (global_ap, ap_root, ap_p)):
        for task, arm, cfg, seed, _ in module.jobs(root, p, 'outer'):
            if arm == 'tabm':
                continue
            dest = parent.directory(root, 'outer', task, arm, cfg['id'], seed)
            r = parent.checked_result(dest, p)
            with np.load(dest/'scores.npz') as saved:
                y, pairs = reference_arrays[(task, seed)]
                np.testing.assert_array_equal(saved['y_val'], y)
                np.testing.assert_array_equal(saved['pairs_val'], pairs)
                for k, value in parent.metrics(y, saved['val'], pairs).items():
                    if value is not None:
                        np.testing.assert_allclose(r[k], value, atol=1e-12, rtol=0)
            rows.append(dict(task=task, method=arm, seed=seed,
                             **{k: r[k] for k in ('auc', 'ap', 'macro_drug_ap')}))
    frame = pd.DataFrame(rows)
    frame.to_csv(ap_root/'combined_outer_results.csv', index=False)
    methods = ['original_tabm', 'ple_quantile', 'ple_tree', 'smooth_ap']
    names = dict(original_tabm='原 TabM', ple_quantile='调参分位数 PLE',
                 ple_tree='调参树分段 PLE', smooth_ap='全局 AP 软排序')
    summary = frame.groupby(['task', 'method'])[['auc', 'ap', 'macro_drug_ap']].agg(['mean', 'std'])
    summary.to_csv(ap_root/'combined_summary.csv')
    lines = ['# PLE 调参与机制切换：最终比较', '',
        '先完成 114 次 PLE/TabM 同预算调参；两种 PLE 均未通过预设双任务条件，随后完成 76 次全局 AP 软排序及对照训练。'
        '以下为同一历史外层验证、seed 666/0/1 的均值。所有参数和重训轮数由内部验证选择。', '',
        '| 任务 | 方案 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for task in ple_p['tasks']:
        for method in methods:
            r = summary.loc[(task, method)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f |' %
                (task, names[method], r[('auc', 'mean')], r[('ap', 'mean')], r[('macro_drug_ap', 'mean')]))
    lines += ['', '## 相对原 TabM 的配对差值', '',
        '| 任务 | 方案 | ΔAUC | 正向种子 | ΔAP | 正向种子 | Δ宏 AP |',
        '|---|---|---:|---:|---:|---:|---:|']
    deltas = []
    for task in ple_p['tasks']:
        baseline = frame[(frame.task == task) & (frame.method == 'original_tabm')].set_index('seed')
        for method in methods[1:]:
            target = frame[(frame.task == task) & (frame.method == method)].set_index('seed')
            if set(target.index) != set(baseline.index):
                raise ValueError('Comparison seeds differ')
            d = target[['auc', 'ap', 'macro_drug_ap']] - baseline[['auc', 'ap', 'macro_drug_ap']]
            for seed, row in d.iterrows():
                deltas.append(dict(task=task, method=method, seed=int(seed), **row.to_dict()))
            lines.append('| %s | %s | %+.6f | %d/3 | %+.6f | %d/3 | %+.6f |' %
                (task, names[method], d.auc.mean(), int((d.auc > 0).sum()), d.ap.mean(),
                 int((d.ap > 0).sum()), d.macro_drug_ap.mean()))
    pd.DataFrame(deltas).to_csv(ap_root/'combined_deltas.csv', index=False)
    ap_outcome = json.loads((ap_root/'OUTCOME.json').read_text())
    lines += ['', '## 预设结论与核验', '',
        '全局 AP 软排序的双任务判定：**%s**。条件为两任务 AUC/AP 同时高于旧/同预算 TabM，且每项至少 2/3 个种子正向。'
        % ('通过' if ap_outcome['joint']['smooth_ap'] else '未通过'), '',
        '- 两轮共 %d 个正式检查点全部重放；参数选择、训练轮数、预测指标均核对。' %
            (ple_v['completed_runs']+ap_v['completed_runs']),
        '- 两轮 %d 个重复 TabM 对照的预测最大差异为 %.9g，不能把重复对照当成新增独立种子。' %
            (len(repeated), max(v['max_abs'] for v in repeated)),
        '- 表中所有组使用完全相同的评价配对和标签；未读取测试集。',
        '- 这些结果是反复使用的开发验证划分，不能据三个初始化宣称跨划分泛化或统计显著性。', '',
        '详见 [PLE 调参报告](../numeric_embedding_tuning_20260912/REPORT.md) 与 [全局 AP 报告](REPORT.md)。', '']
    (ap_root/'RETRY_REPORT.md').write_text('\n'.join(lines))
    parent.write_json(ap_root/'COMPARISON_VALIDATION.json', dict(passed=True,
        ple_protocol_sha256=ple_p['sha256'], ap_protocol_sha256=ap_p['sha256'],
        repeated_controls=repeated, all_evaluation_pairs_equal=True,
        analysis_source_sha256=parent.digest(__file__)))
    print(summary.to_string())
    print('VERIFIED repeated scalar controls:', len(repeated))


if __name__ == '__main__':
    main()
