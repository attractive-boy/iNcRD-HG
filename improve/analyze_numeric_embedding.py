"""Summarize the completed numerical-embedding trial and verify its anchor."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from improve import backbone_replacement as parent
from improve import graph_tabm
from improve import numeric_embedding as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--reference-dir', default='improve/results/graph_tabm_20260910')
    args = parser.parse_args()
    out, reference = Path(args.out_dir), Path(args.reference_dir)
    p = experiment.protocol(out)
    original = graph_tabm.protocol(reference)
    verified = json.loads((out / 'VALIDATION.json').read_text())
    if p['smoke'] or not verified['passed'] or not verified['all_checkpoints_replayed']:
        raise ValueError('A fully replayed formal run is required')
    rows, anchors = [], []
    for stage in ('screen', 'confirm', 'outer'):
        for task, arm, cfg, seed, _ in experiment.jobs(out, p, stage):
            directory = parent.directory(out, stage, task, arm, cfg['id'], seed)
            row = parent.checked_result(directory, p)
            rows.append(row)
            if arm == 'tabm':
                old_dir = parent.directory(reference, stage, task, arm, cfg['id'], seed)
                old = parent.checked_result(old_dir, original)
                if row['selected_epoch'] != old['selected_epoch']:
                    raise AssertionError('Scalar anchor selected a different epoch')
                with np.load(directory / 'scores.npz') as new, np.load(old_dir / 'scores.npz') as old_scores:
                    np.testing.assert_array_equal(new['y_val'], old_scores['y_val'])
                    np.testing.assert_array_equal(new['pairs_val'], old_scores['pairs_val'])
                    np.testing.assert_allclose(new['val'], old_scores['val'], atol=2e-6, rtol=2e-6)
                    error = float(np.max(np.abs(new['val'] - old_scores['val'])))
                anchors.append(dict(stage=stage, task=task, seed=seed, config_id=cfg['id'], max_abs=error))
    feature_checks = []
    for phase in ('inner', 'outer'):
        for task in p['tasks']:
            relative = Path('features') / phase / task / 'arrays.npz'
            with np.load(out / relative) as new, np.load(reference / relative) as old:
                if set(new.files) != set(old.files):
                    raise AssertionError('Feature-cache schema changed')
                for key in old.files:
                    np.testing.assert_array_equal(new[key], old[key])
            feature_checks.append(str(relative))
    if len(rows) != verified['completed_runs']:
        raise AssertionError('Wrong number of verified runs')
    outer = pd.DataFrame([r for r in rows if r['stage'] == 'outer'])
    summary = outer.groupby(['task', 'arm'])[['auc', 'ap', 'macro_drug_ap', 'params', 'macro_drugs']].agg(['mean', 'std'])
    summary.to_csv(out / 'summary.csv')
    differences = []
    for task in p['tasks']:
        baseline = outer[(outer.task == task) & (outer.arm == 'tabm')].set_index('seed')
        for arm in p['arms']:
            if arm == 'tabm':
                continue
            candidate = outer[(outer.task == task) & (outer.arm == arm)].set_index('seed')
            for metric in ('auc', 'ap', 'macro_drug_ap'):
                delta = (candidate[metric] - baseline[metric]).reindex(p['refit_seeds'])
                differences.append(dict(task=task, arm=arm, metric=metric, mean=float(delta.mean()),
                    std=float(delta.std()), positive_seeds=int((delta > 0).sum()), seeds=len(delta),
                    by_seed={str(s): float(delta.loc[s]) for s in p['refit_seeds']}))
    delta_frame = pd.DataFrame(differences)
    delta_frame.to_csv(out / 'paired_deltas.csv', index=False)
    embedding_differences = []
    for task in p['tasks']:
        linear = outer[(outer.task == task) & (outer.arm == 'linear')].set_index('seed')
        for arm in ('ple_quantile', 'ple_tree'):
            candidate = outer[(outer.task == task) & (outer.arm == arm)].set_index('seed')
            for metric in ('auc', 'ap', 'macro_drug_ap'):
                delta = (candidate[metric] - linear[metric]).reindex(p['refit_seeds'])
                embedding_differences.append(dict(task=task, arm=arm, reference='linear', metric=metric,
                    mean=float(delta.mean()), positive_seeds=int((delta > 0).sum()), seeds=len(delta)))
    pd.DataFrame(embedding_differences).to_csv(out / 'embedding_control_deltas.csv', index=False)
    outcome = {}
    for arm in ('ple_quantile', 'ple_tree'):
        criteria = [d for d in differences if d['arm'] == arm and d['metric'] in ('auc', 'ap')]
        outcome[arm] = dict(passed=all(d['mean'] > 0 and d['positive_seeds'] >= 2 for d in criteria),
                            criteria=criteria)
    parent.write_json(out / 'ANCHOR_VALIDATION.json', dict(passed=True, reference_dir=str(reference.resolve()),
        scalar_anchor_runs=anchors, feature_arrays_equal=feature_checks,
        analysis_source_sha256=parent.digest(__file__)))
    parent.write_json(out / 'OUTCOME.json', dict(protocol_sha256=p['sha256'], decision=outcome,
        differences=differences, embedding_control_differences=embedding_differences,
        all_arms_reported=True, no_test_evaluation=True))
    refit = parent.decision(out, 'REFIT.json', p)['chosen']
    lines = ['# 数值嵌入 + TabM：同条件机制对照', '',
        '使用与现有 TabM 完全相同的图特征、样本划分、分位数预处理和训练目标，比较普通标量、'
        '逐特征线性/ReLU 嵌入、分位数 PLE 和监督树分段 PLE。所有嵌入维度固定为 16，PLE 最多 16 个区间。', '',
        '两种 PLE 的双任务筛选判定：' + '；'.join(arm + (' 通过' if result['passed'] else ' 未通过')
            for arm, result in outcome.items()) + '。判定要求两个任务的 AUC/AP 均值都高于标量 TabM，且每项至少 2/3 个种子为正。', '',
        '## 历史外层验证：三个初始化的均值 ± 标准差', '',
        '| 任务 | 组 | AUC | AP | 按药物宏 AP | 可计算宏 AP 的药物数 |',
        '|---|---|---:|---:|---:|---:|']
    for task in p['tasks']:
        for arm in p['arms']:
            row = summary.loc[(task, arm)]
            cells = ['%.6f ± %.6f' % (row[(metric, 'mean')], row[(metric, 'std')])
                     for metric in ('auc', 'ap', 'macro_drug_ap')]
            lines.append('| %s | %s | %s | %d |' % (task, arm, ' | '.join(cells), row[('macro_drugs', 'mean')]))
    lines += ['', '分位数 PLE 相对标量 TabM 的主要变化：', '']
    for task in p['tasks']:
        changes = {d['metric']: d for d in differences if d['task'] == task and d['arm'] == 'ple_quantile'}
        lines.append('- %s：ΔAUC %+.6f（%d/3 正向），ΔAP %+.6f（%d/3 正向），Δ宏 AP %+.6f。' %
            (task, changes['auc']['mean'], changes['auc']['positive_seeds'], changes['ap']['mean'],
             changes['ap']['positive_seeds'], changes['macro_drug_ap']['mean']))
    lines += ['', '## 相对原 TabM 的配对差值', '',
        '差值是绝对指标单位；乘以 100 才是百分点。正向种子按 666、0、1 配对计算。', '',
        '| 任务 | 组 | 指标 | 平均差值 | 正向种子 |', '|---|---|---|---:|---:|']
    for d in differences:
        lines.append('| %s | %s | %s | %+.6f | %d/%d |' % (d['task'], d['arm'], d['metric'],
                     d['mean'], d['positive_seeds'], d['seeds']))
    lines += ['', '## PLE 相对普通数值嵌入的增量', '',
        '这一对照用于区分分段基函数与简单扩维的影响；两者的参数量仍不完全相同。', '',
        '| 任务 | 组 | 指标 | 相对 linear 的平均差值 | 正向种子 |', '|---|---|---|---:|---:|']
    for d in embedding_differences:
        lines.append('| %s | %s | %s | %+.6f | %d/%d |' % (d['task'], d['arm'], d['metric'],
                     d['mean'], d['positive_seeds'], d['seeds']))
    lines += ['', '## 冻结选参', '',
        '每组只搜索学习率 0.001/0.003，seed 666 按内部最近五轮平均 (AUC+AP)/2 选择实际检查点；'
        '随后用 0、1 两个种子确认，取三个所选轮数的中位数，在完整拟合集重训。外层训练过程不计算外层指标。', '',
        '| 任务 | 组 | 学习率 | 重训轮数 | 参数量 |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for arm in p['arms']:
            chosen = refit[task][arm]
            lines.append('| %s | %s | %g | %d | %d |' % (task, arm, chosen['config']['lr'],
                chosen['epochs'], summary.loc[(task, arm), ('params', 'mean')]))
    lines += ['', '## 核验与解释边界', '',
        '- 全部 %d 个正式检查点重放通过；4 份内/外层特征缓存与现有 TabM 逐项完全一致。' % len(rows),
        '- %d 个标量 TabM 对照复现原检查点预测，最大误差 %.9g。' %
            (len(anchors), max(a['max_abs'] for a in anchors)),
        '- 分段点只使用拟合数据；树分段使用拟合标签，这是受监督的输入变换。没有使用验证标签确定分段点。',
        '- 普通嵌入对照具有相同的后续网络宽度和输入维度；PLE 仍额外增加基函数投影参数，不能据此声称参数量完全匹配。',
        '- 这里的 AP 使用 sklearn.average_precision_score，与梯形积分的 PR-AUC 不完全等价。',
        '- 外层是多次研究中重复使用的验证划分。三个初始化衡量初始化波动，不能证明跨划分泛化或统计显著性。',
        '- 零标签仍代表未观测关联；输出不是经过校准的生物学概率。此轮未载入测试集。',
        '- PLE 是已有数值编码方法（Gorishniy 等，NeurIPS 2022）；本实验只检验它在当前任务中的增量。', '',
        '原始结果见 `results.csv`，逐种子差值见 `paired_deltas.csv`，重放见 `VALIDATION.json`，'
        '基线和特征一致性见 `ANCHOR_VALIDATION.json`，预设判定见 `OUTCOME.json`。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(summary[['auc', 'ap', 'macro_drug_ap']].to_string())
    print(json.dumps({arm: result['passed'] for arm, result in outcome.items()}, indent=2))
    print('ANCHOR VERIFIED', len(anchors), 'runs; report:', out / 'REPORT.md')


if __name__ == '__main__':
    main()
