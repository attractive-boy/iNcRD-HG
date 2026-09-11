"""Report predefined screens and within-drug validation performance.

Reads completed, verified predictions. This script never trains a model or
changes checkpoint selection. Its extra metrics are diagnostic, not selectors.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from improve.expression_consensus import ARMS, TASKS, check_result, load_experiment
from improve.first_round_features import write_json


def within_drug(labels, pairs, predictions):
    aucs, aps, covered, drugs = [], [], 0, []
    for drug in np.unique(pairs[1]):
        selected = pairs[1] == drug
        y = labels[selected]
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, predictions[selected]))
        aps.append(average_precision_score(y, predictions[selected]))
        covered += len(y)
        drugs.append(int(drug))
    if not aucs:
        raise ValueError('No drug has both validation classes')
    return {'macro_drug_auc': float(np.mean(aucs)), 'macro_drug_aupr': float(np.mean(aps)),
            'eligible_drugs': len(drugs), 'covered_pairs': covered,
            'validation_pairs': len(labels), 'eligible_drug_ids': drugs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    protocol = load_experiment(out)
    verification = json.loads((out / 'VALIDATION.json').read_text())
    if not verification['passed'] or verification['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Run report verification first')
    frame = pd.read_csv(out / 'results.csv')
    rows, degree_controls = [], []
    for task in TASKS:
        with np.load(out / 'inputs' / task / 'arrays.npz') as arrays:
            degree = np.bincount(arrays['pairs_train'][1, arrays['y_train'] == 1],
                                 minlength=len(arrays['x_drug']))
            scores = np.log1p(degree[arrays['pairs_val'][1]])
            degree_controls.append(dict(task=task,
                val_auc=float(roc_auc_score(arrays['y_val'], scores)),
                val_aupr=float(average_precision_score(arrays['y_val'], scores)),
                **within_drug(arrays['y_val'], arrays['pairs_val'], scores)))
        for seed in protocol['seeds']:
            for arm in ARMS:
                directory = out / 'runs' / task / ('s' + str(seed)) / arm
                check_result(directory, protocol)
                with np.load(directory / 'scores.npz') as scores:
                    row = within_drug(scores['y_val'], scores['pairs_val'], scores['val'])
                rows.append(dict(task=task, seed=seed, arm=arm, **row))
    pd.DataFrame(rows).to_csv(out / 'within_drug.csv', index=False)
    write_json(out / 'degree_controls.json', degree_controls)
    index = frame.set_index(['task', 'arm', 'seed'])
    comparisons = []
    screens = {}
    for candidate in ARMS:
        if candidate in ('base', 'pca'):
            continue
        task_passes = []
        for task in TASKS:
            row = dict(task=task, candidate=candidate, control='pca')
            for metric in ('val_auc', 'val_aupr', 'val_criterion'):
                delta = np.array([index.loc[(task, candidate, s), metric] - index.loc[(task, 'pca', s), metric] for s in protocol['seeds']])
                row[metric + '_delta'] = float(delta.mean())
                row[metric + '_positive_seeds'] = int((delta > 0).sum())
            row['passes'] = bool(row['val_auc_delta'] > 0 and row['val_aupr_delta'] > 0
                                 and row['val_criterion_positive_seeds'] >= 4)
            task_passes.append(row['passes'])
            comparisons.append(row)
        screens[candidate] = all(task_passes) and len(protocol['seeds']) == 5 and not protocol['smoke']
    pd.DataFrame(comparisons).to_csv(out / 'screens.csv', index=False)
    write_json(out / 'SCREENS.json', {'primary': 'joint', 'primary_passes': screens['joint'],
        'other_arms_exploratory': {k: v for k, v in screens.items() if k != 'joint'},
        'criterion': protocol['screen'], 'not_a_significance_test': True,
        'new_split_generalization_tested': False, 'literature_novelty_established': False})
    merged = frame.merge(pd.DataFrame(rows), on=['task', 'seed', 'arm'], validate='one_to_one')
    columns = ['val_auc', 'val_aupr', 'macro_drug_auc', 'macro_drug_aupr']
    summary = merged.groupby(['task', 'arm'])[columns].mean()
    summary.to_csv(out / 'diagnostic_summary.csv')
    lines = ['# 按药物验证与预设筛选', '',
        '以下宏平均仅纳入验证集中同时含正例和负例的药物；每个药物权重相同。它们是诊断指标，没有参与选模型。', '',
        '| 任务 | 方案 | Global AUC | Global AUPR | Macro-drug AUC | Macro-drug AUPR |',
        '|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        for arm in ARMS:
            r = summary.loc[(task, arm)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f | %.6f |' % (task, arm, *[r[c] for c in columns]))
    lines += ['', '## 度数对照', '']
    for r in degree_controls:
        lines.append('- %s：仅训练药物度数的验证 AUC/AUPR 为 %.6f/%.6f；纳入 %d 个药物、%d/%d 个验证样本，组内 AUC 为 %.6f。' % (
            r['task'], r['val_auc'], r['val_aupr'], r['eligible_drugs'], r['covered_pairs'], r['validation_pairs'], r['macro_drug_auc']))
    lines += ['', '## 预设筛选', '',
        '主候选 joint：' + ('通过' if screens['joint'] else '未通过') + '。',
        '其他组仅作探索性结果：`' + json.dumps({k: v for k, v in screens.items() if k != 'joint'}) + '`。',
        '', '详细配对差值在 paired_deltas.csv，逐项筛选在 screens.csv。对照通过不能单独确立方法新颖性或独立划分泛化。', '']
    (out / 'DIAGNOSTICS.md').write_text('\n'.join(lines))
    print(summary.to_string())
    print('SCREENS', screens)


if __name__ == '__main__':
    main()
