"""Within-drug diagnostics and predefined comparisons for context MIL."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from improve.analyze_expression_consensus import within_drug
from improve.cell_context_mil import ARMS, CONTRASTS, TASKS, checked_result, load_config
from improve.first_round_features import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    protocol = load_config(out)
    validation = json.loads((out / 'VALIDATION.json').read_text())
    if not validation['passed'] or validation['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Run report verification first')
    frame = pd.read_csv(out / 'results.csv')
    indexed = frame.set_index(['task', 'arm', 'seed'])
    rows, contrasts = [], []
    for task in TASKS:
        for seed in protocol['seeds']:
            for arm in ARMS:
                directory = out / 'runs' / task / ('s' + str(seed)) / arm
                checked_result(directory, protocol)
                with np.load(directory / 'scores.npz') as z:
                    row = within_drug(z['y_val'], z['pairs_val'], z['val'])
                rows.append(dict(task=task, seed=seed, arm=arm, **row))
        for candidate, control in CONTRASTS:
            row = dict(task=task, candidate=candidate, control=control)
            for metric in ('val_auc', 'val_aupr', 'val_criterion'):
                delta = np.array([indexed.loc[(task, candidate, s), metric] - indexed.loc[(task, control, s), metric] for s in protocol['seeds']])
                row[metric + '_delta'] = float(delta.mean())
                row[metric + '_positive_seeds'] = int((delta > 0).sum())
            row['passes'] = bool(row['val_auc_delta'] > 0 and row['val_aupr_delta'] > 0
                and row['val_criterion_positive_seeds'] >= 4 and len(protocol['seeds']) == 5
                and not protocol['smoke'])
            contrasts.append(row)
    pd.DataFrame(rows).to_csv(out / 'within_drug.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out / 'screens.csv', index=False)
    primary = all(row['passes'] for row in contrasts if row['candidate'] == 'context_softor' and row['control'] == 'pca')
    mechanism = all(row['passes'] for row in contrasts if row['candidate'] == 'context_softor' and row['control'] in ('context_mean', 'random_softor'))
    write_json(out / 'SCREENS.json', {'primary': 'context_softor', 'primary_passes': primary,
        'mean_and_random_controls_pass': mechanism, 'criterion': protocol['screen'],
        'adaptive_validation_exploration': True, 'new_split_generalization_tested': False,
        'literature_novelty_established': False})
    merged = frame.merge(pd.DataFrame(rows), on=['task', 'arm', 'seed'], validate='one_to_one')
    cols = ['val_auc', 'val_aupr', 'macro_drug_auc', 'macro_drug_aupr']
    summary = merged.groupby(['task', 'arm'])[cols].mean()
    summary.to_csv(out / 'diagnostic_summary.csv')
    lines = ['# 多实例实验：组内排名与机制对照', '',
        '宏平均仅纳入验证集中同时存在正、负例的药物；没有参与选模。', '',
        '| 任务 | 方案 | Global AUC | Global AUPR | Macro-drug AUC | Macro-drug AUPR |',
        '|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        for arm in ARMS:
            row = summary.loc[(task, arm)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f | %.6f |' % (task, arm, *[row[c] for c in cols]))
    lines += ['', '主候选相对 PCA：' + ('通过' if primary else '未通过') + '双任务筛选。',
              '相对均值池化和随机分组：' + ('通过' if mechanism else '未通过') + '机制对照。',
              '', '详细差值见 paired_deltas.csv，逐项条件见 screens.csv。不能从全局 AUC/AUPR 单独推导癌种特异性或生物学机制结论。', '']
    (out / 'DIAGNOSTICS.md').write_text('\n'.join(lines))
    print(summary.to_string())
    print('SCREENS primary', primary, 'mechanism', mechanism)


if __name__ == '__main__':
    main()
