# Paired arm-vs-base comparison on the released protocol.
#
# Every arm shares its seed list with `base`, and the split is frozen, so the
# comparison is paired seed-by-seed: report the mean paired delta, how many
# seeds moved in its favour, and a paired t-test / Wilcoxon.  The published
# margin over the best baseline is +0.0128 AUC, so an unpaired mean with no
# spread cannot support a claim at this scale.
import argparse

import pandas as pd
from scipy import stats


def compare(df, metric, base='base'):
    rows = []
    for task, g in df.groupby('task'):
        b = g[g['arm'] == base].set_index('seed')[metric]
        for arm, ga in g.groupby('arm'):
            a = ga.set_index('seed')[metric]
            common = b.index.intersection(a.index)
            d = (a[common] - b[common]).values
            row = dict(task=task, arm=arm, n=len(common),
                       mean=a[common].mean(), std=a[common].std(),
                       delta=d.mean(), wins=int((d > 0).sum()))
            if arm != base and len(common) > 1 and d.std() > 0:
                row['t_p'] = stats.ttest_rel(a[common], b[common]).pvalue
                row['wilcoxon_p'] = stats.wilcoxon(d).pvalue
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csvs', nargs='+')
    p.add_argument('--metrics', nargs='+',
                   default=['sel_val_auc', 'best_test_auc', 'sel_val_aupr'])
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True)
    for m in args.metrics:
        print(f'\n=== {m} (paired vs base) ===')
        out = compare(df, m)
        print(out.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
