# Select a configuration from the hpsearch trace, by the author's own rule.
#
# `find_best_params.py` maximises (val AUC + val AUPR)/2 over the sweep on a
# single seed, reading the last epoch of each run.  That rule is reproduced
# here as `author`, so the selected cell is comparable to the paper's, and
# alongside it the rule this repo will actually quote:
#
#   author  -- seed 666, value at epoch checkpoint E, argmax of the mean of the
#              two validation metrics.  Exactly find_best_params.py.
#   robust  -- mean over seeds of the best-validation-epoch value, argmax of
#              the same criterion.  Averaging over seeds is what stops the
#              argmax from being a seed draw; the released rule does not do it.
#
# Both are selections ON the validation fold, which is the fold the paper's
# Table 2 reports -- so the resulting S_val figure is optimistic in exactly the
# same way the paper's is, and the S_test column is the honest one.
import argparse

import pandas as pd

EPOCH_CKPTS = [51, 101, 151, 201, 251, 301]
BASE_KEYS = ['arm', 'lr', 'channels', 'aggr']
# Axes added after the author's grid.  CSVs written before they existed have no
# such column; filling them with the released defaults is what makes an old and
# a new sweep poolable -- without it, groupby drops every row with a NaN key and
# the older sweeps vanish silently.
EXTRA_DEFAULTS = dict(layers=2, dropout=0.0, weight_decay=0.0,
                      optimizer='rmsprop')
KEYS = list(BASE_KEYS)


def set_keys(df):
    """Fill the post-hoc axes with their released defaults, in place, and set
    the grouping keys to whatever the pooled frame actually carries."""
    global KEYS
    for c, v in EXTRA_DEFAULTS.items():
        if c not in df.columns:
            df[c] = v
        else:
            df[c] = df[c].fillna(v)
    KEYS = BASE_KEYS + list(EXTRA_DEFAULTS)
    return KEYS


def author_rule(df, seed=666):
    """Long-form: one row per (config, epoch checkpoint) at a single seed."""
    d = df[df['seed'] == seed]
    out = []
    for e in EPOCH_CKPTS:
        ca, cp = f'ep{e}_val_auc', f'ep{e}_val_aupr'
        if ca not in d.columns:
            continue
        sub = d[KEYS + [ca, cp, f'ep{e}_test_auc', f'ep{e}_test_aupr']].copy()
        sub.columns = KEYS + ['val_auc', 'val_aupr', 'test_auc', 'test_aupr']
        sub['epochs'] = e
        out.append(sub)
    r = pd.concat(out, ignore_index=True).dropna(subset=['val_auc'])
    r['criterion'] = (r['val_auc'] + r['val_aupr']) / 2
    return r.sort_values('criterion', ascending=False)


def robust_rule(df):
    """Seed-averaged, best-validation-epoch."""
    cols = ['bv_val_auc', 'bv_val_aupr', 'bv_test_auc', 'bv_test_aupr',
            'bv_epoch', 'bp_epoch']
    g = df.dropna(subset=['bv_val_auc']).groupby(KEYS)[cols]
    r = g.mean().reset_index()
    r['n_seeds'] = g.size().values
    r = r.rename(columns={'bv_val_auc': 'val_auc', 'bv_val_aupr': 'val_aupr',
                          'bv_test_auc': 'test_auc',
                          'bv_test_aupr': 'test_aupr'})
    r['criterion'] = (r['val_auc'] + r['val_aupr']) / 2
    return r.sort_values('criterion', ascending=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csvs', nargs='+')
    p.add_argument('--top', type=int, default=12)
    p.add_argument('--paper', nargs=2, type=float, default=None,
                   metavar=('AUC', 'AUPR'))
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True)
    set_keys(df)
    task = df['task'].iloc[0]
    print(f'{task}: {len(df)} runs, '
          f"{df.groupby(KEYS).ngroups} configs, seeds {sorted(df['seed'].unique())}")

    fmt = lambda v: f'{v:.4f}'
    for name, r in (('author (seed 666, epoch checkpoint)', author_rule(df)),
                    ('robust (seed-averaged, best-val epoch)', robust_rule(df))):
        print(f'\n=== {name} ===')
        show = [c for c in ['arm', 'lr', 'channels', 'aggr', 'layers',
                            'dropout', 'weight_decay', 'optimizer', 'epochs',
                            'n_seeds', 'val_auc', 'val_aupr', 'test_auc',
                            'test_aupr', 'criterion'] if c in r.columns]
        top = r.head(args.top)[show].copy()
        # `lr` is a float column and would be rounded to 4dp by float_format,
        # which collapses 5e-05 and 1e-04 onto the same printed value.
        top['lr'] = top['lr'].map(lambda v: f'{v:g}')
        print(top.to_string(index=False, float_format=fmt))
        if args.paper:
            pa, pp = args.paper
            ok = r[(r['val_auc'] > pa) & (r['val_aupr'] > pp)]
            print(f'  configs clearing paper {pa:.4f}/{pp:.4f} on BOTH: '
                  f'{len(ok)}/{len(r)}')


if __name__ == '__main__':
    main()
