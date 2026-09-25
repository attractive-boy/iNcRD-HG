"""Verify the accepted TabDPT transfer technique on the three independent resplits.

The accepted improvement (`improve/FOUNDATION_TRANSFER.md`) was only measured on
the repeatedly used historical inner partition.  This module re-builds the same
`tiered62` / `context76` feature banks for the three new development resplits
created by `validate_transfer_splits.py`, runs the frozen TabDPT (1.3) classifier
on them with the existing `foundation_transfer_runner.py`, and compares against
the resplits' accepted `tiered` reference.

Commands
    prepare   build sealed resplit feature banks (numpy/tabicl env)
    eval      score TabDPT candidates vs the resplit reference
Inference is run separately with the frontier environment:
    FRONTIER improve/foundation_transfer_runner.py --inputs-dir <split inputs> ...
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from improve import tabular_foundation as base
from improve import tabicl_context_program_modules as feature


ROOT = Path(__file__).resolve().parent
PATHWAY_INPUTS = ROOT / 'results/tabicl_pathway_context_inputs_20260913'
RESPLITS = ROOT / 'results/transfer_splits_20260914'
OUT = ROOT / 'results/resplit_foundation_verify_20260914'
TASKS = ('MiDrug', 'LncDrug')
SPLITS = (42, 2026, 31415)
SEEDS = (666, 0, 1)
BANKS = ('base36', 'tiered62', 'context76')
FRONTIER = '/data1/licheng/incrd-hg/.venv-frontier/bin/python'
TABICL = '/data1/licheng/incrd-hg/.venv-tabicl/bin/python'
LOGIT_EPS = 1e-12


def logit(p):
    p = np.clip(np.asarray(p, float), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p) - np.log1p(-p)


def split_inputs_dir(out, split):
    return Path(out) / ('inputs_s%d' % split)


def split_scores_dir(out, split):
    return Path(out) / ('scores_s%d' % split)


def prepare(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    encoder = feature.ContextPrograms.load(PATHWAY_INPUTS)
    base_view = {'MiDrug': 'raw', 'LncDrug': 'x'}
    for split in SPLITS:
        dest = split_inputs_dir(out, split)
        dest.mkdir(parents=True)
        files = {}
        for task in TASKS:
            with np.load(RESPLITS / ('inputs/%s/%d/arrays.npz' % (task, split)), allow_pickle=False) as z:
                prefix = base_view[task]
                gtr, gval = z[prefix + '_train'], z[prefix + '_val']
                ytr, yval = z['y_train'], z['y_val']
                ptr, pval = z['pairs_train'], z['pairs_val']
            for bank, variant in (('tiered62', 'tiered'), ('context76', 'context')):
                etr = encoder.transform(task, ptr, variant).astype(np.float32)
                eval_ = encoder.transform(task, pval, variant).astype(np.float32)
                xtr = np.concatenate([gtr, etr], axis=1)
                xval = np.concatenate([gval, eval_], axis=1)
                if not np.isfinite(xtr).all() or not np.isfinite(xval).all():
                    raise ValueError('Non-finite resplit features')
                name = '%s_%s.npz' % (bank, task)
                np.savez_compressed(dest / name, x_train=xtr, x_val=xval, y_train=ytr, y_val=yval,
                                    pairs_train=ptr, pairs_val=pval)
                files[name] = base.digest(dest / name)
                print('BUILT split', split, name, xtr.shape, xval.shape, flush=True)
        doc = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                             split=split, tasks=list(TASKS), banks=['tiered62', 'context76'],
                             pathway_inputs=str(PATHWAY_INPUTS),
                             pathway_inputs_sha256=base.digest(PATHWAY_INPUTS / 'INPUTS.json'),
                             files=files,
                             note='Same frozen encoders and bank construction as foundation_transfer_features.'))
        base.write_json(dest / 'INPUTS.json', doc)
    print('PREPARED', out, flush=True)


def reference(task, split):
    with np.load(RESPLITS / ('runs/%s/%d/tiered/scores.npz' % (task, split)), allow_pickle=False) as z:
        return z['pred'].copy(), z['y'].copy(), z['pairs'].copy()


def bank_scores(out, split, bank, task, seed):
    path = split_scores_dir(out, split) / ('%s_%s_s%d.npz' % (bank, task, seed))
    with np.load(path, allow_pickle=False) as z:
        return z['score'].copy()


def evaluate(out):
    out = Path(out)
    rows, per_cell = [], []
    for split in SPLITS:
        for task in TASKS:
            ref, y, pairs = reference(task, split)
            for bank in BANKS:
                s = {seed: bank_scores(out, split, bank, task, seed) for seed in SEEDS}
                for seed in SEEDS:
                    m = base.metrics(y, s[seed], pairs)
                    rows.append(dict(split=split, task=task, recipe='tdp_' + bank, seed=seed,
                                     auc=m['auc'], ap=m['ap'], macro_drug_ap=m['macro_drug_ap']))
            # equal-logit ensemble of reference with the two strongest banks
            for name, sources in (('ensemble_ref_tiered62', ('tiered62',)),
                                  ('ensemble_ref_context76', ('context76',)),
                                  ('ensemble_ref_tiered62_context76', ('tiered62', 'context76'))):
                for seed in SEEDS:
                    comps = [logit(ref)] + [logit(bank_scores(out, split, b, task, seed)) for b in sources]
                    v = expit(np.mean(comps, axis=0))
                    m = base.metrics(y, v, pairs)
                    rows.append(dict(split=split, task=task, recipe=name, seed=seed,
                                     auc=m['auc'], ap=m['ap'], macro_drug_ap=m['macro_drug_ap']))
    table = pd.DataFrame(rows)
    for (split, task), g in table.groupby(['split', 'task']):
        ref, y, pairs = reference(task, split)
        rm = base.metrics(y, ref, pairs)
        for recipe in sorted(g.recipe.unique()):
            sub = g[g.recipe == recipe]
            per_cell.append(dict(split=split, task=task, recipe=recipe,
                                 d_auc=sub.auc.mean() - rm['auc'], d_ap=sub.ap.mean() - rm['ap'],
                                 ref_auc=rm['auc'], ref_ap=rm['ap'],
                                 mean_auc=sub.auc.mean(), mean_ap=sub.ap.mean()))
    cells = pd.DataFrame(per_cell)
    table.to_csv(out / 'metrics.csv', index=False)
    cells.to_csv(out / 'deltas.csv', index=False)
    summary = {}
    for recipe in sorted(cells.recipe.unique()):
        sub = cells[cells.recipe == recipe]
        mi = sub[sub.task == 'MiDrug']
        ln = sub[sub.task == 'LncDrug']
        summary[recipe] = dict(
            mi_mean_dauc=float(mi.d_auc.mean()), mi_mean_dap=float(mi.d_ap.mean()),
            ln_mean_dauc=float(ln.d_auc.mean()), ln_mean_dap=float(ln.d_ap.mean()),
            mi_positive_splits=int((mi.d_auc > 0).sum()), mi_positive_splits_ap=int((mi.d_ap > 0).sum()),
            ln_nonneg_splits=int((ln.d_auc >= 0).sum()), ln_nonneg_splits_ap=int((ln.d_ap >= 0).sum()),
            joint_pass=bool((mi.d_auc > 0).all() and (mi.d_ap > 0).all() and
                            (ln.d_auc >= 0).all() and (ln.d_ap >= 0).all()))
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  seeds=list(SEEDS), splits=list(SPLITS), reference='resplit accepted `tiered` arm',
                  gate='MiDrug AUC/AP up in every resplit and LncDrug AUC/AP not below reference in every resplit.',
                  summary=summary)
    base.write_json(out / 'ASSESSMENT.json', report)
    pd.set_option('display.width', 220)
    print(cells.to_string(index=False))
    print(json.dumps(summary, indent=1))
    print('WROTE', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'eval'))
    parser.add_argument('--out-dir', default=str(OUT))
    args = parser.parse_args()
    {'prepare': prepare, 'eval': evaluate}[args.command](args.out_dir)


if __name__ == '__main__':
    main()
