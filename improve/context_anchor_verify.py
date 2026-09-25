"""General (task-shared) anchor shrinkage: one recipe, same operators and weight for
MiDrug and LncDrug.  No task-specific branch, no conditional TabDPT, no per-task weight.

score = sigmoid( (1-w) * logit(reference) + w * mean_m logit(member_m) )

The accepted reference is the frozen TabICLv2 tiered_0p5000 composition.  Members are
from-scratch, non-foundation learners fitted on a frozen feature bank.  The same subset
and the same weight are used for both tasks.  Evaluation follows the project gates on
the historical inner partition and the three development resplits:
  inner   : MiDrug mean AUC/AP up with >= 2/3 seeds positive; LncDrug AUC/AP >= reference in every seed.
  resplit : MiDrug AUC/AP > reference in every split; LncDrug AUC/AP >= reference in every split.
A label-permutation control with the same operators must fail.

Run:
    /data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.context_anchor_verify
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import Nystroem

from improve import tabular_foundation as base
from improve import anchor_shrinkage as anchors


ROOT = Path(__file__).resolve().parent
INNER = ROOT / 'results/foundation_transfer_20260914/inputs'
INREF = ROOT / 'results/tabicl_pathway_context_20260913'
RB = ROOT / 'results/resplit_foundation_verify_20260914'
RREF = ROOT / 'results/transfer_splits_20260914'
OUT = ROOT / 'results/context_anchor_verify_20260915'
SPLITS = (42, 2026, 31415)
TASKS = ('MiDrug', 'LncDrug')
SEEDS = (666, 0, 1)
SEED = 20260914

BANKS = ('tiered62', 'context76')
SUBSETS = (
    ('hgb', 'et', 'mlp', 'lr'),
    ('hgb', 'et', 'lr'),
    ('hgb', 'et', 'rf', 'mlp', 'lr'),
    ('hgb', 'et', 'rf', 'mlp', 'lr', 'knn', 'nys', 'gbr'),
)
WEIGHTS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)


def learner(name, seed=SEED):
    table = {
        'hgb': lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=seed),
        'et': lambda: ExtraTreesClassifier(n_estimators=600, max_features='sqrt', n_jobs=8, random_state=seed),
        'rf': lambda: RandomForestClassifier(n_estimators=600, max_features='sqrt', n_jobs=8, random_state=seed),
        'mlp': lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=600,
                                                                    random_state=seed, early_stopping=True)),
        'lr': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
        'knn': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=40, weights='distance', n_jobs=8)),
        'nys': lambda: make_pipeline(StandardScaler(), Nystroem(gamma=0.1, n_components=300, random_state=seed),
                                     LogisticRegression(C=1.0, max_iter=2000)),
        'gbr': lambda: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=seed),
    }
    return table[name]()


def bank_arrays(bank, task, scope, split):
    if scope == 'inner':
        return np.load(INNER / ('%s_%s.npz' % (bank, task)), allow_pickle=False)
    return np.load(RB / ('inputs_s%d/%s_%s.npz' % (split, bank, task)), allow_pickle=False)


def fit_component(bank, task, scope, split, subset, shuffled, rng):
    a = bank_arrays(bank, task, scope, split)
    xtr, ytr, xq = a['x_train'], a['y_train'], a['x_val']
    yy = rng.permutation(ytr) if shuffled else ytr
    logits = []
    for name in subset:
        m = learner(name)
        m.fit(xtr, yy)
        logits.append(anchors.logit(m.predict_proba(xq)[:, 1]))
    return np.mean(logits, axis=0)


def inner_reference(task, seed):
    with np.load(INREF / ('predictions/inner/%s/s%d/scores.npz' % (task, seed)), allow_pickle=False) as z:
        return z['tiered_0p5000'].copy(), z['y_val'].copy(), z['pairs_val'].copy()


def resplit_reference(task, split):
    with np.load(RREF / ('runs/%s/%d/tiered/scores.npz' % (task, split)), allow_pickle=False) as z:
        return z['pred'].copy(), z['y'].copy(), z['pairs'].copy()


def run():
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    combos = [(b, s, False) for b in BANKS for s in SUBSETS] + [('tiered62', SUBSETS[0], True), ('context76', SUBSETS[0], True)]
    comps = {}
    for bank, subset, shuffled in combos:
        for task in TASKS:
            comps[('inner', bank, task, subset, shuffled)] = fit_component(bank, task, 'inner', None, subset, shuffled, rng)
            for split in SPLITS:
                comps[('resplit', split, bank, task, subset, shuffled)] = fit_component(bank, task, 'resplit', split, subset, shuffled, rng)
        print('component', bank, subset, 'shuffled=%s' % shuffled, flush=True)

    rows = []
    for bank, subset, shuffled in combos:
        for w in WEIGHTS:
            label = '%s|%s|w%.2f%s' % (bank, '+'.join(subset), w, '|shuffled' if shuffled else '')
            for task in TASKS:
                for seed in SEEDS:
                    ref, y, pairs = inner_reference(task, seed)
                    value = expit((1 - w) * anchors.logit(ref) + w * comps[('inner', bank, task, subset, shuffled)])
                    m, r = base.metrics(y, value, pairs), base.metrics(y, ref, pairs)
                    rows.append(dict(recipe=label, scope='inner', task=task, key='seed%d' % seed,
                                     d_auc=m['auc'] - r['auc'], d_ap=m['ap'] - r['ap'], auc=m['auc'], ap=m['ap']))
                for split in SPLITS:
                    ref, y, pairs = resplit_reference(task, split)
                    value = expit((1 - w) * anchors.logit(ref) + w * comps[('resplit', split, bank, task, subset, shuffled)])
                    m, r = base.metrics(y, value, pairs), base.metrics(y, ref, pairs)
                    rows.append(dict(recipe=label, scope='resplit', task=task, key='split%d' % split,
                                     d_auc=m['auc'] - r['auc'], d_ap=m['ap'] - r['ap'], auc=m['auc'], ap=m['ap']))
    t = pd.DataFrame(rows)
    t.to_csv(out / 'metrics.csv', index=False)

    summary = []
    for label, sub in t.groupby('recipe'):
        g = {}
        for task in TASKS:
            it = sub[(sub.scope == 'inner') & (sub.task == task)]
            rt = sub[(sub.scope == 'resplit') & (sub.task == task)]
            inner_ok = (bool((it.d_auc.values >= 0).all() and (it.d_ap.values >= 0).all()) if task == 'LncDrug'
                        else bool(it.d_auc.mean() > 0 and it.d_ap.mean() > 0 and (it.d_auc > 0).sum() >= 2 and (it.d_ap > 0).sum() >= 2))
            res_ok = (bool((rt.d_auc.values > 0).all() and (rt.d_ap.values > 0).all()) if task == 'MiDrug'
                      else bool((rt.d_auc.values >= 0).all() and (rt.d_ap.values >= 0).all()))
            g[task] = dict(inner_ok=inner_ok, resplit_ok=res_ok,
                           inner_d_auc=float(it.d_auc.mean()), inner_d_ap=float(it.d_ap.mean()),
                           resplit_d_auc=float(rt.d_auc.mean()), resplit_d_ap=float(rt.d_ap.mean()))
        summary.append(dict(recipe=label, shared=True,
                            inner_Mi_dAUC=g['MiDrug']['inner_d_auc'], inner_Mi_dAP=g['MiDrug']['inner_d_ap'],
                            inner_Ln_dAUC=g['LncDrug']['inner_d_auc'], inner_Ln_dAP=g['LncDrug']['inner_d_ap'],
                            resplit_Mi_dAUC=g['MiDrug']['resplit_d_auc'], resplit_Mi_dAP=g['MiDrug']['resplit_d_ap'],
                            resplit_Ln_dAUC=g['LncDrug']['resplit_d_auc'], resplit_Ln_dAP=g['LncDrug']['resplit_d_ap'],
                            joint=bool(all(g[k][p] for k in TASKS for p in ('inner_ok', 'resplit_ok')))))
    s = pd.DataFrame(summary).sort_values(['joint', 'inner_Mi_dAP'], ascending=False)
    s.to_csv(out / 'summary.csv', index=False)
    best = s[s.joint].head(3).to_dict('records')
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  reference='accepted tiered_0p5000 (MiDrug 0.962593/0.927204, LncDrug 0.993750/0.987141 inner)',
                  model_class='general/shared: one subset and one weight for both tasks; no task-specific branch',
                  composition='sigmoid((1-w)*logit(reference) + w*mean_m logit(member_m))',
                  banks=list(BANKS), subsets=[list(x) for x in SUBSETS], weights=list(WEIGHTS),
                  gate_inner='MiDrug mean AUC/AP up with >=2/3 seeds positive; LncDrug AUC/AP >= reference in every seed.',
                  gate_resplit='MiDrug AUC/AP > reference in every split; LncDrug AUC/AP >= reference in every split.',
                  n_joint=int(s.joint.sum()), n_recipes=int(len(s)), best_joint=best,
                  no_task_specific_branch=True, no_outer_or_test_access=True, development_surfaces_reused=True,
                  limitations=['Inner and resplits are repeatedly used development surfaces.',
                               'Selection over many shared recipes has selection bias; the shuffled control is the information check.'])
    base.write_json(out / 'ASSESSMENT.json', result)
    print(s.head(12).to_string(index=False))
    print('joint recipes:', int(s.joint.sum()), '/', len(s))


if __name__ == '__main__':
    run()
