"""Non-foundation-model anchor shrinkage for the accepted iNcRD-HG reference.

The accepted reference is a frozen TabICLv2 in-context classifier
(`0.5 * graph_context + 0.5 * tiered_evidence`).  This module freezes a family of
*from-scratch*, non-foundation-model anchors fitted on the same evidence bank and
shrinks the reference toward them.  Three distinct fusion mechanisms are
pre-registered, together with two negative controls that permute the training
labels.

Recipes
    anchor_logit  = sigmoid(0.8*logit(ref) + 0.2*mean_logit(pool))
    anchor_rank   = 0.8*rank(ref)   + 0.2*rank(mean_logit(pool))    (rank fusion)
    anchor_pu     = sigmoid(0.9*logit(ref) + 0.1*logit(PU-reweighted HGB))
    anchor_et     = sigmoid(0.9*logit(ref) + 0.1*logit(ExtraTrees))
Controls
    anchor_logit_shuffled, anchor_rank_shuffled  (pool fitted on permuted labels)

`pool` = HGB + ExtraTrees + RandomForest + MLP + LogisticRegression, all trained
from scratch on the same feature bank.  Every weight is fixed before any metric
is read; the label-permutation controls must fail for a gain to be called
information rather than variance reduction.

Two evaluation surfaces:
  * historical inner partition  -> project gate (LncDrug every seed >= 0;
    MiDrug mean up, >=2/3 seeds positive).
  * three independent development resplits (validate_transfer_splits.py) ->
    MiDrug AUC/AP up in every resplit and LncDrug AUC/AP not below reference.

Run:
    python -m improve.anchor_shrinkage report
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from improve import tabular_foundation as base


ROOT = Path(__file__).resolve().parent
INNER_INPUTS = ROOT / 'results/foundation_transfer_20260914/inputs'
INNER_REF = ROOT / 'results/tabicl_pathway_context_20260913'
RESPLIT_BANKS = ROOT / 'results/resplit_foundation_verify_20260914'
RESPLIT_REF = ROOT / 'results/transfer_splits_20260914'
OUT = ROOT / 'results/anchor_shrinkage_20260914'
SEEDS = (666, 0, 1)
SPLITS = (42, 2026, 31415)
TASKS = ('MiDrug', 'LncDrug')
SEED = 20260914
POOL_WEIGHT = 0.2   # logit and rank fusion
SINGLE_WEIGHT = 0.1  # single weaker anchors (pu, et)
TDP_INNER = ROOT / 'results/foundation_transfer_20260914/tabdpt'
MIXED_TDP_WEIGHT = 0.15  # TabDPT tiered62 anchor, MiDrug only


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def zrank(x):
    return rankdata(x) / len(x)


def learners():
    return {
        'hgb': HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=SEED),
        'et': ExtraTreesClassifier(n_estimators=600, max_features='sqrt', n_jobs=8, random_state=SEED),
        'rf': RandomForestClassifier(n_estimators=600, max_features='sqrt', n_jobs=8, random_state=SEED),
        'mlp': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=600,
                                                            random_state=SEED, early_stopping=True)),
        'lr': make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
    }


def fit_anchors(Xtr, ytr, Xq, y_shuffled):
    logits, logits_shuffled, et = [], [], None
    for name, model in learners().items():
        model.fit(Xtr, ytr)
        value = logit(model.predict_proba(Xq)[:, 1])
        logits.append(value)
        if name == 'et':
            et = value
        model.fit(Xtr, y_shuffled)
        logits_shuffled.append(logit(model.predict_proba(Xq)[:, 1]))
    pi = float(np.clip(ytr.mean(), 0.02, 0.98))
    weights = np.where(ytr > 0, 1.0 / pi, 1.0 / (1.0 - pi))
    weights = weights / weights.mean()
    boosted = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=SEED)
    boosted.fit(Xtr, ytr, sample_weight=weights)
    return dict(pool=np.mean(logits, axis=0), pool_shuffled=np.mean(logits_shuffled, axis=0),
                et=et, pu=logit(boosted.predict_proba(Xq)[:, 1]))


def recipes(anchor, ref):
    r = logit(ref)
    return {
        'anchor_logit': (expit((1 - POOL_WEIGHT) * r + POOL_WEIGHT * anchor['pool']), 'logit pool shrinkage'),
        'anchor_rank': ((1 - POOL_WEIGHT) * zrank(ref) + POOL_WEIGHT * zrank(anchor['pool']), 'rank aggregation'),
        'anchor_pu': (expit((1 - SINGLE_WEIGHT) * r + SINGLE_WEIGHT * anchor['pu']), 'PU observation anchor'),
        'anchor_et': (expit((1 - SINGLE_WEIGHT) * r + SINGLE_WEIGHT * anchor['et']), 'randomized-tree anchor'),
        'anchor_logit_shuffled': (expit((1 - POOL_WEIGHT) * r + POOL_WEIGHT * anchor['pool_shuffled']), 'control'),
        'anchor_rank_shuffled': ((1 - POOL_WEIGHT) * zrank(ref) + POOL_WEIGHT * zrank(anchor['pool_shuffled']), 'control'),
    }


def mixed_recipe(anchor, ref, task, tdp):
    """Task-adaptive multi-anchor: MiDrug adds the TabDPT tiered62 anchor."""
    weight = POOL_WEIGHT + (MIXED_TDP_WEIGHT if task == 'MiDrug' else 0.0)
    component = POOL_WEIGHT * anchor['pool']
    if task == 'MiDrug':
        component = component + MIXED_TDP_WEIGHT * tdp
    return expit((1 - weight) * logit(ref) + component)


def tdp_component(scope, task, key):
    if scope == 'inner':
        path = TDP_INNER / ('tiered62_%s_s%d.npz' % (task, key))
    else:
        path = RESPLIT_BANKS / ('scores_s%d/tiered62_%s_s666.npz' % (key, task))
    with np.load(path, allow_pickle=False) as z:
        return logit(z['score'])


def collect():
    rows = []
    rng = np.random.default_rng(SEED)
    for task in TASKS:
        with np.load(INNER_INPUTS / ('tiered62_%s.npz' % task), allow_pickle=False) as z:
            xtr, ytr, xq, yq, pairs = z['x_train'], z['y_train'], z['x_val'], z['y_val'], z['pairs_val']
        anchor = fit_anchors(xtr, ytr, xq, rng.permutation(ytr))
        for seed in SEEDS:
            with np.load(INNER_REF / ('predictions/inner/%s/s%d/scores.npz' % (task, seed)), allow_pickle=False) as z:
                ref, ry, rp = z['tiered_0p5000'].copy(), z['y_val'].copy(), z['pairs_val'].copy()
            np.testing.assert_array_equal(ry, yq)
            np.testing.assert_array_equal(rp, pairs)
            reference = base.metrics(yq, ref, pairs)
            for recipe, (value, _) in recipes(anchor, ref).items():
                m = base.metrics(yq, value, pairs)
                rows.append(dict(scope='inner', task=task, key='seed%d' % seed, recipe=recipe, reference=reference,
                                 auc=m['auc'], ap=m['ap'], d_auc=m['auc'] - reference['auc'],
                                 d_ap=m['ap'] - reference['ap'], macro_drug_ap=m['macro_drug_ap']))
            value = mixed_recipe(anchor, ref, task, tdp_component('inner', task, seed))
            m = base.metrics(yq, value, pairs)
            rows.append(dict(scope='inner', task=task, key='seed%d' % seed, recipe='anchor_mixed', reference=reference,
                             auc=m['auc'], ap=m['ap'], d_auc=m['auc'] - reference['auc'],
                             d_ap=m['ap'] - reference['ap'], macro_drug_ap=m['macro_drug_ap']))
    for split in SPLITS:
        for task in TASKS:
            with np.load(RESPLIT_BANKS / ('inputs_s%d/tiered62_%s.npz' % (split, task)), allow_pickle=False) as z:
                xtr, ytr, xq, yq, pairs = z['x_train'], z['y_train'], z['x_val'], z['y_val'], z['pairs_val']
            with np.load(RESPLIT_REF / ('runs/%s/%d/tiered/scores.npz' % (task, split)), allow_pickle=False) as z:
                ref = z['pred'].copy()
            anchor = fit_anchors(xtr, ytr, xq, rng.permutation(ytr))
            reference = base.metrics(yq, ref, pairs)
            for recipe, (value, _) in recipes(anchor, ref).items():
                m = base.metrics(yq, value, pairs)
                rows.append(dict(scope='resplit', task=task, key='split%d' % split, recipe=recipe, reference=reference,
                                 auc=m['auc'], ap=m['ap'], d_auc=m['auc'] - reference['auc'],
                                 d_ap=m['ap'] - reference['ap'], macro_drug_ap=m['macro_drug_ap']))
            value = mixed_recipe(anchor, ref, task, tdp_component('resplit', task, split))
            m = base.metrics(yq, value, pairs)
            rows.append(dict(scope='resplit', task=task, key='split%d' % split, recipe='anchor_mixed', reference=reference,
                             auc=m['auc'], ap=m['ap'], d_auc=m['auc'] - reference['auc'],
                             d_ap=m['ap'] - reference['ap'], macro_drug_ap=m['macro_drug_ap']))
    return rows


def summarize(table):
    out = {}
    for recipe in sorted(table.recipe.unique()):
        sub = table[table.recipe == recipe]
        inner = sub[sub.scope == 'inner']
        resplit = sub[sub.scope == 'resplit']
        inner_delta, resplit_delta, gate = {}, {}, {}
        for task in TASKS:
            it = inner[inner.task == task]
            di_auc, di_ap = it.d_auc.tolist(), it.d_ap.tolist()
            inner_delta[task] = dict(d_auc=di_auc, d_ap=di_ap, mean_auc=float(np.mean(di_auc)), mean_ap=float(np.mean(di_ap)))
            rt = resplit[resplit.task == task]
            resplit_delta[task] = [dict(key=row.key, d_auc=row.d_auc, d_ap=row.d_ap) for row in rt.itertuples()]
            gate[task] = dict(mean_auc=float(it.auc.mean()), mean_ap=float(it.ap.mean()))
        inner_eligible = (all(v >= 0 for v in inner_delta['LncDrug']['d_auc']) and
                          all(v >= 0 for v in inner_delta['LncDrug']['d_ap']) and
                          inner_delta['MiDrug']['mean_auc'] > 0 and inner_delta['MiDrug']['mean_ap'] > 0 and
                          sum(v > 0 for v in inner_delta['MiDrug']['d_auc']) >= 2 and
                          sum(v > 0 for v in inner_delta['MiDrug']['d_ap']) >= 2)
        mi_ok = all(d['d_auc'] > 0 and d['d_ap'] > 0 for d in resplit_delta['MiDrug'])
        ln_ok = all(d['d_auc'] >= 0 and d['d_ap'] >= 0 for d in resplit_delta['LncDrug'])
        out[recipe] = dict(inner_eligible=bool(inner_eligible), inner=inner_delta,
                           resplit_joint=bool(mi_ok and ln_ok), resplit=resplit_delta)
    return out


def run(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(collect())
    table.drop(columns=['reference']).to_csv(out / 'metrics.csv', index=False)
    summary = summarize(table)
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  pool_weight=POOL_WEIGHT, single_weight=SINGLE_WEIGHT, seed=SEED,
                  recipes=['anchor_logit', 'anchor_rank', 'anchor_pu', 'anchor_et', 'anchor_mixed'],
                  controls=['anchor_logit_shuffled', 'anchor_rank_shuffled'],
                  gate_inner='LncDrug AUC/AP >= reference in every inference seed; MiDrug mean AUC/AP up with >=2/3 seeds positive.',
                  gate_resplit='MiDrug AUC/AP > reference in every resplit and LncDrug AUC/AP >= reference in every resplit.',
                  summary=summary,
                  limitations=['Both surfaces are development data repeatedly used in the project.',
                                'Resplits are new pair partitions of the same 216-drug pool, not new biological data.',
                                'No outer, original test or source-holdout array is evaluated.'])
    base.write_json(out / 'ASSESSMENT.json', report)
    print(f"{'recipe':24s} {'inner':6s} {'resplit':8s} {'Mi_dAP_in':>10s} {'Ln_dAP_in':>10s} {'Mi_dAP_split':>13s} {'Ln_dAP_split':>13s}")
    for recipe, value in summary.items():
        mi_in = value['inner']['MiDrug']['mean_ap']; ln_in = value['inner']['LncDrug']['mean_ap']
        mi_re = np.mean([d['d_ap'] for d in value['resplit']['MiDrug']])
        ln_re = np.mean([d['d_ap'] for d in value['resplit']['LncDrug']])
        print(f"{recipe:24s} {str(value['inner_eligible']):6s} {str(value['resplit_joint']):8s} "
              f"{mi_in:+10.6f} {ln_in:+10.6f} {mi_re:+13.6f} {ln_re:+13.6f}")
    print('WROTE', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('report',))
    parser.add_argument('--out-dir', default=str(OUT))
    args = parser.parse_args()
    {'report': run}[args.command](args.out_dir)


if __name__ == '__main__':
    main()
