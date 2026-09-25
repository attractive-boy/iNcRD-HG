"""Export the frozen selection and audit numerical sensitivity of its gains.

No model fitting, recipe choice or new inference occurs here. The deliberately
adverse score perturbation is a post-hoc numerical diagnostic, not statistical
uncertainty and never a replacement for the actual prediction metrics.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from improve import tabular_foundation as base
from improve import tabicl_graph_context as experiment
from improve import tabicl_graph_context_audit as audit


def scores(path, label, pairs, key):
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['y_val'], label)
        np.testing.assert_array_equal(archive['pairs_val'], pairs)
        return archive[key].copy()


def rank_metrics(label, prediction):
    return dict(auc=float(roc_auc_score(label, prediction)),
                ap=float(average_precision_score(label, prediction)))


def analyze(out):
    p = experiment.protocol(out)
    s = experiment.selection(out, p)
    ap = audit.audit_protocol(out, p)
    verified = base.read_json(out/'VALIDATION.json')
    if p['smoke'] or not verified['passed'] or not verified['all_estimators_replayed']:
        raise ValueError('Need a fully verified formal experiment')
    if verified.get('numerical_audit_protocol_sha256') != ap['sha256']:
        raise ValueError('Missing amended verification')
    report = pd.read_csv(out/'outer_results.csv')
    outcome = base.read_json(out/'OUTCOME.json')
    exported, sensitivity, actual_deltas = [], [], []
    # The declared cross-shape allclose envelope is at most atol + rtol for
    # probabilities in [0,1]. Perturb the candidate adversely and the baseline
    # favorably, without refitting or changing the stored actual scores.
    epsilon = float(p['query_invariance_atol']+p['query_invariance_rtol'])
    for task in p['tasks']:
        recipe = s['chosen'][task]['recipe']
        for seed in p['seeds']:
            original = Path(p['parent_dir'])/p['baseline']['outer'][task][str(seed)]/'scores.npz'
            with np.load(original, allow_pickle=False) as archive:
                if set(archive.files) != {'val', 'y_val', 'pairs_val'}:
                    raise ValueError('Unexpected original arrays')
                baseline, labels, pairs = archive['val'].copy(), archive['y_val'].copy(), archive['pairs_val'].copy()
            if set(np.unique(labels)) != {0, 1}:
                raise ValueError('Nonbinary validation labels')
            full_path = experiment.directory(out, 'outer', task, seed, 'full')/'scores.npz'
            full_views = scores(full_path, labels, pairs, 'views')
            if full_views.shape != (6, len(labels)):
                raise ValueError('Wrong full-context view order/shape')
            per_fold = [scores(experiment.directory(out, 'outer', task, seed, 'fold%d' % fold)/'scores.npz',
                labels, pairs, 'views') for fold in range(5)]
            if any(v.shape != (2, len(labels)) for v in per_fold):
                raise ValueError('Wrong fold-context view order/shape')
            components = dict(query_bag=np.mean(full_views[1:], axis=0),
                              fold_full=np.mean([v[0] for v in per_fold], axis=0),
                              fold_match=np.mean([v[1] for v in per_fold], axis=0))
            selected = baseline.copy() if recipe['arm'] == 'frozen' else (
                (1-recipe['weight'])*baseline + recipe['weight']*components[recipe['arm']])
            if not np.isfinite(selected).all() or np.any((selected < 0) | (selected > 1)):
                raise ValueError('Invalid selected probabilities')
            new, old = rank_metrics(labels, selected), rank_metrics(labels, baseline)
            row = report[(report.task == task) & (report.seed == seed) & (report.method == 'selected')]
            if len(row) != 1:
                raise ValueError('Missing or duplicate selected result')
            for key in ('auc', 'ap'):
                np.testing.assert_allclose(float(row.iloc[0][key]), new[key], atol=1e-12, rtol=0)
            delta = {key:new[key]-old[key] for key in ('auc', 'ap')}
            actual_deltas.append(dict(task=task, seed=seed, **delta))
            direction = 1.-2.*labels.astype(np.float64)
            adverse = rank_metrics(labels, selected+epsilon*direction)
            favorable = rank_metrics(labels, baseline-epsilon*direction)
            sensitivity.append(dict(task=task, seed=seed, epsilon=epsilon,
                adverse_delta_auc=adverse['auc']-favorable['auc'],
                adverse_delta_ap=adverse['ap']-favorable['ap']))
            dest = out/'selected'/task/('s%d' % seed)
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest/'scores.npz', val=selected, baseline=baseline,
                                y_val=labels, pairs_val=pairs, **components)
            exported.append(dict(task=task, seed=seed, recipe=recipe, actual_metrics=new,
                path=str((dest/'scores.npz').relative_to(out)), sha256=base.digest(dest/'scores.npz')))
    summaries = {}
    for task in p['tasks']:
        group = [row for row in actual_deltas if row['task'] == task]
        perturb = [row for row in sensitivity if row['task'] == task]
        for key in ('auc', 'ap'):
            np.testing.assert_allclose(np.mean([row[key] for row in group]),
                outcome['gates']['selected'][task]['mean_deltas'][key], atol=1e-12, rtol=0)
            if sum(row[key] > 0 for row in group) != outcome['gates']['selected'][task]['positive_seed_labels'][key]:
                raise ValueError('Positive seed-label count changed')
        summaries[task] = dict(actual_mean_deltas={key:float(np.mean([r[key] for r in group])) for key in ('auc','ap')},
            adverse_mean_deltas={key:float(np.mean([r['adverse_delta_'+key] for r in perturb])) for key in ('auc','ap')},
            adverse_positive_seed_labels={key:sum(r['adverse_delta_'+key] > 0 for r in perturb) for key in ('auc','ap')})
    pd.DataFrame(sensitivity).to_csv(out/'numerical_sensitivity.csv', index=False)
    result = dict(protocol_sha256=p['sha256'], numerical_audit_protocol_sha256=ap['sha256'],
        source_sha256=base.digest(__file__), source_file=Path(__file__).name,
        selection_sha256=base.digest(out/'SELECTION.json'), validation_sha256=base.digest(out/'VALIDATION.json'),
        selected_predictions_recomputed=True, exported=exported,
        numerical_diagnostic=dict(posthoc=True, epsilon=epsilon, summaries=summaries,
            meaning='Worst direction within this per-score absolute perturbation bound: candidate positives down/negatives up, baseline positives up/negatives down. This audits numerical precision only; no confidence interval, data-split robustness or statistical significance claim.'))
    base.write_json(out/'SELECTED_PREDICTIONS.json', result)
    print('EXPORTED_FROZEN_SELECTION', len(exported), flush=True)
    print('NUMERICAL_SENSITIVITY', summaries, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=experiment.DEFAULT_OUT)
    args = parser.parse_args()
    analyze(Path(args.out_dir).resolve())


if __name__ == '__main__':
    main()
