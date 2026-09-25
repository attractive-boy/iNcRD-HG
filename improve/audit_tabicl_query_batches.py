"""Post-hoc direct single-view versus concatenated-view numerical audit.

The internally frozen recipe and original baseline are retained. No fitting,
configuration selection or replacement of the primary predictions is allowed.
"""
import argparse
from pathlib import Path
import gc

import numpy as np
import torch
from tabicl import TabICLClassifier

from improve import tabular_foundation as base
from improve import tabicl_graph_context as experiment
from improve import tabicl_graph_context_audit as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=experiment.DEFAULT_OUT)
    parser.add_argument('--task', choices=('LncDrug', 'MiDrug'), required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    p = experiment.protocol(out)
    ap = audit.audit_protocol(out, p)
    selected = experiment.selection(out, p)['chosen'][args.task]['recipe']
    exports = base.read_json(out/'SELECTED_PREDICTIONS.json')
    if exports['validation_sha256'] != base.digest(out/'VALIDATION.json'):
        raise ValueError('Verified primary predictions changed')
    a = experiment.features(out, p, args.task, 'outer')
    x, full_query, query_views = experiment.view(p, args.task, a)
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    arrays = dict(y_val=a['y_val'], pairs_val=a['pairs_val'])
    for seed in p['seeds']:
        original = experiment.baseline_score(p, 'outer', args.task, seed, a)
        export = next(v for v in exports['exported'] if (v['task'], v['seed']) == (args.task, seed))
        if base.digest(out/export['path']) != export['sha256']:
            raise ValueError('Exported selection changed')
        with np.load(out/export['path'], allow_pickle=False) as archive:
            primary = archive['val'].copy()
        if selected['arm'] == 'frozen':
            separate, view_errors = original.copy(), []
        else:
            views, view_errors = [], []
            full_model = None
            for fold in range(5):
                kind = 'full' if selected['arm'] == 'query_bag' else 'fold%d' % fold
                dest = experiment.directory(out, 'outer', args.task, seed, kind)
                experiment.checked_result(dest, p)
                if kind == 'full' and full_model is not None:
                    model = full_model
                else:
                    model = TabICLClassifier.load(dest/'fitted.pkl', device=args.device)
                    if base.parameter_digest(model) != p['parameter_sha256']:
                        raise ValueError('Official model parameters changed')
                    if kind == 'full':
                        full_model = model
                query = full_query if selected['arm'] == 'fold_full' else query_views[fold]
                score = base.query_scores(model, query)
                row = fold+1 if kind == 'full' else (0 if selected['arm'] == 'fold_full' else 1)
                with np.load(dest/'scores.npz', allow_pickle=False) as archive:
                    reference = archive['views'][row].copy()
                np.testing.assert_allclose(score, reference,
                    atol=p['query_invariance_atol'], rtol=p['query_invariance_rtol'])
                error = float(np.max(np.abs(score-reference)))
                view_errors.append(error)
                views.append(score)
                print('SINGLE_VIEW', args.task, seed, fold, error, flush=True)
                if kind != 'full':
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            separate = (1-selected['weight'])*original + selected['weight']*np.mean(views, axis=0)
            if full_model is not None:
                del model, full_model
                gc.collect()
                torch.cuda.empty_cache()
        np.testing.assert_allclose(separate, primary,
            atol=p['query_invariance_atol'], rtol=p['query_invariance_rtol'])
        before = base.metrics(a['y_val'], primary, a['pairs_val'])
        after = base.metrics(a['y_val'], separate, a['pairs_val'])
        baseline = base.metrics(a['y_val'], original, a['pairs_val'])
        records.append(dict(seed=seed, max_view_abs=max(view_errors, default=0.),
            max_selected_abs=float(np.max(np.abs(separate-primary))), primary_metrics=before,
            separate_query_metrics=after, separate_query_delta={key:after[key]-baseline[key] for key in base.METRICS}))
        arrays['separate_s%d' % seed] = separate
        arrays['primary_s%d' % seed] = primary
    output = out/('QUERY_BATCH_AUDIT_%s.npz' % args.task)
    np.savez_compressed(output, **arrays)
    result = dict(protocol_sha256=p['sha256'], numerical_audit_protocol_sha256=ap['sha256'],
        task=args.task, source_sha256=base.digest(__file__), selected_recipe=selected,
        posthoc=True, passed=True, primary_predictions_unchanged=True, records=records,
        arrays_file=output.name, arrays_sha256=base.digest(output),
        mean_separate_query_delta={key:float(np.mean([r['separate_query_delta'][key] for r in records])) for key in base.METRICS},
        positive_seed_labels={key:sum(r['separate_query_delta'][key] > 0 for r in records) for key in ('auc','ap')})
    base.write_json(out/('QUERY_BATCH_AUDIT_%s.json' % args.task), result)
    print('QUERY_BATCH_AUDIT', args.task, result['mean_separate_query_delta'], result['positive_seed_labels'], flush=True)


if __name__ == '__main__':
    main()
