"""Frozen outer recheck for nested-anchor residual graph candidates.

Uses existing historically exposed outer inputs/references without refitting
them. This is not an independent test. New OOF offsets use outer TRAINING
observations only. No candidate advances without its frozen partition gate.
"""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from improve import anchor_residual_validate_v2 as development
from improve import mechanism_outer_recheck as previous
from improve import assess_declared_graph as audit

experiment, base = development.experiment, development.base
ROOT, DEV = development.ROOT, development.shared.data_source.OUT
OUT = ROOT / 'results/mechanism_outer_recheck_20260915'
CACHE = ROOT / 'results/nested_anchor_v3_20260915/outer'
OWN = ('anchor_residual_outer_v1.py', 'test_anchor_residual_outer_v1.py')


def label_for(family):
    return development.label_for(family)


def verify_partition_report(path, dp, cp, development_root=DEV):
    """Require the complete, unchanged paired evidence behind a passing gate."""
    path, development_root = Path(path), Path(development_root)
    assessment = base.read_json(path)
    gates = ('all_metrics_recomputed', 'additional_development_gate',
             'performance_gate', 'primary_mechanism_gate', 'offset_null_gate')
    if not all(assessment.get(name) is True for name in gates):
        raise ValueError('Candidate did not pass its fixed-partition gates')
    if (assessment['candidate_sha256'] != cp['sha256'] or
            assessment['data_protocol_sha256'] != dp['sha256']):
        raise ValueError('Partition assessment belongs to another candidate or data protocol')
    expected = {str(development_root / 'runs' / cp['label'] / task / str(split) / kind / 'result.json')
                for task in dp['tasks'] for split in dp['split_seeds']
                for kind in cp['recipes'][task]}
    if set(assessment['result_sha256']) != expected:
        raise ValueError('Partition assessment has missing or extra registered fits')
    for name, sha in assessment['result_sha256'].items():
        if base.digest(name) != sha:
            raise ValueError('Partition fit record changed after assessment')
        record = base.read_json(name)
        if (record['candidate_sha256'] != cp['sha256'] or
                record['data_protocol_sha256'] != dp['sha256']):
            raise ValueError('Partition fit candidate or data protocol mismatch')
        development.shared.verify_artifacts(Path(name).parent, record)
    required = {'metrics.csv', 'contrasts.csv', 'summary.csv', 'endpoint_uncertainty.json'}
    if set(assessment['artifacts']) != required:
        raise ValueError('Incomplete partition report artifacts')
    for name, sha in assessment['artifacts'].items():
        if base.digest(path.parent / name) != sha:
            raise ValueError('Partition report artifact changed after assessment')
    return assessment


def freeze(out, family, cache=CACHE):
    out, cache = Path(out).resolve(), Path(cache).resolve()
    label = label_for(family)
    development_dp, cp = development.checked(DEV, family)
    assessment_path = DEV / 'reports' / label / 'ASSESSMENT.json'
    verify_partition_report(assessment_path, development_dp, cp)
    dest = out / 'candidates' / (label + '.json')
    if dest.exists():
        raise ValueError('This candidate is already frozen for outer recheck')
    dp = previous.data_protocol(out)  # Metadata/hashes only; no query outcomes.
    if not cache.exists():
        raise ValueError('Prepare outer training features and freeze the v3 reference cache first')
    cache_p = experiment.nested.protocol(cache)
    if cache_p['scope'] != 'outer' or cache_p['version'] != 'nested_anchor_v3_constant_filter_audit':
        raise ValueError('Wrong nested reference scope or audit version')
    development_cache = experiment.nested.protocol(cp['nested_reference'])
    for key in ('members', 'reference_seed', 'inner_folds', 'feature_fold_seed',
                'graph_recipes', 'input_configs', 'evidence_dir', 'anchor', 'icl_protocol'):
        if cache_p[key] != development_cache[key]:
            raise ValueError('Outer training reference recipe changed: ' + key)
    sources = {**cp['source_sha256'], **{n: base.digest(ROOT / n) for n in OWN}}
    recipe = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        label=label, family=family, tasks=dp['tasks'], kinds=cp['training_parameters']['kinds'],
        seeds=dp['seeds'], primary_kind=cp['primary_kind'], primary_controls=cp['primary_controls'],
        recipes=cp['recipes'], training_parameters=cp['training_parameters'],
        outer_data_protocol_sha256=dp['sha256'],
        development_candidate_sha256=base.digest(DEV / 'candidates' / (label + '.json')),
        development_assessment_sha256=base.digest(assessment_path),
        development_gate=True, source_sha256=sources,
        nested_reference=str(cache), nested_protocol_sha256=cache_p['sha256'],
        strong_weight=.1, lnc_absolute_tolerance=.001,
        training_policy='Unchanged median historical steps and learning rates. Train only on fixed outer training observations plus their strictly nested OOF anchor offsets. No outer outcome enters fitting or selection.',
        prediction='Saved outer anchor + 0.1 * mean raw graph residual logits.',
        historical_surface_previously_exposed=True, independent_test=False,
        this_candidate_outer_predictions_used_for_selection=False,
        original_reserved_test_source_access=False, confirmed_final_innovations=0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    base.write_json(dest, recipe)
    snapshot = out / 'candidate_sources' / label
    snapshot.mkdir(parents=True)
    for name in sources:
        shutil.copy2(ROOT / name, snapshot / name)
    print('FROZEN_ANCHOR_RESIDUAL_OUTER', family, recipe['recipes'], flush=True)


def checked(out, family):
    out = Path(out)
    dp = previous.data_protocol(out)
    label = label_for(family)
    cp = base.check_seal(base.read_json(out / 'candidates' / (label + '.json')))
    if cp['outer_data_protocol_sha256'] != dp['sha256']:
        raise ValueError('Outer source protocol changed')
    for name, expected in cp['source_sha256'].items():
        if base.digest(ROOT / name) != expected:
            raise ValueError('Frozen outer candidate source changed: ' + name)
    for path, key in ((DEV / 'candidates' / (label + '.json'), 'development_candidate_sha256'),
                      (DEV / 'reports' / label / 'ASSESSMENT.json', 'development_assessment_sha256')):
        if base.digest(path) != cp[key]:
            raise ValueError('Frozen development result or recipe changed')
    if experiment.nested.protocol(cp['nested_reference'])['sha256'] != cp['nested_protocol_sha256']:
        raise ValueError('Outer OOF reference protocol changed')
    return dp, cp


def training_inputs(a, cache, task):
    training = {k: v for k, v in a.items() if k in experiment.TRAIN_KEYS}
    training['anchor_oof'], manifest = experiment.offset_arrays(cache, task, 'outer', training)
    return training, manifest


def run(out, family, task, device):
    out = Path(out)
    dp, cp = checked(out, family)
    a = previous.arrays(out, dp, task)
    rd = out / 'references' / task
    rr = base.read_json(rd / 'result.json')
    if rr['data_protocol_sha256'] != dp['sha256']:
        raise ValueError('Wrong outer reference input protocol')
    development.shared.verify_artifacts(rd, rr)
    refs = experiment.common.load(rd / 'scores.npz')
    for name, key in (('pairs', 'pairs_val'), ('y', 'y_val')):
        np.testing.assert_array_equal(refs[name], a[key])
    training, offset_manifest = training_inputs(a, cp['nested_reference'], task)
    torch.set_num_threads(4)
    for kind, recipe in cp['recipes'][task].items():
        for seed in cp['seeds']:
            dest = out / 'runs' / cp['label'] / task / kind / ('s' + str(seed))
            if (dest / 'result.json').exists():
                r = base.read_json(dest / 'result.json')
                if r['candidate_sha256'] != cp['sha256']:
                    raise ValueError('Existing run belongs to a different candidate')
                development.shared.verify_artifacts(dest, r)
                continue
            model, _, trace, detail = experiment.fit(training, cp['training_parameters'], kind,
                recipe['lr'], seed, device, fixed_epochs=recipe['epochs'])
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(training, cp['training_parameters'], kind, device)
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
            query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
            value = experiment.predict(model, xs, graphs, query, cp['training_parameters']).cpu().numpy()
            replay = experiment.predict(loaded, xs, graphs, query.flip(1), cp['training_parameters']).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, query[:, i:i + 800],
                cp['training_parameters']).cpu().numpy() for i in range(0, query.shape[1], 800)])
            for actual in (replay, chunks):
                np.testing.assert_allclose(value, actual, atol=3e-5, rtol=3e-5)
            scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=value,
                anchor=refs['anchor_' + str(seed)], tabicl=refs['tabicl_' + str(seed)])
            for name in ('anchor', 'tabicl'):
                scores[name + '_composed'] = scores[name] + cp['strong_weight'] * value
            scores['offset'] = scores['anchor'].copy()
            scores['offset_composed'] = scores['anchor'].copy()
            np.savez_compressed(dest / 'scores.npz', **scores)
            metrics = {n: experiment.common.base.metrics(scores['y'], v, scores['pairs'])
                       for n, v in scores.items() if n not in ('y', 'pairs')}
            record = dict(candidate_sha256=cp['sha256'], data_protocol_sha256=dp['sha256'],
                task=task, kind=kind, seed=seed, recipe=recipe, **detail,
                nested_offset_manifest_sha256=base.digest(offset_manifest),
                replay_max_abs=float(max(abs(value - replay).max(), abs(value - chunks).max())),
                metrics={n: v for n, v in metrics.items() if n not in ('offset', 'offset_composed')},
                offset_metrics={n: metrics[n] for n in ('offset', 'offset_composed')},
                files={n: base.digest(dest / n) for n in ('model.pt', 'trace.json', 'scores.npz')})
            base.write_json(dest / 'result.json', record)
            print('ANCHOR_RESIDUAL_OUTER_VERIFIED', family, task, kind, seed,
                  record['metrics']['anchor_composed'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()


def report(out, family):
    out = Path(out)
    dp, cp = checked(out, family)
    records, rows, files, intervals = {}, [], {}, {}
    for task in cp['tasks']:
        cells = []
        for seed in cp['seeds']:
            cell = None
            for kind in cp['kinds']:
                dest = out / 'runs' / cp['label'] / task / kind / ('s' + str(seed))
                r = base.read_json(dest / 'result.json')
                if r['candidate_sha256'] != cp['sha256']:
                    raise ValueError('Outer fit candidate mismatch')
                development.shared.verify_artifacts(dest, r)
                z = experiment.common.load(dest / 'scores.npz')
                for name, expected in {**r['metrics'], **r['offset_metrics']}.items():
                    actual = experiment.common.base.metrics(z['y'], z[name], z['pairs'])
                    for m in ('auc', 'ap', 'macro_drug_ap'):
                        if abs(actual[m] - expected[m]) >= 1e-12:
                            raise ValueError('Stored outer metric mismatch')
                    rows.append(dict(task=task, seed=seed, kind=kind, score=name, **actual))
                if cell is None:
                    cell = dict(y=z['y'], pairs=z['pairs'], scores={n: z[n] for n in ('anchor', 'offset_composed')})
                for name in ('y', 'pairs'):
                    np.testing.assert_array_equal(z[name], cell[name])
                for name in ('anchor', 'offset_composed'):
                    np.testing.assert_array_equal(z[name], cell['scores'][name])
                np.testing.assert_array_equal(z['anchor'], z['offset_composed'])
                cell['scores'][kind] = z['anchor_composed']
                records[task, kind, seed] = r
                files[str(dest / 'result.json')] = base.digest(dest / 'result.json')
            cells.append(cell)
        intervals[task] = development.shared.endpoint_intervals(cells, primary=cp['primary_kind'])
        print('OUTER_UNCERTAINTY_COMPLETE', family, task, flush=True)
    result = audit.evaluate(cp, records, cp['seeds'])
    dest = out / 'reports' / cp['label']
    if (dest / 'ASSESSMENT.json').exists():
        raise ValueError('Outer assessment already exists; preserve it')
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / 'metrics.csv', index=False)
    base.write_json(dest / 'endpoint_uncertainty.json', intervals)
    result.update(candidate_sha256=cp['sha256'], data_protocol_sha256=dp['sha256'],
        created_at_utc=datetime.now(timezone.utc).isoformat(), all_metrics_recomputed=True,
        frozen_recheck_gate=result['declared_gate'], historical_surface_previously_exposed=True,
        independent_test=False, original_reserved_test_source_access=False,
        no_recipe_selection_on_this_candidates_outer_scores=True,
        limitations=['Historically exposed outer development surface; not independent biological validation.',
            'Conditional endpoint intervals do not correct extensive method search or retraining uncertainty.',
            'LncDrug tolerance 0.001 is an engineering gate, not a demonstrated noise range.'],
        result_sha256=files, artifacts={n: base.digest(dest / n) for n in ('metrics.csv', 'endpoint_uncertainty.json')})
    base.write_json(dest / 'ASSESSMENT.json', base.seal(result))
    print('ANCHOR_RESIDUAL_OUTER_GATES', family, result['performance_gate'],
          result['primary_mechanism_gate'], result['offset_null_gate'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'report'))
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--family', choices=tuple(experiment.LIBRARIES), required=True)
    parser.add_argument('--cache', default=str(CACHE))
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.out, args.family, args.cache)
    elif args.command == 'run':
        run(args.out, args.family, args.task, args.device)
    else:
        report(args.out, args.family)
