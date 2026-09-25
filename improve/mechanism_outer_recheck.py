"""Frozen candidate rechecks on the historically exposed outer partition.

This is not an independent test. A candidate must pass the additional
development gates and be frozen here before this executor opens outer arrays.
Reference construction is shared and does not rerun the six development fits.
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib
from pathlib import Path
import pickle
import re
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch

from improve import tabular_foundation as base
from improve import graph_mechanism_validate as validation
from improve import residual_graph_validate as development
from improve import context_anchor_verify as anchor
from improve import anchor_prediction_precision as anchor_precision
from improve import graph_observation_offset as offset
from improve import assess_declared_graph as assessor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/mechanism_outer_recheck_20260915'
DEV = ROOT / 'results/mechanism_validation_20260915'
CONTEXT = ROOT / 'results/tabicl_pathway_context_20260913'
ARCHIVES = ROOT / 'results/backbone_replacement_20260910/inputs'
TASKS, SEEDS = ('MiDrug', 'LncDrug'), (666, 0, 1)


def rebuild_physical(template, description, task, pairs, labels, features):
    """Use known non-target relations and exactly the new training positives."""
    rna = 'miRNA' if task == 'MiDrug' else 'lncRNA'
    target_relation = description['physical_relations'].index([rna, task, 'drug'])
    if not np.array_equal(template['relation'] == target_relation, template['target_codes'] >= 0):
        raise ValueError('Template target relation/codes disagree')
    if len(np.unique(pairs, axis=1).T) != len(labels) or not set(np.unique(labels)) <= {0, 1}:
        raise ValueError('Training pairs must be unique binary observations')
    result = {k: template[k].copy() for k in ('node_counts', 'pair_offset', 'nd')}
    for i, name in enumerate(description['node_types']):
        key = ('pca_x_' if name in ('lncRNA', 'miRNA') else 'x_') + name
        np.testing.assert_array_equal(template['x_' + str(i)], features[key])
        result['x_' + str(i)] = features[key].copy()
    edges, relations, codes = [], [], []
    for relation in range(len(description['physical_relations'])):
        if relation == target_relation:
            positive = pairs[:, labels == 1]
            e = positive + result['pair_offset'][:, None]
            c = positive[0] * int(result['nd']) + positive[1]
        else:
            keep = template['relation'] == relation
            e, c = template['edges'][:, keep], template['target_codes'][keep]
        edges.append(e)
        codes.append(c)
        relations.append(np.full(len(c), relation, dtype=np.int64))
    result.update(edges=np.concatenate(edges, 1), relation=np.concatenate(relations),
                  target_codes=np.concatenate(codes))
    return result


def freeze(out, family):
    out = Path(out).resolve()
    _, cp = development.checked(DEV, family)
    label = cp['label']
    assessment_path = DEV / 'reports' / label / 'ASSESSMENT.json'
    assessment = base.read_json(assessment_path)
    if not assessment['all_metrics_recomputed'] or not assessment['additional_development_gate']:
        raise ValueError('Candidate did not pass the additional development gates')
    if (out / 'candidates' / (label + '.json')).exists():
        raise ValueError('Candidate is already frozen for outer recheck')
    executor_names = [n[:-3] for n in cp['source_sha256'] if re.fullmatch(r'offset_graph_candidates_v\d+\.py', n)]
    executor = max(executor_names, key=lambda n: int(n.rsplit('v', 1)[1])) if executor_names else 'residual_graph_experiment'
    module = importlib.import_module('improve.' + executor)
    if family not in module.LIBRARIES:
        raise ValueError('Frozen training executor does not implement this family')
    out.mkdir(parents=True, exist_ok=True)
    data_file = out / 'DATA_PROTOCOL.json'
    if not data_file.exists():
        from improve import tabicl_context_program as context
        from improve import tabicl_graph_context as graph
        pp = context.protocol(CONTEXT)
        context.selected(CONTEXT, pp)  # Metadata only; no outer array loading.
        gp = graph.protocol(pp['parent_dir'])
        own = ('mechanism_outer_recheck.py', 'test_mechanism_outer_recheck.py',
               'anchor_prediction_precision.py', 'context_anchor_verify.py', 'anchor_shrinkage.py',
               'graph_observation_offset.py', 'assess_declared_graph.py', 'graph_mechanism_validate.py',
               'tabular_foundation.py')
        sources = {**pp['source_sha256'], **gp['source_sha256'],
                   **{n: base.digest(ROOT / n) for n in own}}
        evidence = Path(pp['inputs_dir'])
        em = base.read_json(evidence / 'INPUTS.json')
        if em['labels_read']:
            raise ValueError('Context inputs must be free of task outcomes')
        paths = [CONTEXT / 'protocol.json', CONTEXT / 'SELECTION.json',
                 Path(pp['parent_dir']) / 'protocol.json', Path(pp['parent_dir']) / 'SELECTION.json',
                 Path(gp['parent_dir']) / 'protocol.json', Path(gp['parent_dir']) / 'SELECTION.json',
                 evidence / 'INPUTS.json',
                 ROOT / 'research/new_mechanisms_20260915/RESERVED_DATA_PROVENANCE_AUDIT.json']
        paths.extend(evidence / n for n in em['files'])
        for task in TASKS:
            paths.extend([ARCHIVES / task / 'outer.npz', DEV / 'inputs' / task / '61453/arrays.npz',
                          DEV / 'inputs' / task / '61453/graph.json'])
            for directory in (Path(pp['parent_dir']), Path(gp['feature_reference_dir'])):
                fd = directory / 'features/outer' / task
                paths.append(fd / 'manifest.json')
                paths.extend(fd / n for n in base.read_json(fd / 'manifest.json')['files'])
            paths.extend(CONTEXT / 'predictions/outer' / task / ('s' + str(s)) / 'scores.npz' for s in SEEDS)
        dp = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), tasks=list(TASKS),
            seeds=list(SEEDS), neural_fold_seed=20260915, folds=5,
            reference=dict(members=['hgb', 'et', 'mlp', 'lr'], seed=20260914, bank='context76',
                           member_weight=.3, tiered_score='tiered_0p5000'),
            context_protocol=pp, source_sha256=sources,
            input_sha256={str(f): base.digest(f) for f in paths},
            original_reserved_test_labels_opened=False, historical_outer_query_labels_opened_at_freeze=False,
            provenance='Historically exposed outer development partition. Earlier project experiments used this surface; this is a frozen recheck, not an independent test.',
            policy='No recipe selection, early stopping, coefficient tuning or replacement of defaults on outer outcomes. Rebuild only target support edges. Other-task relations and fixed PCA follow the established transductive protocol.',
            uncertainty='1000 paired RNA/drug pigeonhole draws shared across all seeds; conditional on fitted and selected models. No multiple-search or training-uncertainty correction.'))
        (out / 'data_source_snapshot').mkdir()
        for n in sources:
            shutil.copy2(ROOT / n, out / 'data_source_snapshot' / n)
        base.write_json(data_file, dp)
    else:
        dp = data_protocol(out)
    candidate = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        label=label, executor=executor, data_protocol_sha256=dp['sha256'],
        development_candidate_sha256=base.digest(DEV / 'candidates' / (label + '.json')),
        development_assessment_sha256=base.digest(assessment_path), development_gate=True,
        recipes=cp['recipes'], training_parameters=cp['training_parameters'],
        primary_kind=cp['primary_kind'], primary_controls=cp['primary_controls'],
        kinds=list(cp['training_parameters']['kinds']), seeds=list(SEEDS), strong_weight=.1,
        lnc_absolute_tolerance=.001, source_sha256=cp['source_sha256'],
        gate=cp['gate'], candidate_outer_predictions_used_for_selection=False,
        historical_surface_previously_exposed=True))
    (out / 'candidates').mkdir(exist_ok=True)
    base.write_json(out / 'candidates' / (label + '.json'), candidate)
    print('FROZEN_OUTER_RECHECK', label, candidate['recipes'], flush=True)


def data_protocol(out):
    out = Path(out)
    p = base.check_seal(base.read_json(out / 'DATA_PROTOCOL.json'))
    for n, expected in p['source_sha256'].items():
        if base.digest(ROOT / n) != expected:
            raise ValueError('Outer data executor source changed: ' + n)
    for n, expected in p['input_sha256'].items():
        if base.digest(n) != expected:
            raise ValueError('Outer source input changed: ' + n)
    return p


def checked(out, family):
    out = Path(out)
    p = data_protocol(out)
    cp = base.check_seal(base.read_json(out / 'candidates' / ('residual_' + family + '.json')))
    if cp['data_protocol_sha256'] != p['sha256']:
        raise ValueError('Wrong outer data protocol')
    for n, expected in cp['source_sha256'].items():
        if base.digest(ROOT / n) != expected:
            raise ValueError('Frozen candidate source changed: ' + n)
    if base.digest(DEV / 'candidates' / (cp['label'] + '.json')) != cp['development_candidate_sha256']:
        raise ValueError('Development recipe changed')
    if base.digest(DEV / 'reports' / cp['label'] / 'ASSESSMENT.json') != cp['development_assessment_sha256']:
        raise ValueError('Development gate changed')
    return p, cp


def prepare_inputs(out, family, task):
    from improve import tabicl_context_program as context
    from improve.tabicl_context_program_modules import ContextPrograms
    out = Path(out)
    p, _ = checked(out, family)  # Must precede every outer np.load.
    dest = out / 'inputs' / task
    if (dest / 'manifest.json').exists():
        record = base.read_json(dest / 'manifest.json')
        validation.verify_artifacts(dest, record)
        if record['data_protocol_sha256'] != p['sha256']:
            raise ValueError('Input cache protocol mismatch')
        return
    pp = p['context_protocol']
    data, _ = context.stage_data(CONTEXT, pp, task, 'outer')
    features = ContextPrograms.load(pp['inputs_dir'])
    prefix = 'raw' if pp['input_configs'][task]['input_view'] == 'raw' else 'x'
    values = {k: data[k].copy() for k in ('pairs_train', 'y_train', 'pairs_val', 'y_val')}
    with np.load(ARCHIVES / task / 'outer.npz', allow_pickle=False) as archive:
        for k, v in values.items():
            np.testing.assert_array_equal(archive[k], v)
        description = base.read_json(DEV / 'inputs' / task / '61453/graph.json')
        template = validation.block.load(DEV / 'inputs' / task / '61453/arrays.npz')
        physical = rebuild_physical(template, description, task, values['pairs_train'],
                                    values['y_train'], archive)
    values.update(physical)
    query_codes = values['pairs_val'][0] * int(values['nd']) + values['pairs_val'][1]
    train_codes = values['pairs_train'][0] * int(values['nd']) + values['pairs_train'][1]
    if np.isin(query_codes, train_codes).any() or np.isin(query_codes, values['target_codes']).any():
        raise ValueError('Outer query overlaps training support')
    values['folds'] = np.zeros(len(values['y_train']), dtype=np.int64)
    for fold, (_, indices) in enumerate(StratifiedKFold(p['folds'], shuffle=True,
        random_state=p['neural_fold_seed']).split(values['pairs_train'].T, values['y_train'])):
        values['folds'][indices] = fold
    for phase in ('train', 'val'):
        values['context76_' + phase] = np.concatenate((data[prefix + '_' + phase],
            features.transform(task, values['pairs_' + phase], 'context')), 1).astype(np.float32)
        if values['context76_' + phase].shape[1] != 76:
            raise ValueError('Expected original context76 bank')
    training = {k: v for k, v in values.items() if k in development.BASE_TRAIN_KEYS}
    nodes, intercepts, diagnostics = offset.context_offsets(training)
    values.update(offset_nodes=nodes, offset_intercepts=intercepts)
    dest.mkdir(parents=True)
    np.savez_compressed(dest / 'arrays.npz', **values)
    base.write_json(dest / 'offset_diagnostics.json', diagnostics)
    base.write_json(dest / 'manifest.json', dict(data_protocol_sha256=p['sha256'], task=task,
        target_overlap=0, training_rows=len(values['y_train']), query_rows=len(values['y_val']),
        no_prediction_metrics_during_preparation=True, historical_outer_query_labels_opened=True,
        files={n: base.digest(dest / n) for n in ('arrays.npz', 'offset_diagnostics.json')}))
    print('OUTER_INPUTS_PREPARED', task, len(values['y_train']), len(values['y_val']), flush=True)


def arrays(out, p, task):
    dest = Path(out) / 'inputs' / task
    record = base.read_json(dest / 'manifest.json')
    if record['data_protocol_sha256'] != p['sha256']:
        raise ValueError('Input data protocol mismatch')
    validation.verify_artifacts(dest, record)
    return validation.block.load(dest / 'arrays.npz')


def reference(out, family, task):
    out = Path(out)
    p, _ = checked(out, family)
    a = arrays(out, p, task)
    dest = out / 'references' / task
    if (dest / 'result.json').exists():
        r = base.read_json(dest / 'result.json')
        if r['data_protocol_sha256'] != p['sha256']:
            raise ValueError('Reference protocol mismatch')
        validation.verify_artifacts(dest, r)
        return
    dest.mkdir(parents=True, exist_ok=True)
    member_path = dest / 'anchor_members.pkl'
    if member_path.exists():
        with member_path.open('rb') as stream:
            members = pickle.load(stream)
    else:
        members = {n: anchor.learner(n, p['reference']['seed']).fit(a['context76_train'], a['y_train'])
                   for n in p['reference']['members']}
        with member_path.open('wb') as stream:
            pickle.dump(members, stream, protocol=4)
    names = p['reference']['members']
    native = np.stack([anchor.anchors.logit(members[n].predict_proba(a['context76_val'])[:, 1]) for n in names]).mean(0)
    stable = anchor_precision.member_mean(members, names, a['context76_val'])
    with member_path.open('rb') as stream:
        replay_members = pickle.load(stream)
    replay = anchor_precision.member_mean(replay_members, names, a['context76_val'][::-1])[::-1]
    np.testing.assert_allclose(stable, replay, atol=1e-10, rtol=1e-10)
    scores = dict(y=a['y_val'], pairs=a['pairs_val'])
    for seed in p['seeds']:
        z = validation.block.load(CONTEXT / 'predictions/outer' / task / ('s' + str(seed)) / 'scores.npz')
        np.testing.assert_array_equal(z['y_val'], a['y_val'])
        np.testing.assert_array_equal(z['pairs_val'], a['pairs_val'])
        tiered = anchor.anchors.logit(z[p['reference']['tiered_score']])
        scores['tabicl_' + str(seed)] = tiered
        scores['anchor_' + str(seed)] = .7 * tiered + .3 * stable
        scores['anchor_native_' + str(seed)] = .7 * tiered + .3 * native
    np.savez_compressed(dest / 'scores.npz', **scores)
    metrics = {n: validation.block.base.metrics(scores['y'], v, scores['pairs'])
               for n, v in scores.items() if n not in ('y', 'pairs')}
    base.write_json(dest / 'result.json', dict(data_protocol_sha256=p['sha256'], task=task,
        member_seed=p['reference']['seed'], reference_replay_max_abs=float(abs(stable - replay).max()),
        reused_archived_tiered_predictions=True, development_references_refitted=False,
        numerical_metric_deltas={str(s): {m: metrics['anchor_' + str(s)][m] - metrics['anchor_native_' + str(s)][m]
            for m in ('auc', 'ap')} for s in p['seeds']}, metrics=metrics,
        files={n: base.digest(dest / n) for n in ('scores.npz', 'anchor_members.pkl')}))
    print('OUTER_REFERENCE_VERIFIED', task, {n: v for n, v in metrics.items() if n.startswith('anchor_')}, flush=True)


def run(out, family, task, device):
    out = Path(out)
    p, cp = checked(out, family)
    ex = importlib.import_module('improve.' + cp['executor'])
    a = arrays(out, p, task)
    rd = out / 'references' / task
    rr = base.read_json(rd / 'result.json')
    if rr['data_protocol_sha256'] != p['sha256']:
        raise ValueError('Wrong reference data protocol')
    validation.verify_artifacts(rd, rr)
    refs = validation.block.load(rd / 'scores.npz')
    np.testing.assert_array_equal(refs['pairs'], a['pairs_val'])
    np.testing.assert_array_equal(refs['y'], a['y_val'])
    training = {k: v for k, v in a.items() if k in development.experiment.TRAIN_KEYS}
    null = offset.offset_prediction(training, a['pairs_val'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for kind, recipe in cp['recipes'][task].items():
        for seed in p['seeds']:
            dest = out / 'runs' / cp['label'] / task / kind / ('s' + str(seed))
            if (dest / 'result.json').exists():
                r = base.read_json(dest / 'result.json')
                if r['candidate_sha256'] != cp['sha256']:
                    raise ValueError('Existing run used a different candidate')
                validation.verify_artifacts(dest, r)
                continue
            model, _, trace, detail = ex.fit(training, cp['training_parameters'], kind,
                recipe['lr'], seed, device, fixed_epochs=recipe['epochs'])
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), dest / 'model.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = ex.setup(training, cp['training_parameters'], kind, device)
            loaded.load_state_dict(torch.load(dest / 'model.pt', weights_only=True, map_location=device))
            pairs = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
            def predict(fitted, query):
                if hasattr(ex, 'predict'):
                    return ex.predict(fitted, xs, graphs, query, cp['training_parameters']).cpu().numpy()
                return ex.precision.geometry_prediction(fitted, xs, graphs, query).cpu().numpy()
            prediction = predict(model, pairs)
            replay = predict(loaded, pairs.flip(1))[::-1]
            chunks = np.concatenate([predict(loaded, pairs[:, i:i + 800]) for i in range(0, pairs.shape[1], 800)])
            for values in (replay, chunks):
                np.testing.assert_allclose(prediction, values, atol=3e-5, rtol=3e-5)
            scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=prediction,
                          offset=null, anchor=refs['anchor_' + str(seed)], tabicl=refs['tabicl_' + str(seed)])
            scores['anchor_composed'] = .9 * scores['anchor'] + .1 * prediction
            scores['offset_composed'] = .9 * scores['anchor'] + .1 * null
            np.savez_compressed(dest / 'scores.npz', **scores)
            r = dict(candidate_sha256=cp['sha256'], data_protocol_sha256=p['sha256'],
                task=task, kind=kind, seed=seed, recipe=recipe, **detail,
                replay_max_abs=float(max(abs(prediction - replay).max(), abs(prediction - chunks).max())),
                metrics={n: validation.block.base.metrics(scores['y'], v, scores['pairs'])
                         for n, v in scores.items() if n not in ('y', 'pairs', 'offset', 'offset_composed')},
                offset_metrics={n: validation.block.base.metrics(scores['y'], scores[n], scores['pairs'])
                                for n in ('offset', 'offset_composed')},
                files={n: base.digest(dest / n) for n in ('model.pt', 'trace.json', 'scores.npz')})
            base.write_json(dest / 'result.json', r)
            print('OUTER_CANDIDATE_VERIFIED', cp['label'], task, kind, seed, r['metrics']['anchor_composed'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()


def report(out, family):
    out = Path(out)
    p, cp = checked(out, family)
    records, rows, cells, result_hashes = {}, [], {task: [] for task in p['tasks']}, {}
    for task in p['tasks']:
        for seed in p['seeds']:
            cell_scores = {}
            first_y, first_pairs = None, None
            for kind in cp['kinds']:
                dest = out / 'runs' / cp['label'] / task / kind / ('s' + str(seed))
                r = base.read_json(dest / 'result.json')
                if r['candidate_sha256'] != cp['sha256']:
                    raise ValueError('Run used the wrong frozen candidate')
                validation.verify_artifacts(dest, r)
                z = validation.block.load(dest / 'scores.npz')
                if first_y is None:
                    first_y, first_pairs = z['y'], z['pairs']
                else:
                    np.testing.assert_array_equal(z['y'], first_y)
                    np.testing.assert_array_equal(z['pairs'], first_pairs)
                for n, v in z.items():
                    if n in ('y', 'pairs'):
                        continue
                    actual = validation.block.base.metrics(z['y'], v, z['pairs'])
                    expected = r['offset_metrics'][n] if n in r['offset_metrics'] else r['metrics'][n]
                    for m in ('auc', 'ap', 'macro_drug_ap'):
                        if abs(actual[m] - expected[m]) > 1e-12:
                            raise ValueError('Saved outer metric disagrees with raw scores')
                    rows.append(dict(task=task, seed=seed, kind=kind, score=n, **actual))
                records[task, kind, seed] = r
                result_hashes[str(dest / 'result.json')] = base.digest(dest / 'result.json')
                cell_scores[kind] = z['anchor_composed']
                for name in ('anchor', 'offset_composed'):
                    if name in cell_scores:
                        np.testing.assert_array_equal(cell_scores[name], z[name])
                    cell_scores[name] = z[name]
            cells[task].append(dict(y=first_y, pairs=first_pairs, scores=cell_scores))
    evaluation_p = dict(cp['training_parameters'], primary_kind=cp['primary_kind'],
                        primary_controls=cp['primary_controls'])
    assessed = assessor.evaluate(evaluation_p, records, p['seeds'])
    dest = out / 'reports' / cp['label']
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / 'metrics.csv', index=False)
    intervals = {}
    for task in p['tasks']:
        intervals[task] = validation.endpoint_intervals(cells[task], primary=cp['primary_kind'])
        intervals[task]['cells'] = 'Three model initializations evaluated on the identical outer query pairs.'
        intervals[task]['method'] = 'Paired two-way RNA/drug pigeonhole bootstrap; the same endpoint draws are shared across model seeds; mean per-initialization metric difference.'
        print('OUTER_UNCERTAINTY_COMPLETE', task, flush=True)
    base.write_json(dest / 'endpoint_uncertainty.json', intervals)
    assessed.update(created_at_utc=datetime.now(timezone.utc).isoformat(), candidate_sha256=cp['sha256'],
        data_protocol_sha256=p['sha256'], development_gate=cp['development_gate'], all_metrics_recomputed=True,
        recheck_gate=assessed['declared_gate'], historically_exposed_outer=True,
        original_reserved_test_labels_opened=False, independent_validation=False,
        candidate_choices_changed_on_outer=False, result_sha256=result_hashes,
        confirmed_final_innovations=int(cp['development_gate'] and assessed['declared_gate']),
        limitations=[p['provenance'], p['uncertainty'],
            'The 0.001 LncDrug tolerance is engineering, not demonstrated random noise or statistical noninferiority.',
            'A passing engineering recheck does not establish statistical significance or external biological validity.'],
        files={n: base.digest(dest / n) for n in ('metrics.csv', 'endpoint_uncertainty.json')})
    base.write_json(dest / 'ASSESSMENT.json', base.seal(assessed))
    print('OUTER_RECHECK_GATES', assessed['performance_gate'], assessed['primary_mechanism_gate'],
          assessed['offset_null_gate'], assessed['recheck_gate'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'inputs', 'reference', 'run', 'report'))
    parser.add_argument('--family', required=True)
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.out, args.family)
    elif args.command == 'inputs':
        prepare_inputs(args.out, args.family, args.task)
    elif args.command == 'reference':
        reference(args.out, args.family, args.task)
    elif args.command == 'run':
        run(args.out, args.family, args.task, args.device)
    else:
        report(args.out, args.family)
