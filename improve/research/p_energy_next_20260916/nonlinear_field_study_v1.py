"""Frozen nonlinear pair-field study; numerical solves precede score access."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import sparse
from scipy.special import expit
import torch

from improve import nonlinear_pair_field_v1 as core
from improve import anchor_priority_graph_v1 as reference
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = reference.ROOT
OUT = ROOT / 'results/nonlinear_pair_field_v1_20260916'
PROBE = ROOT / 'results/nonlinear_pair_field_smoke_v1_20260916'
PARTITIONS = ROOT / 'results/mechanism_validation_20260915'
HISTORICAL = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs'
ARMS = core.KINDS


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, keys):
    return reference.nested.load_selected(path, keys)


def entry(split):
    historical = split == 'historical'
    cache = ROOT / ('results/nested_anchor_v2_20260915/historical' if historical
                    else 'results/nested_anchor_v3_20260915/partitions')
    source = HISTORICAL / 'MiDrug/arrays.npz' if historical else PARTITIONS / 'inputs/MiDrug' / split / 'arrays.npz'
    return source, cache


def training_arrays(split):
    source, cache = entry(split)
    a = read(source, reference.nested.TRAIN_KEYS)
    a['anchor_oof'], manifest = reference.offset_arrays(cache, 'MiDrug', split, a)
    if split == 'historical':
        stored = read(source, ('anchor_oof',))['anchor_oof']
        np.testing.assert_array_equal(a['anchor_oof'], stored)
    ids = a['pairs_train'][0] * int(a['nd']) + a['pairs_train'][1]
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate training pairs')
    return a, manifest


def freeze():
    if OUT.exists():
        raise ValueError('Preserve existing nonlinear-field study')
    unit_path, probe_path = HERE / 'NONLINEAR_FIELD_UNIT_V1.json', HERE / 'NONLINEAR_FIELD_SMOKE_V1.json'
    unit, probe = [base.check_seal(base.read_json(path)) for path in (unit_path, probe_path)]
    pipeline_path = HERE / 'NONLINEAR_FIELD_STUDY_UNIT_V1.json'
    pipeline = base.check_seal(base.read_json(pipeline_path))
    if not unit['success'] or unit['tests'] != 5 or len(probe['checks']) != 2:
        raise ValueError('Implementation and real-size numerical checks are required')
    if not pipeline['success'] or pipeline['tests'] != 2 or not pipeline['synthetic_data_only']:
        raise ValueError('Synthetic adoption and corruption checks are required')
    if probe['performance_metrics_computed'] or probe['evaluation_labels_read']:
        raise ValueError('Only outcome-blind training solutions may be adopted')
    for record in (unit, probe, pipeline):
        for path, digest in record['source_sha256'].items():
            if base.digest(path) != digest:
                raise ValueError('Checked implementation changed: ' + path)
    for path, digest in probe['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Probe input changed: ' + path)
    for name, digest in probe['files'].items():
        if base.digest(PROBE / name) != digest:
            raise ValueError('Probe solution changed: ' + name)
    if probe['parameters'] != core.parameters():
        raise ValueError('Probe and formal parameters differ')
    sources = [Path(__file__), Path(core.__file__), Path(base.__file__), Path(reference.__file__),
               ROOT / 'nested_anchor_inputs_v2.py', ROOT / 'nested_anchor_fit_v2.py',
               ROOT / 'residual_graph_experiment.py', ROOT / 'non_tabular_experiment.py',
               ROOT / 'graph_prediction_precision.py', ROOT / 'test_nonlinear_pair_field_v1.py',
               ROOT / 'test_nonlinear_field_study_v1.py',
               HERE / 'nonlinear_field_smoke_v1.py']
    inputs = [unit_path, probe_path, pipeline_path, HERE / 'NONLINEAR_FIELD_GRAPH_PROBE_V1.json',
              HERE / 'P_ENERGY_SELECTION_V1.json', HERE / 'NONLINEAR_FIELD_ADAPTATION_V1.md',
              HERE / 'P_ENERGY_FULLTEXT_V1.json', HERE / 'P_ENERGY_AUTHOR_SOURCE_V1.json',
              HERE / 'continuation_solver_v1.pdf', HERE / 'continuation_solver_v1.txt']
    inputs += [PROBE / name for name in probe['files']]
    for split in ('historical', '61453', '7919', '104729'):
        source, cache = entry(split)
        inputs += [source, cache / 'protocol.json', cache / 'offsets/MiDrug' / split / 'manifest.json',
                   cache / 'offsets/MiDrug' / split / 'arrays.npz']
        if split == 'historical':
            inputs.append(HISTORICAL / 'LncDrug/arrays.npz')
        else:
            for task in ('MiDrug', 'LncDrug'):
                inputs += [PARTITIONS / 'references' / task / split / name for name in ('scores.npz', 'result.json')]
    p = base.seal(dict(created_at_utc=now(), family='nonlinear_pair_field', version=1,
        parameters=core.parameters(), primary='full', controls=['quadratic'],
        splits=['61453', '7919', '104729'], historical_reference_seeds=[666, 0, 1],
        historical_training_policy='adopt both exact training-only probe solutions; no refit or reselection',
        adopted_probe_sha256=probe['sha256'], method_is_deterministic=True,
        reference_contexts_are_not_independent_method_fits=True,
        gate='Positive mean MiDrug AUC/AP vs reference and quadratic control; each reference metric improves in >=2/3 contexts; exact Lnc identity; historical then partitions then frozen outer.',
        candidate_performance_seen=False, prior_development_exposure=True,
        original_test_source_access=False, confirmed_final_innovations=0,
        source_sha256={str(path): base.digest(path) for path in sources},
        input_sha256={str(path): base.digest(path) for path in inputs}))
    OUT.mkdir(parents=True)
    for path in sources:
        destination = OUT / 'source_snapshot' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    base.write_json(OUT / 'protocol.json', p)
    print('NONLINEAR_FIELD_FROZEN', p['sha256'], flush=True)


def checked():
    p = base.check_seal(base.read_json(OUT / 'protocol.json'))
    if p['parameters'] != core.parameters():
        raise ValueError('Frozen numerical parameters changed')
    for key in ('source_sha256', 'input_sha256'):
        for path, digest in p[key].items():
            if base.digest(path) != digest:
                raise ValueError('Frozen artifact changed: ' + path)
    return p


def verify_training(split, p):
    """Independent NumPy/SciPy energy and stationarity, without fitting."""
    dest = OUT / 'runs' / split
    record = base.check_seal(base.read_json(dest / 'TRAINING.json'))
    if record['protocol_sha256'] != p['sha256'] or record['split'] != split:
        raise ValueError('Training protocol or split mismatch')
    for name, digest in record['files'].items():
        if base.digest(dest / name) != digest:
            raise ValueError('Saved numerical evidence changed: ' + name)
    a, manifest = training_arrays(split)
    if record['nested_manifest_sha256'] != base.digest(manifest):
        raise ValueError('Nested training offset changed')
    saved = read(dest / 'training_arrays.npz', ('pairs_train', 'y_train', 'folds', 'anchor_oof'))
    for key in saved:
        np.testing.assert_array_equal(saved[key], a[key])
    graph, graph_audit = core.training_graph(a, p['parameters'])
    if base.read_json(dest / 'graph.json') != graph_audit:
        raise ValueError('Saved pair graph disagrees with fixed attributes')
    edges = sparse.triu(graph, 1, format='coo')
    train_ids = a['pairs_train'][0] * int(a['nd']) + a['pairs_train'][1]
    traces = base.read_json(dest / 'traces.json')
    fields = read(dest / 'fields.npz', ARMS)
    checks = []
    params = p['parameters']
    for arm, exponent in (('quadratic', 2.), ('full', 3.)):
        field = fields[arm]
        if field.shape != (graph.shape[0],) or not np.isfinite(field).all():
            raise ValueError('Invalid stored full-universe field')
        difference = field[edges.row] - field[edges.col]
        flow = params['regularization'] * edges.data * np.abs(difference) ** (exponent - 2.) * difference
        gradient = params['ridge'] * field + np.bincount(edges.row, flow, minlength=len(field))
        gradient -= np.bincount(edges.col, flow, minlength=len(field))
        logits = a['anchor_oof'] + field[train_ids]
        gradient[train_ids] += expit(logits) - a['y_train']
        objective = np.sum(np.logaddexp(0., logits) - a['y_train'] * logits)
        objective += params['regularization'] / exponent * np.sum(edges.data * np.abs(difference) ** exponent)
        objective += .5 * params['ridge'] * np.sum(field ** 2)
        maximum = float(np.max(np.abs(gradient)))
        if maximum > params['gradient_tolerance'] + 1e-12:
            raise ValueError('Saved full-space field is not stationary')
        if abs(objective - traces[arm][-1]['objective']) > 1e-8:
            raise ValueError('Independent energy disagrees with saved trace')
        objectives = np.array([row['objective'] for row in traces[arm]])
        if np.any(np.diff(objectives) > 1e-8):
            raise ValueError('Saved Armijo trajectory is not monotone')
        for row in traces[arm][:-1]:
            if row['directional_derivative'] >= 0 or row['cg']['residual_norm'] > max(
                    4. * row['cg']['tolerance'], params['cg_absolute_tolerance']):
                raise ValueError('Saved inner solve failed its declared gate')
        known = np.zeros(len(field), dtype=bool)
        known[train_ids] = True
        slack = params['gradient_tolerance'] / params['ridge']
        if (field[~known].min() < min(field[train_ids].min(), 0.) - slack
                or field[~known].max() > max(field[train_ids].max(), 0.) + slack):
            raise ValueError('Unlabeled field violates the discrete maximum principle')
        checks.append(dict(arm=arm, objective=float(objective), maximum_gradient=maximum,
            newton_steps=len(traces[arm]) - 1,
            cg_steps=sum(row.get('cg', {}).get('steps', 0) for row in traces[arm])))
    if split == 'historical':
        probe = base.check_seal(base.read_json(HERE / 'NONLINEAR_FIELD_SMOKE_V1.json'))
        if record['adopted_probe_sha256'] != p['adopted_probe_sha256'] or probe['sha256'] != p['adopted_probe_sha256']:
            raise ValueError('Adopted probe identity mismatch')
        if record['new_numerical_solves'] != 0 or record['adopted_numerical_solves'] != 2:
            raise ValueError('Historical adoption was counted as a new fit')
        for name, digest in probe['files'].items():
            if base.digest(PROBE / name) != digest or base.digest(dest / name) != digest:
                raise ValueError('Historical fields or traces differ from their source')
    elif record['new_numerical_solves'] != 2 or record['adopted_numerical_solves'] != 0:
        raise ValueError('Partition numerical solve count mismatch')
    return record, checks


def train(split, p):
    dest = OUT / 'runs' / split
    if (dest / 'TRAINING.json').exists():
        return verify_training(split, p)[0]
    if dest.exists():
        raise ValueError('Preserve an incomplete numerical run before recovery')
    dest.mkdir(parents=True)
    a, manifest = training_arrays(split)
    start = time.monotonic()
    if split == 'historical':
        for name in ('fields.npz', 'traces.json', 'graph.json'):
            shutil.copy2(PROBE / name, dest / name)
        new_solves, adopted_solves = 0, 2
    else:
        fields, traces, graph_audit, operator = core.fit(a, 'cuda:1', p['parameters'])
        np.savez_compressed(dest / 'fields.npz', **{arm: fields[arm].cpu().numpy() for arm in ARMS})
        base.write_json(dest / 'traces.json', traces)
        base.write_json(dest / 'graph.json', graph_audit)
        del fields, operator
        torch.cuda.empty_cache()
        new_solves, adopted_solves = 2, 0
    np.savez_compressed(dest / 'training_arrays.npz', **{key: a[key]
        for key in ('pairs_train', 'y_train', 'folds', 'anchor_oof')})
    record = base.seal(dict(created_at_utc=now(), split=split, protocol_sha256=p['sha256'],
        nested_manifest_sha256=base.digest(manifest), new_numerical_solves=new_solves,
        adopted_numerical_solves=adopted_solves, new_neural_fits=0, evaluation_outcomes_used=False,
        adopted_probe_sha256=p['adopted_probe_sha256'] if split == 'historical' else None,
        seconds=time.monotonic() - start, files={name: base.digest(dest / name)
            for name in ('fields.npz', 'traces.json', 'graph.json', 'training_arrays.npz')}))
    base.write_json(dest / 'TRAINING.json', record)
    verified, checks = verify_training(split, p)
    print('NONLINEAR_FIELD_TRAINING', split, new_solves, adopted_solves, checks, flush=True)
    return verified


def reference_scores(split, seed, query):
    if split == 'historical':
        source, _ = entry(split)
        anchor = read(source, ('anchor_' + str(seed),))['anchor_' + str(seed)]
        ln = read(HISTORICAL / 'LncDrug/arrays.npz', ('y_val', 'pairs_val', 'anchor_' + str(seed)))
        ly, lp, la = ln['y_val'], ln['pairs_val'], ln['anchor_' + str(seed)]
    else:
        ref = read(PARTITIONS / 'references/MiDrug' / split / 'scores.npz', ('y', 'pairs', 'anchor'))
        np.testing.assert_array_equal(ref['y'], query['y_val'])
        np.testing.assert_array_equal(ref['pairs'], query['pairs_val'])
        anchor = ref['anchor']
        ln = read(PARTITIONS / 'references/LncDrug' / split / 'scores.npz', ('y', 'pairs', 'anchor'))
        ly, lp, la = ln['y'], ln['pairs'], ln['anchor']
    return anchor, dict(y=ly, pairs=lp, anchor=la)


def evaluate(split, p):
    dest = OUT / 'runs' / split
    if (dest / 'result.json').exists():
        return verify_result(split, p)
    # No evaluation outcome is loaded before the complete field is checked.
    verify_training(split, p)
    source, _ = entry(split)
    query = read(source, ('pairs_val', 'y_val', 'nd'))
    fields = read(dest / 'fields.npz', ARMS)
    training = read(dest / 'training_arrays.npz', ('pairs_train',))
    train_ids = training['pairs_train'][0] * int(query['nd']) + training['pairs_train'][1]
    query_ids = query['pairs_val'][0] * int(query['nd']) + query['pairs_val'][1]
    if np.intersect1d(train_ids, query_ids).size:
        raise ValueError('Evaluation pairs overlap fitting observations')
    params = p['parameters']
    residual = {arm: np.clip(fields[arm][query_ids], -params['residual_clip'], params['residual_clip'])
                for arm in ARMS}
    for arm in ARMS:
        reverse = np.clip(fields[arm][query_ids[::-1]], -params['residual_clip'], params['residual_clip'])[::-1]
        chunks = np.concatenate([np.clip(fields[arm][ids], -params['residual_clip'], params['residual_clip'])
                                 for ids in np.array_split(query_ids, 5)])
        np.testing.assert_array_equal(residual[arm], reverse)
        np.testing.assert_array_equal(residual[arm], chunks)
    contexts = p['historical_reference_seeds'] if split == 'historical' else [666]
    scores = dict(y=query['y_val'], pairs=query['pairs_val'], **{arm + '_residual': residual[arm] for arm in ARMS})
    metrics, identity = {}, {}
    for seed in contexts:
        anchor, ln = reference_scores(split, seed, query)
        key = str(seed)
        scores['anchor_' + key] = anchor
        metrics[key] = dict(reference=base.metrics(query['y_val'], anchor, query['pairs_val']))
        for arm in ARMS:
            scores[arm + '_' + key] = anchor + params['strong_weight'] * residual[arm]
            metrics[key][arm] = base.metrics(query['y_val'], scores[arm + '_' + key], query['pairs_val'])
        np.savez_compressed(dest / ('lnc_identity_' + key + '.npz'), **ln,
            **{arm: ln['anchor'].copy() for arm in ARMS})
        identity[key] = base.metrics(ln['y'], ln['anchor'], ln['pairs'])
    np.savez_compressed(dest / 'scores.npz', **scores)
    files = ['scores.npz', 'TRAINING.json'] + ['lnc_identity_' + str(seed) + '.npz' for seed in contexts]
    result = base.seal(dict(created_at_utc=now(), split=split, protocol_sha256=p['sha256'],
        metrics=metrics, lnc_identity_metrics=identity, contexts=contexts,
        context_count_is_not_fit_count=True, query_order_and_batch_max_error=0.,
        files={name: base.digest(dest / name) for name in files}))
    base.write_json(dest / 'result.json', result)
    return verify_result(split, p)


def verify_result(split, p):
    dest = OUT / 'runs' / split
    result = base.check_seal(base.read_json(dest / 'result.json'))
    if result['protocol_sha256'] != p['sha256'] or result['split'] != split:
        raise ValueError('Result protocol or split mismatch')
    for name, digest in result['files'].items():
        if base.digest(dest / name) != digest:
            raise ValueError('Saved prediction evidence changed: ' + name)
    fields = read(dest / 'fields.npz', ARMS)
    source, _ = entry(split)
    query = read(source, ('pairs_val', 'y_val', 'nd'))
    params = p['parameters']
    with np.load(dest / 'scores.npz', allow_pickle=False) as scores:
        np.testing.assert_array_equal(scores['pairs'], query['pairs_val'])
        np.testing.assert_array_equal(scores['y'], query['y_val'])
        ids = scores['pairs'][0] * int(query['nd']) + scores['pairs'][1]
        for arm in ARMS:
            np.testing.assert_array_equal(scores[arm + '_residual'], np.clip(fields[arm][ids],
                -params['residual_clip'], params['residual_clip']))
        for seed in result['contexts']:
            anchor, ln = reference_scores(split, seed, query)
            key = str(seed)
            np.testing.assert_array_equal(scores['anchor_' + key], anchor)
            for arm in ('reference',) + ARMS:
                score_key = ('anchor' if arm == 'reference' else arm) + '_' + key
                if arm != 'reference':
                    np.testing.assert_array_equal(scores[score_key], anchor + params['strong_weight'] * scores[arm + '_residual'])
                if base.metrics(scores['y'], scores[score_key], scores['pairs']) != result['metrics'][key][arm]:
                    raise ValueError('Stored metric disagrees with score replay')
            with np.load(dest / ('lnc_identity_' + key + '.npz'), allow_pickle=False) as identity:
                for name, value in ln.items():
                    np.testing.assert_array_equal(identity[name], value)
                for arm in ARMS:
                    np.testing.assert_array_equal(identity[arm], ln['anchor'])
                if base.metrics(ln['y'], ln['anchor'], ln['pairs']) != result['lnc_identity_metrics'][key]:
                    raise ValueError('Lnc identity metric mismatch')
    return result


def assess(records, p, name):
    cells = [cell for result in records for cell in result['metrics'].values()]
    if len(cells) != 3:
        raise ValueError('A complete stage requires exactly three contexts')
    means = {arm: {metric: float(np.mean([cell[arm][metric] for cell in cells])) for metric in ('auc', 'ap')}
             for arm in ('reference',) + ARMS}
    differences = {arm: {metric: means['full'][metric] - means[arm][metric] for metric in ('auc', 'ap')}
                   for arm in ('reference',) + tuple(p['controls'])}
    positive = {metric: sum(cell['full'][metric] > cell['reference'][metric] for cell in cells)
                for metric in ('auc', 'ap')}
    performance = all(differences['reference'][metric] > 0 and positive[metric] >= 2 for metric in ('auc', 'ap'))
    mechanism = all(differences[control][metric] > 0 for control in p['controls'] for metric in ('auc', 'ap'))
    fields = dict(protocol_sha256=p['sha256'], stage=name, mean_metrics=means, differences=differences,
        positive_reference_contexts=positive, contexts=len(cells),
        historical_contexts_are_not_independent_method_fits=(name == 'HISTORICAL_ASSESSMENT'),
        performance_gate=performance, primary_mechanism_gate=mechanism, lnc_exact_identity=True,
        declared_gate=performance and mechanism, confirmed_final_innovations=0,
        result_sha256={result['split']: base.digest(OUT / 'runs' / result['split'] / 'result.json') for result in records})
    path = OUT / (name + '.json')
    if path.exists():
        saved = base.check_seal(base.read_json(path))
        if any(saved.get(key) != value for key, value in fields.items()):
            raise ValueError('Assessment changed on read-only recomputation')
        return saved
    result = base.seal(dict(created_at_utc=now(), **fields))
    base.write_json(path, result)
    print('NONLINEAR_FIELD_ASSESSMENT', name, means, differences, positive, result['declared_gate'], flush=True)
    return result


def run():
    p = checked()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train('historical', p)
    history = assess([evaluate('historical', p)], p, 'HISTORICAL_ASSESSMENT')
    partitions = None
    if history['declared_gate']:
        results = []
        for split in p['splits']:
            train(split, p)
            results.append(evaluate(split, p))
        partitions = assess(results, p, 'PARTITION_ASSESSMENT')
    queue_path = HERE / 'NONLINEAR_FIELD_QUEUE_RESULT_V1.json'
    training = [base.check_seal(base.read_json(path)) for path in sorted((OUT / 'runs').glob('*/TRAINING.json'))]
    new_solves = sum(record['new_numerical_solves'] for record in training)
    fields = dict(protocol_sha256=p['sha256'], historical_gate=history['declared_gate'],
        partition_gate=partitions['declared_gate'] if partitions else None,
        new_formal_numerical_solves=new_solves,
        adopted_training_only_numerical_solves=sum(record['adopted_numerical_solves'] for record in training),
        total_distinct_numerical_solves_including_probe=new_solves + 2,
        new_neural_fits=0, new_outer_fits=0, cumulative_formal_partition_neural_fits=333,
        confirmed_final_innovations=0,
        next_step='freeze_outer_recheck' if partitions and partitions['declared_gate'] else 'stop_this_version')
    if queue_path.exists():
        previous = base.check_seal(base.read_json(queue_path))
        if any(previous.get(key) != value for key, value in fields.items()):
            raise ValueError('Queue accounting changed')
    else:
        base.write_json(queue_path, base.seal(dict(created_at_utc=now(), **fields)))


def audit():
    p = checked()
    records, stationary = [], {}
    for path in sorted((OUT / 'runs').glob('*/result.json')):
        split = path.parent.name
        _, stationary[split] = verify_training(split, p)
        records.append(verify_result(split, p))
    assess([result for result in records if result['split'] == 'historical'], p, 'HISTORICAL_ASSESSMENT')
    if len(records) > 1:
        assess([result for result in records if result['split'] != 'historical'], p, 'PARTITION_ASSESSMENT')
    path = HERE / 'NONLINEAR_FIELD_CONSOLIDATION_V1.json'
    if path.exists():
        raise ValueError('Preserve completed read-only consolidation')
    queue = base.check_seal(base.read_json(HERE / 'NONLINEAR_FIELD_QUEUE_RESULT_V1.json'))
    base.write_json(path, base.seal(dict(created_at_utc=now(), protocol_sha256=p['sha256'],
        source_sha256=base.digest(__file__), results=records, independent_stationarity=stationary,
        all_metrics_recomputed=True, adopted_solutions_bitwise_verified=True,
        saved_field_objective_and_maximum_principle_verified=True, exact_lnc_identity=True,
        added_by_this_audit_fits=0, queue=queue, original_test_source_access=False,
        confirmed_final_innovations=0)))
    print('NONLINEAR_FIELD_READ_ONLY_AUDIT', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'audit'))
    globals()[parser.parse_args().command]()
