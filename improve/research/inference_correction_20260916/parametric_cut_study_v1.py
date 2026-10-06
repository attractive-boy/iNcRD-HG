"""Freeze, execute, and read-only audit the deterministic two-stage cut study."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np

from improve import parametric_cut_rank_v1 as cut
from improve import parametric_cut_calibration_v1 as calibration
from improve import tabular_foundation as base
from improve import anchor_priority_graph_v1 as reference_io

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT.parent
OUT = ROOT / 'results/parametric_cut_rank_v1_20260916'
PARTITIONS = ROOT / 'results/mechanism_validation_20260915'
HISTORICAL = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs'
KERNELS = ROOT / 'results/non_tabular_20260914/inputs/MiDrug/arrays.npz'
ARMS = ('full', 'first_stage')
TRAIN_KEYS = ('pairs_train', 'y_train', 'folds')


def now():
    return datetime.now(timezone.utc).isoformat()


def entry(split):
    historical = split == 'historical'
    cache = ROOT / ('results/nested_anchor_v2_20260915/historical' if historical
                    else 'results/nested_anchor_v3_20260915/partitions')
    source = HISTORICAL / 'MiDrug/arrays.npz' if historical else PARTITIONS / 'inputs/MiDrug' / split / 'arrays.npz'
    return source, cache


def read(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in keys}


def freeze():
    if OUT.exists():
        raise ValueError('Preserve the existing frozen study')
    unit = base.check_seal(base.read_json(HERE / 'PARAMETRIC_CUT_UNIT_V1.json'))
    scalar = base.check_seal(base.read_json(HERE / 'PARAMETRIC_CUT_CALIBRATION_UNIT_V1.json'))
    if not unit['passed'] or unit['tests_run'] != 4 or not scalar['passed'] or scalar['tests_run'] != 2:
        raise ValueError('Missing implementation checks')
    for record in (unit, scalar):
        for name, digest in record['source_sha256'].items():
            if base.digest(PROJECT / name) != digest:
                raise ValueError('Checked source changed')
    if base.digest(cut.DEFAULT_LIBRARY) != unit['native_binary_sha256']:
        raise ValueError('Native library changed')
    sources = [Path(cut.__file__), Path(calibration.__file__), Path(__file__), Path(base.__file__),
               Path(reference_io.__file__), ROOT / 'test_parametric_cut_rank_v1.py',
               ROOT / 'test_parametric_cut_calibration_v1.py', cut.DEFAULT_LIBRARY,
               HERE / 'hnc_author_bareHPF.c']
    inputs = [KERNELS, HERE / 'PARAMETRIC_CUT_ADAPTATION_V1.md', HERE / 'PARAMETRIC_CUT_UNIT_V1.json',
              HERE / 'PARAMETRIC_CUT_CALIBRATION_UNIT_V1.json', HERE / 'PARAMETRIC_CUT_PROBE_V1.json',
              HERE / '2505_08212.pdf', HERE / 'hnc_KDIR2024.pdf', HERE / 'HNC_AUTHOR_FETCH_V1.json']
    for split in ('historical', '61453', '7919', '104729'):
        source, cache = entry(split)
        inputs += [source, cache / 'protocol.json', cache / 'offsets/MiDrug' / split / 'manifest.json',
                   cache / 'offsets/MiDrug' / split / 'arrays.npz']
        if split == 'historical':
            inputs.append(HISTORICAL / 'LncDrug/arrays.npz')
        else:
            for task in ('MiDrug', 'LncDrug'):
                inputs += [PARTITIONS / 'references' / task / split / name
                           for name in ('scores.npz', 'result.json')]
    p = base.seal(dict(created_at_utc=now(), family='two_stage_parametric_cut', version=1,
        source_sha256={str(path): base.digest(path) for path in sources},
        input_sha256={str(path): base.digest(path) for path in inputs},
        primary='full', controls=['first_stage'], k=5, grid_size=501, folds=5,
        scalar_l2=.01, residual_bound=2., residual_weight=.1,
        query_policy='mean of the five support-fold ranks, calibrated once',
        lnc_policy='exact_reference_identity', negative_budget='support_sampled_zero_count',
        splits=['61453', '7919', '104729'], historical_reference_seeds=[666, 0, 1],
        method_is_deterministic=True, repeated_reference_contexts_are_not_independent_fits=True,
        gate='Positive mean MiDrug AUC/AP vs reference and first stage; >=2/3 positive reference differences per metric; exact Lnc identity; historical then partitions then frozen outer.',
        candidate_performance_seen=False, prior_development_exposure=True,
        original_test_source_access=False, confirmed_final_innovations=0))
    OUT.mkdir(parents=True)
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    base.write_json(OUT / 'protocol.json', p)
    print('FROZEN_PARAMETRIC_CUT', p['sha256'], flush=True)


def checked():
    p = base.check_seal(base.read_json(OUT / 'protocol.json'))
    for key in ('source_sha256', 'input_sha256'):
        for name, digest in p[key].items():
            if base.digest(name) != digest:
                raise ValueError('Frozen artifact changed: ' + name)
    return p


def train(split, p, graph):
    """No evaluation arrays, labels, or reference predictions enter this function."""
    dest = OUT / 'runs' / split
    marker = dest / 'TRAINING.json'
    if marker.exists():
        record = base.check_seal(base.read_json(marker))
        if record['protocol_sha256'] != p['sha256']:
            raise ValueError('Existing training protocol mismatch')
        for name, digest in record['files'].items():
            if base.digest(dest / name) != digest:
                raise ValueError('Saved training artifact changed')
        return record
    dest.mkdir(parents=True, exist_ok=True)
    source, cache = entry(split)
    a = read(source, TRAIN_KEYS)
    offset, manifest = reference_io.offset_arrays(cache, 'MiDrug', split, a)
    ids = a['pairs_train'][0] * 216 + a['pairs_train'][1]
    if len(np.unique(ids)) != len(ids) or not np.array_equal(np.unique(a['folds']), np.arange(5)):
        raise ValueError('Invalid training pair identifiers or folds')
    oof = {arm: np.empty(len(ids)) for arm in ARMS}
    mean = {arm: np.zeros(graph.shape[0]) for arm in ARMS}
    records = []
    for fold in range(p['folds']):
        heldout = a['folds'] == fold
        positive = ids[(~heldout) & (a['y_train'] == 1)]
        if np.intersect1d(positive, ids[heldout]).size:
            raise ValueError('Query-fold seed leakage')
        negative_count = int(np.sum((~heldout) & (a['y_train'] == 0)))
        start = time.perf_counter()
        values = cut.two_stage(graph, positive, negative_count, p['grid_size'])
        seconds = time.perf_counter() - start
        np.savez_compressed(dest / ('cut_f%d.npz' % fold), **values)
        for arm, key in (('full', 'full_rank'), ('first_stage', 'first_rank')):
            oof[arm][heldout] = values[key][ids[heldout]]
            mean[arm] += values[key] / p['folds']
        records.append(dict(fold=fold, seconds=seconds, positive_seeds=len(positive),
            likely_negative_seeds=negative_count, heldout_training_rows=int(heldout.sum()),
            query_seed_overlap=0, first_unique_ranks=len(np.unique(values['first_rank'])),
            second_unique_ranks=len(np.unique(values['full_rank'])),
            negative_boundary_ties=int(values['negative_boundary_ties'])))
        print('CUT_SUPPORT', split, records[-1], flush=True)
    fitted = {arm: calibration.fit(oof[arm], a['y_train'], offset, p['scalar_l2']) for arm in ARMS}
    np.savez_compressed(dest / 'rank_arrays.npz', pairs_train=a['pairs_train'], y_train=a['y_train'],
        folds=a['folds'], anchor_oof=offset, **{arm+'_oof': oof[arm] for arm in ARMS},
        **{arm+'_mean_universe': mean[arm] for arm in ARMS})
    base.write_json(dest / 'calibration.json', fitted)
    record = base.seal(dict(created_at_utc=now(), protocol_sha256=p['sha256'], split=split,
        nested_manifest_sha256=base.digest(manifest), support_records=records,
        native_cut_solves=2*p['folds'], scalar_calibration_fits=2, new_neural_fits=0,
        evaluation_outcomes_used=False,
        files={name: base.digest(dest / name) for name in
               ['cut_f%d.npz' % f for f in range(5)] + ['rank_arrays.npz', 'calibration.json']}))
    base.write_json(marker, record)
    return record


def evaluate(split, p):
    dest = OUT / 'runs' / split
    marker = dest / 'result.json'
    if marker.exists():
        return verify_result(split, p)
    source, _ = entry(split)
    query = read(source, ('pairs_val', 'y_val'))
    rank = read(dest / 'rank_arrays.npz', ['pairs_train'] + [arm+'_mean_universe' for arm in ARMS])
    train_ids = rank['pairs_train'][0] * 216 + rank['pairs_train'][1]
    query_ids = query['pairs_val'][0] * 216 + query['pairs_val'][1]
    if np.intersect1d(train_ids, query_ids).size:
        raise ValueError('Evaluation pairs overlap the training set')
    fitted = base.read_json(dest / 'calibration.json')
    residual = {arm: calibration.predict(rank[arm+'_mean_universe'][query_ids], fitted[arm], p['residual_bound'])
                for arm in ARMS}
    for arm in ARMS:
        replay = calibration.predict(rank[arm+'_mean_universe'][query_ids[::-1]], fitted[arm], p['residual_bound'])[::-1]
        chunked = np.concatenate([calibration.predict(rank[arm+'_mean_universe'][ids], fitted[arm], p['residual_bound'])
                                  for ids in np.array_split(query_ids, 4)])
        np.testing.assert_array_equal(residual[arm], replay)
        np.testing.assert_array_equal(residual[arm], chunked)
    contexts = p['historical_reference_seeds'] if split == 'historical' else [666]
    scores = dict(y=query['y_val'], pairs=query['pairs_val'], **{arm+'_residual': residual[arm] for arm in ARMS})
    metrics, identity = {}, {}
    for seed in contexts:
        if split == 'historical':
            anchor = read(source, ('anchor_'+str(seed),))['anchor_'+str(seed)]
            ln = read(HISTORICAL / 'LncDrug/arrays.npz', ('y_val', 'pairs_val', 'anchor_'+str(seed)))
            ly, lp, la = ln['y_val'], ln['pairs_val'], ln['anchor_'+str(seed)]
        else:
            ref = read(PARTITIONS / 'references/MiDrug' / split / 'scores.npz', ('y', 'pairs', 'anchor'))
            np.testing.assert_array_equal(ref['y'], query['y_val'])
            np.testing.assert_array_equal(ref['pairs'], query['pairs_val'])
            anchor = ref['anchor']
            ln = read(PARTITIONS / 'references/LncDrug' / split / 'scores.npz', ('y', 'pairs', 'anchor'))
            ly, lp, la = ln['y'], ln['pairs'], ln['anchor']
        key = str(seed)
        scores['anchor_'+key] = anchor
        metrics[key] = dict(reference=base.metrics(query['y_val'], anchor, query['pairs_val']))
        for arm in ARMS:
            scores[arm+'_'+key] = anchor + p['residual_weight'] * residual[arm]
            metrics[key][arm] = base.metrics(query['y_val'], scores[arm+'_'+key], query['pairs_val'])
        np.savez_compressed(dest / ('lnc_identity_'+key+'.npz'), y=ly, pairs=lp, anchor=la, full=la.copy(), first_stage=la.copy())
        identity[key] = base.metrics(ly, la, lp)
    np.savez_compressed(dest / 'scores.npz', **scores)
    files = ['scores.npz', 'TRAINING.json'] + ['lnc_identity_'+str(seed)+'.npz' for seed in contexts]
    result = base.seal(dict(created_at_utc=now(), split=split, protocol_sha256=p['sha256'],
        metrics=metrics, lnc_identity_metrics=identity, contexts=contexts,
        context_count_is_not_fit_count=True, query_order_and_batch_max_error=0.,
        files={name: base.digest(dest / name) for name in files}))
    base.write_json(marker, result)
    return verify_result(split, p)


def verify_result(split, p):
    dest = OUT / 'runs' / split
    r = base.check_seal(base.read_json(dest / 'result.json'))
    if r['protocol_sha256'] != p['sha256']:
        raise ValueError('Result protocol mismatch')
    for name, digest in r['files'].items():
        if base.digest(dest / name) != digest:
            raise ValueError('Saved score evidence changed')
    saved = read(dest / 'rank_arrays.npz', ('pairs_train', 'y_train', 'folds', 'anchor_oof',
                 'full_oof', 'first_stage_oof', 'full_mean_universe', 'first_stage_mean_universe'))
    train_ids = saved['pairs_train'][0] * 216 + saved['pairs_train'][1]
    mean = {arm: np.zeros(605*216) for arm in ARMS}
    for fold in range(5):
        heldout = saved['folds'] == fold
        values = read(dest / ('cut_f%d.npz' % fold), ('positive', 'negative', 'first_rank', 'full_rank',
                      'first_breakpoints', 'second_breakpoints'))
        np.testing.assert_array_equal(values['positive'], np.unique(train_ids[(~heldout) & (saved['y_train'] == 1)]))
        if np.intersect1d(values['positive'], train_ids[heldout]).size:
            raise ValueError('Saved cut contains a held-out positive seed')
        if len(values['negative']) != int(np.sum((~heldout) & (saved['y_train'] == 0))):
            raise ValueError('Saved likely-negative budget changed')
        for arm, key, bk in (('full', 'full_rank', 'second_breakpoints'),
                             ('first_stage', 'first_rank', 'first_breakpoints')):
            np.testing.assert_array_equal(values[key], cut.rank_from_breakpoints(values[bk], p['grid_size']))
            np.testing.assert_array_equal(saved[arm+'_oof'][heldout], values[key][train_ids[heldout]])
            mean[arm] += values[key] / p['folds']
    fitted = base.read_json(dest / 'calibration.json')
    for arm in ARMS:
        np.testing.assert_array_equal(mean[arm], saved[arm+'_mean_universe'])
        x = (saved[arm+'_oof'] - fitted[arm]['mean']) / fitted[arm]['scale']
        _, gradient = calibration.objective(np.array([fitted[arm]['beta'], fitted[arm]['intercept']]),
                                            x, saved['y_train'], saved['anchor_oof'], p['scalar_l2'])
        if abs(gradient).max() > 1e-7:
            raise ValueError('Saved calibration is not stationary')
    with np.load(dest / 'scores.npz', allow_pickle=False) as z:
        query_ids = z['pairs'][0] * 216 + z['pairs'][1]
        for arm in ARMS:
            expected = calibration.predict(mean[arm][query_ids], fitted[arm], p['residual_bound'])
            np.testing.assert_array_equal(expected, z[arm+'_residual'])
        for seed in r['contexts']:
            for arm in ('reference',)+ARMS:
                key = ('anchor' if arm == 'reference' else arm)+'_'+str(seed)
                if arm != 'reference':
                    np.testing.assert_array_equal(z[key], z['anchor_'+str(seed)] + p['residual_weight'] * z[arm+'_residual'])
                actual = base.metrics(z['y'], z[key], z['pairs'])
                if actual != r['metrics'][str(seed)][arm]:
                    raise ValueError('Stored metrics disagree with raw scores')
    for seed in r['contexts']:
        with np.load(dest / ('lnc_identity_'+str(seed)+'.npz'), allow_pickle=False) as z:
            for arm in ARMS:
                np.testing.assert_array_equal(z[arm], z['anchor'])
            if base.metrics(z['y'], z['anchor'], z['pairs']) != r['lnc_identity_metrics'][str(seed)]:
                raise ValueError('Identity metrics changed')
    return r


def assess(records, p, name):
    target = OUT / (name+'.json')
    cells = [cell for r in records for cell in r['metrics'].values()]
    means = {arm: {m: float(np.mean([cell[arm][m] for cell in cells])) for m in ('auc', 'ap')}
             for arm in ('reference',)+ARMS}
    delta = {arm: {m: means['full'][m]-means[arm][m] for m in ('auc', 'ap')}
             for arm in ('reference', 'first_stage')}
    positive = {m: sum(cell['full'][m] > cell['reference'][m] for cell in cells) for m in ('auc', 'ap')}
    performance = all(delta['reference'][m] > 0 and positive[m] >= 2 for m in ('auc', 'ap'))
    mechanism = all(delta['first_stage'][m] > 0 for m in ('auc', 'ap'))
    fields = dict(protocol_sha256=p['sha256'], stage=name,
        mean_metrics=means, differences=delta, positive_reference_contexts=positive, contexts=len(cells),
        historical_contexts_are_not_independent_method_fits=(name == 'HISTORICAL_ASSESSMENT'),
        performance_gate=performance, primary_mechanism_gate=mechanism, lnc_exact_identity=True,
        declared_gate=performance and mechanism, confirmed_final_innovations=0,
        result_sha256={r['split']: base.digest(OUT / 'runs' / r['split'] / 'result.json') for r in records})
    if target.exists():
        stored = base.check_seal(base.read_json(target))
        if any(stored.get(key) != value for key, value in fields.items()):
            raise ValueError('Stored assessment differs from recomputed metrics and gates')
        return stored
    result = base.seal(dict(created_at_utc=now(), **fields))
    base.write_json(target, result)
    print('PARAMETRIC_CUT_ASSESSMENT', name, means, delta, positive, result['declared_gate'], flush=True)
    return result


def run():
    p = checked()
    kernels = read(KERNELS, ('kr', 'kd'))
    graph = cut.cartesian_graph(kernels['kr'], kernels['kd'], p['k'])
    train('historical', p, graph)
    historical = assess([evaluate('historical', p)], p, 'HISTORICAL_ASSESSMENT')
    partition = None
    if historical['declared_gate']:
        results = []
        for split in p['splits']:
            train(split, p, graph)
            results.append(evaluate(split, p))
        partition = assess(results, p, 'PARTITION_ASSESSMENT')
    target = HERE / 'PARAMETRIC_CUT_QUEUE_RESULT_V1.json'
    if not target.exists():
        trains = [base.read_json(path) for path in (OUT / 'runs').glob('*/TRAINING.json')]
        base.write_json(target, base.seal(dict(created_at_utc=now(), protocol_sha256=p['sha256'],
            historical_gate=historical['declared_gate'],
            partition_gate=partition['declared_gate'] if partition else None,
            native_cut_solves=sum(r['native_cut_solves'] for r in trains),
            scalar_calibration_fits=sum(r['scalar_calibration_fits'] for r in trains),
            new_neural_fits=0, new_outer_fits=0, confirmed_final_innovations=0,
            next_step='freeze_outer_recheck' if partition and partition['declared_gate'] else 'stop_this_version')))


def audit():
    p = checked()
    records = [verify_result(path.parent.name, p) for path in sorted((OUT / 'runs').glob('*/result.json'))]
    for r in records:
        train(r['split'], p, None)
    assess([r for r in records if r['split'] == 'historical'], p, 'HISTORICAL_ASSESSMENT')
    if len(records) > 1:
        assess([r for r in records if r['split'] != 'historical'], p, 'PARTITION_ASSESSMENT')
    output = HERE / 'PARAMETRIC_CUT_CONSOLIDATION_V1.json'
    if output.exists():
        raise ValueError('Preserve the existing consolidation')
    base.write_json(output, base.seal(dict(created_at_utc=now(), scores_recomputed=True,
        protocol_sha256=p['sha256'], results=records,
        queue=base.check_seal(base.read_json(HERE / 'PARAMETRIC_CUT_QUEUE_RESULT_V1.json')),
        original_test_source_access=False, confirmed_final_innovations=0)))
    print('PARAMETRIC_CUT_READ_ONLY_AUDIT', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'audit'))
    globals()[parser.parse_args().command]()
