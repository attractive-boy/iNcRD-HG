"""Frozen historical -> partition validation of relational demonstration inference."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from improve import kgpfn_context_v1 as core
from improve import kgpfn_runtime_v1 as runtime
from improve import parametric_cut_calibration_v1 as calibration
from improve import anchor_priority_graph_v1 as reference_io
from improve import tabular_foundation as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/kgpfn_context_v1_20260916'
HISTORICAL = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs'
PARTITIONS = ROOT / 'results/mechanism_validation_20260915'
ARMS = core.ARMS


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in keys}


def entry(split):
    historical = split == 'historical'
    source = HISTORICAL / 'MiDrug/arrays.npz' if historical else PARTITIONS / 'inputs/MiDrug' / split / 'arrays.npz'
    cache = ROOT / ('results/nested_anchor_v2_20260915/historical' if historical
                    else 'results/nested_anchor_v3_20260915/partitions')
    return source, cache


def freeze():
    if OUT.exists():
        raise ValueError('Use a new study version')
    unit = base.check_seal(base.read_json(HERE / 'KGPFN_UNIT_V1.json'))
    probe = base.check_seal(base.read_json(HERE / 'KGPFN_PROBE_V1.json'))
    if not unit['passed'] or unit['tests_run'] != 5 or not probe['passed']:
        raise ValueError('Missing runtime verification')
    for record in (unit, probe):
        for name, digest in record['source_sha256'].items():
            if base.digest(name) != digest:
                raise ValueError('Checked implementation changed')
    sources = [Path(__file__), Path(core.__file__), Path(runtime.__file__), Path(calibration.__file__),
               Path(reference_io.__file__), Path(base.__file__), ROOT / 'test_kgpfn_runtime_v1.py']
    external = sorted(runtime.SOURCE.rglob('*.py'))
    inputs = [HERE / name for name in ('KGPFN_ADAPTATION_V1.md', 'KGPFN_UNIT_V1.json',
        'KGPFN_PROBE_V1.json', 'KGPFN_CHECKPOINT_V1.json', 'KGPFN_CHECKPOINT_INSPECTION_V1.json',
        'KGPFN_SOURCE_CHECKOUT_V1.json', 'kgpfn_pdf_v1.pdf')]
    inputs += [runtime.CHECKPOINT, runtime.SOURCE / 'config/script/train_all.yaml',
               runtime.SOURCE / 'config/script/test.yaml']
    for split in ('historical', '61453', '7919', '104729'):
        source, cache = entry(split)
        inputs += [source, cache / 'protocol.json', cache / 'offsets/MiDrug' / split / 'manifest.json',
                   cache / 'offsets/MiDrug' / split / 'arrays.npz']
        if split == 'historical':
            inputs.append(HISTORICAL / 'LncDrug/arrays.npz')
        else:
            for task in ('MiDrug', 'LncDrug'):
                inputs += [PARTITIONS / 'references' / task / split / name for name in ('scores.npz', 'result.json')]
    p = base.seal(dict(created_at_utc=now(), family='relational_demonstration_inference', version=1,
        source_sha256={str(path): base.digest(path) for path in sources},
        author_source_sha256={str(path): base.digest(path) for path in external},
        input_sha256={str(path): base.digest(path) for path in inputs},
        primary='full', controls=['shuffled_context', 'ultra_only'],
        splits=['61453', '7919', '104729'], historical_reference_seeds=[666, 0, 1],
        context_seeds=list(core.CONTEXT_SEEDS), positives=20, negatives=80, folds=5,
        scalar_l2=.01, residual_bound=2., residual_weight=.1,
        lnc_policy='exact_reference_identity', method_is_deterministic=True,
        checkpoint_items=368, pretrained_parameter_elements=26668874, trainable_parameter_elements=0,
        graph='Remove all query-fold and all three contexts target facts before inverse arcs and relation graph.',
        training='OOF scores from own excluded fold; no local neural fitting; one two-parameter calibration per arm.',
        inference='Mean three contexts per graph, then mean five graph raw scores; calibration once; bounded residual.',
        control='Shuffle already corrected labels, retaining the exact label multiset; ULTRA uses the same masked graph.',
        gate='Mean MiDrug AUC/AP > reference and both controls; each metric improves in >=2/3 reference contexts or partitions; exact Lnc identity. Historical then partitions then frozen outer.',
        context_draws_are_not_model_initializations=True, candidate_performance_seen=False,
        previous_development_exposure=True, original_test_source_access=False,
        pretrained_corpus_full_manifest_available=False,
        innovation_scope='Conditional on both controls, distinct relational label-conditioned inference; overlaps broad graph/PFN families; compatibility and calibration do not count.',
        confirmed_final_innovations=0))
    OUT.mkdir(parents=True)
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    base.write_json(OUT / 'protocol.json', p)
    print('FROZEN_KGPFN', p['sha256'], flush=True)


def checked():
    p = base.check_seal(base.read_json(OUT / 'protocol.json'))
    for key in ('source_sha256', 'author_source_sha256', 'input_sha256'):
        for name, digest in p[key].items():
            if base.digest(name) != digest:
                raise ValueError('Frozen file changed: ' + name)
    return p


def checked_record(path, p):
    record = base.check_seal(base.read_json(path))
    if record['protocol_sha256'] != p['sha256']:
        raise ValueError('Protocol mismatch')
    for name, digest in record['files'].items():
        if base.digest(path.parent / name) != digest:
            raise ValueError('Saved record changed: ' + name)
    return record


def save_contexts(path, contexts):
    torch.save([{k: (v.detach().cpu() if torch.is_tensor(v) else torch.as_tensor(v))
                 for k, v in item.items()} for item in contexts], path)


def load_contexts(path, device='cuda:1'):
    records = torch.load(path, map_location=device, weights_only=True)
    for item in records:
        item['indices'] = item['indices'].cpu().numpy()
    return records


def train(split, p):
    dest = OUT / 'runs' / split
    marker = dest / 'TRAINING.json'
    if marker.exists():
        return checked_record(marker, p)
    dest.mkdir(parents=True, exist_ok=True)
    source, cache = entry(split)
    a = read(source, runtime.TRAIN_KEYS)
    offset, manifest = reference_io.offset_arrays(cache, 'MiDrug', split, a)
    oof = {arm: np.full(len(a['y_train']), np.nan) for arm in ARMS}
    model = runtime.load_model('cuda:1')
    files, folds = [], []
    for fold in range(p['folds']):
        prefix = 'fold_' + str(fold)
        fold_marker = dest / (prefix + '.json')
        query_rows = np.flatnonzero(a['folds'] == fold)
        if fold_marker.exists():
            fold_record = checked_record(fold_marker, p)
            scores = read(dest / (prefix + '_oof.npz'), ['query_rows'] + list(ARMS))
            np.testing.assert_array_equal(scores['query_rows'], query_rows)
        else:
            start = time.monotonic()
            graph, selected, excluded = core.setup_fold(a, fold)
            contexts = core.encode_contexts(model, graph, a, selected)
            scores = core.predict(model, graph, a, a['pairs_train'][:, query_rows], contexts)
            np.savez_compressed(dest / (prefix + '_oof.npz'), query_rows=query_rows, **scores)
            save_contexts(dest / (prefix + '_context.pt'), contexts)
            names = [prefix + '_oof.npz', prefix + '_context.pt']
            fold_record = base.seal(dict(created_at_utc=now(), protocol_sha256=p['sha256'],
                split=split, fold=fold, seconds=time.monotonic() - start,
                query_rows=len(query_rows), excluded_rows=excluded.tolist(),
                context_indices=[x.tolist() for x in selected], neural_fits=0,
                files={name: base.digest(dest / name) for name in names}))
            base.write_json(fold_marker, fold_record)
            del graph, contexts
            runtime.OPERATOR.clear()
            print('KGPFN_OOF', split, fold, round(fold_record['seconds'], 2), flush=True)
        for arm in ARMS:
            oof[arm][query_rows] = scores[arm]
        files += [prefix + '.json', *fold_record['files']]
        folds.append(fold_record)
    if not all(np.isfinite(value).all() for value in oof.values()):
        raise ValueError('Incomplete OOF predictions')
    calibrations = {arm: calibration.fit(oof[arm], a['y_train'], offset, p['scalar_l2']) for arm in ARMS}
    base.write_json(dest / 'calibration.json', calibrations)
    np.savez_compressed(dest / 'oof.npz', **oof, y_train=a['y_train'], pairs_train=a['pairs_train'],
                        folds=a['folds'], anchor_oof=offset)
    files += ['calibration.json', 'oof.npz']
    record = base.seal(dict(created_at_utc=now(), protocol_sha256=p['sha256'], split=split,
        nested_manifest_sha256=base.digest(manifest), calibration_fits=len(ARMS),
        frozen_support_graphs=p['folds'], frozen_context_draws=p['folds'] * len(p['context_seeds']),
        new_neural_fits=0, query_outcomes_used=False,
        files={name: base.digest(dest / name) for name in files}))
    base.write_json(marker, record)
    return record


def evaluate(split, p):
    dest = OUT / 'runs' / split
    marker = dest / 'result.json'
    if marker.exists():
        return verify_result(split, p)
    checked_record(dest / 'TRAINING.json', p)
    source, _ = entry(split)
    a = read(source, runtime.TRAIN_KEYS)
    query_pairs = read(source, ('pairs_val',))['pairs_val']
    ids = lambda pairs: pairs[0] * int(a['nd']) + pairs[1]
    if np.intersect1d(ids(query_pairs), ids(a['pairs_train'])).size:
        raise ValueError('Training and evaluation overlap')
    model = runtime.load_model('cuda:1')
    scores_by_fold, replay_error, raw_files = [], {arm: 0. for arm in ARMS}, []
    for fold in range(p['folds']):
        graph, selected, excluded = core.setup_fold(a, fold)
        contexts = load_contexts(dest / ('fold_' + str(fold) + '_context.pt'))
        for original, cached in zip(selected, contexts):
            np.testing.assert_array_equal(original, cached['indices'])
        scores = core.predict(model, graph, a, query_pairs, contexts)
        reverse = core.predict(model, graph, a, query_pairs[:, ::-1].copy(), contexts, batch_size=127)
        for arm in ARMS:
            np.testing.assert_allclose(scores[arm], reverse[arm][::-1], atol=3e-5, rtol=3e-5)
            replay_error[arm] = max(replay_error[arm], float(abs(scores[arm] - reverse[arm][::-1]).max()))
        name = 'fold_' + str(fold) + '_query.npz'
        np.savez_compressed(dest / name, pairs=query_pairs, **scores)
        raw_files.append(name)
        scores_by_fold.append(scores)
        del graph, contexts
        runtime.OPERATOR.clear()
        print('KGPFN_QUERY', split, fold, flush=True)
    means = {arm: np.mean([scores[arm] for scores in scores_by_fold], axis=0) for arm in ARMS}
    fitted = base.read_json(dest / 'calibration.json')
    residual = {arm: calibration.predict(means[arm], fitted[arm], p['residual_bound']) for arm in ARMS}
    # Only now load evaluation outcomes, after all predictions and replay checks.
    labels = read(source, ('y_val',))['y_val']
    reference_seeds = p['historical_reference_seeds'] if split == 'historical' else [666]
    saved = dict(y=labels, pairs=query_pairs, **{arm + '_raw': means[arm] for arm in ARMS},
                 **{arm + '_residual': residual[arm] for arm in ARMS})
    metrics, identities = {}, {}
    for seed in reference_seeds:
        key = str(seed)
        if split == 'historical':
            anchor = read(source, ('anchor_' + key,))['anchor_' + key]
            ln = read(HISTORICAL / 'LncDrug/arrays.npz', ('y_val', 'pairs_val', 'anchor_' + key))
            ly, lp, la = ln['y_val'], ln['pairs_val'], ln['anchor_' + key]
        else:
            ref = read(PARTITIONS / 'references/MiDrug' / split / 'scores.npz', ('y', 'pairs', 'anchor'))
            np.testing.assert_array_equal(ref['y'], labels)
            np.testing.assert_array_equal(ref['pairs'], query_pairs)
            anchor = ref['anchor']
            ln = read(PARTITIONS / 'references/LncDrug' / split / 'scores.npz', ('y', 'pairs', 'anchor'))
            ly, lp, la = ln['y'], ln['pairs'], ln['anchor']
        saved['anchor_' + key] = anchor
        metrics[key] = dict(reference=base.metrics(labels, anchor, query_pairs))
        for arm in ARMS:
            saved[arm + '_' + key] = anchor + p['residual_weight'] * residual[arm]
            metrics[key][arm] = base.metrics(labels, saved[arm + '_' + key], query_pairs)
        name = 'lnc_identity_' + key + '.npz'
        np.savez_compressed(dest / name, y=ly, pairs=lp, anchor=la, **{arm: la.copy() for arm in ARMS})
        raw_files.append(name)
        identities[key] = base.metrics(ly, la, lp)
    np.savez_compressed(dest / 'scores.npz', **saved)
    raw_files += ['scores.npz', 'TRAINING.json']
    result = base.seal(dict(created_at_utc=now(), split=split, protocol_sha256=p['sha256'],
        metrics=metrics, lnc_identity_metrics=identities, contexts=reference_seeds,
        context_count_is_not_fit_count=True, query_order_and_batch_max_errors=replay_error,
        files={name: base.digest(dest / name) for name in raw_files}))
    base.write_json(marker, result)
    return verify_result(split, p)


def verify_result(split, p):
    dest = OUT / 'runs' / split
    result = checked_record(dest / 'result.json', p)
    checked_record(dest / 'TRAINING.json', p)
    oof = read(dest / 'oof.npz', list(ARMS) + ['y_train', 'pairs_train', 'folds', 'anchor_oof'])
    fitted = base.read_json(dest / 'calibration.json')
    for fold in range(p['folds']):
        saved = read(dest / ('fold_' + str(fold) + '_oof.npz'), ['query_rows'] + list(ARMS))
        rows = np.flatnonzero(oof['folds'] == fold)
        np.testing.assert_array_equal(rows, saved['query_rows'])
        for arm in ARMS:
            np.testing.assert_array_equal(oof[arm][rows], saved[arm])
    for arm in ARMS:
        fit = fitted[arm]
        np.testing.assert_allclose([fit['mean'], fit['scale']],
            [oof[arm].mean(), max(oof[arm].std(), 1e-8)], atol=1e-12, rtol=1e-12)
        x = (oof[arm] - fit['mean']) / fit['scale']
        value, gradient = calibration.objective(np.array([fit['beta'], fit['intercept']]),
            x, oof['y_train'], oof['anchor_oof'], p['scalar_l2'])
        if abs(gradient).max() > 1e-7 or abs(value - fit['objective']) > 1e-12:
            raise ValueError('Calibration stationarity check failed')
    with np.load(dest / 'scores.npz', allow_pickle=False) as z:
        per_fold = [read(dest / ('fold_' + str(fold) + '_query.npz'), ['pairs'] + list(ARMS))
                    for fold in range(p['folds'])]
        for item in per_fold:
            np.testing.assert_array_equal(item['pairs'], z['pairs'])
        for arm in ARMS:
            mean = np.mean([item[arm] for item in per_fold], axis=0)
            np.testing.assert_array_equal(mean, z[arm + '_raw'])
            residual = calibration.predict(mean, fitted[arm], p['residual_bound'])
            np.testing.assert_array_equal(residual, z[arm + '_residual'])
        for seed in result['contexts']:
            key = str(seed)
            for arm in ('reference',) + ARMS:
                name = ('anchor' if arm == 'reference' else arm) + '_' + key
                if arm != 'reference':
                    np.testing.assert_array_equal(z[name], z['anchor_' + key] + p['residual_weight'] * z[arm + '_residual'])
                if base.metrics(z['y'], z[name], z['pairs']) != result['metrics'][key][arm]:
                    raise ValueError('Metric reconstruction failed')
    for seed in result['contexts']:
        with np.load(dest / ('lnc_identity_' + str(seed) + '.npz'), allow_pickle=False) as z:
            for arm in ARMS:
                np.testing.assert_array_equal(z[arm], z['anchor'])
            if base.metrics(z['y'], z['anchor'], z['pairs']) != result['lnc_identity_metrics'][str(seed)]:
                raise ValueError('Lnc identity metric reconstruction failed')
    return result


def assess(records, p, stage):
    cells = [cell for result in records for cell in result['metrics'].values()]
    if len(cells) != 3:
        raise ValueError('Expected three prespecified comparison contexts')
    means = {arm: {metric: float(np.mean([cell[arm][metric] for cell in cells])) for metric in ('auc', 'ap')}
             for arm in ('reference',) + ARMS}
    differences = {arm: {metric: means['full'][metric] - means[arm][metric] for metric in ('auc', 'ap')}
                   for arm in ('reference',) + tuple(p['controls'])}
    positives = {metric: sum(cell['full'][metric] > cell['reference'][metric] for cell in cells)
                 for metric in ('auc', 'ap')}
    performance = all(differences['reference'][metric] > 0 and positives[metric] >= 2 for metric in ('auc', 'ap'))
    mechanism = all(differences[arm][metric] > 0 for arm in p['controls'] for metric in ('auc', 'ap'))
    fields = dict(protocol_sha256=p['sha256'], stage=stage, mean_metrics=means, differences=differences,
        positive_reference_contexts=positives, contexts=len(cells), performance_gate=performance,
        primary_mechanism_gate=mechanism, lnc_exact_identity=True, declared_gate=performance and mechanism,
        historical_contexts_are_not_independent_method_fits=stage == 'HISTORICAL_ASSESSMENT',
        confirmed_final_innovations=0,
        result_sha256={r['split']: base.digest(OUT / 'runs' / r['split'] / 'result.json') for r in records})
    target = OUT / (stage + '.json')
    if target.exists():
        saved = base.check_seal(base.read_json(target))
        if any(saved[key] != value for key, value in fields.items()):
            raise ValueError('Assessment reconstruction failed')
        return saved
    record = base.seal(dict(created_at_utc=now(), **fields))
    base.write_json(target, record)
    print('KGPFN_ASSESSMENT', stage, means, differences, positives, record['declared_gate'], flush=True)
    return record


def run():
    p = checked()
    train('historical', p)
    historical = assess([evaluate('historical', p)], p, 'HISTORICAL_ASSESSMENT')
    partition = None
    if historical['declared_gate']:
        results = []
        for split in p['splits']:
            train(split, p)
            results.append(evaluate(split, p))
        partition = assess(results, p, 'PARTITION_ASSESSMENT')
    target = HERE / 'KGPFN_QUEUE_RESULT_V1.json'
    if not target.exists():
        trains = [base.read_json(path) for path in (OUT / 'runs').glob('*/TRAINING.json')]
        base.write_json(target, base.seal(dict(created_at_utc=now(), protocol_sha256=p['sha256'],
            historical_gate=historical['declared_gate'], partition_gate=partition['declared_gate'] if partition else None,
            new_neural_fits=0, calibration_fits=sum(x['calibration_fits'] for x in trains),
            support_graphs=sum(x['frozen_support_graphs'] for x in trains),
            context_draws=sum(x['frozen_context_draws'] for x in trains),
            new_outer_fits=0, confirmed_final_innovations=0,
            next_step='freeze_outer_recheck' if partition and partition['declared_gate'] else 'stop_this_version')))


def replay(split, p, model):
    dest = OUT / 'runs' / split
    source, _ = entry(split)
    a = read(source, runtime.TRAIN_KEYS)
    query_pairs = read(source, ('pairs_val',))['pairs_val']
    errors = {arm: 0. for arm in ARMS}
    context_error = 0.
    for fold in range(p['folds']):
        graph, selected, excluded = core.setup_fold(a, fold)
        contexts = core.encode_contexts(model, graph, a, selected)
        stored = load_contexts(dest / ('fold_' + str(fold) + '_context.pt'))
        for actual, expected in zip(contexts, stored):
            np.testing.assert_array_equal(actual['indices'], expected['indices'])
            for key in ('labels', 'shuffled'):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            torch.testing.assert_close(actual['context'], expected['context'], atol=3e-5, rtol=3e-5)
            context_error = max(context_error, float((actual['context'] - expected['context']).abs().max()))
        train_rows = np.flatnonzero(a['folds'] == fold)
        for tag, pairs in (('oof', a['pairs_train'][:, train_rows]), ('query', query_pairs)):
            expected = read(dest / ('fold_' + str(fold) + '_' + tag + '.npz'), ARMS)
            actual = core.predict(model, graph, a, pairs, contexts)
            for arm in ARMS:
                np.testing.assert_allclose(actual[arm], expected[arm], atol=3e-5, rtol=3e-5)
                errors[arm] = max(errors[arm], float(abs(actual[arm] - expected[arm]).max()))
        runtime.OPERATOR.clear()
        print('KGPFN_REPLAY', split, fold, flush=True)
    return dict(split=split, context_representation_max_error=context_error,
                all_oof_and_query_raw_scores_replayed=True, max_raw_score_errors=errors)


def audit():
    p = checked()
    results = [verify_result(path.parent.name, p) for path in sorted((OUT / 'runs').glob('*/result.json'))]
    assess([r for r in results if r['split'] == 'historical'], p, 'HISTORICAL_ASSESSMENT')
    if len(results) > 1:
        assess([r for r in results if r['split'] != 'historical'], p, 'PARTITION_ASSESSMENT')
    target = HERE / 'KGPFN_CONSOLIDATION_V1.json'
    if target.exists():
        raise ValueError('Preserve the existing audit')
    model = runtime.load_model('cuda:1')
    replay_records = [replay(r['split'], p, model) for r in results]
    base.write_json(target, base.seal(dict(created_at_utc=now(), protocol_sha256=p['sha256'],
        all_saved_metrics_recomputed=True, calibration_stationarity_checked=True,
        replay_records=replay_records, results=results,
        queue=base.check_seal(base.read_json(HERE / 'KGPFN_QUEUE_RESULT_V1.json')),
        original_test_source_access=False, confirmed_final_innovations=0)))
    print('KGPFN_AUDIT_COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'audit'))
    globals()[parser.parse_args().command]()
