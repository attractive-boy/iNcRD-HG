"""Freeze, execute, and audit the auxiliary-graph reference-residual GP study."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import numpy as np

import torch

from improve import attention_gp_residual_v2 as core
from improve import tabular_foundation as base
from improve import anchor_priority_graph_v1 as reference_io

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT.parent
OUT = ROOT / 'results/attention_gp_residual_v2_20260916'
PARTITIONS = ROOT / 'results/mechanism_validation_20260915'
HISTORICAL = ROOT / 'results/anchor_priority_morphological_alignment_v1_20260916/inputs'
ARMS = core.KINDS
TRAIN_KEYS = core.TRAIN_KEYS


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
    unit = base.check_seal(base.read_json(HERE / 'ATTENTION_GP_RESIDUAL_UNIT_V2.json'))
    probe = base.check_seal(base.read_json(HERE / 'ATTENTION_GP_RESIDUAL_PROBE_V2.json'))
    if not unit['passed'] or unit['tests_run'] != 3 or len(probe['records']) != 2:
        raise ValueError('Missing implementation checks')
    for record in (unit, probe):
        for name, digest in record['source_sha256'].items():
            if base.digest(PROJECT / name) != digest:
                raise ValueError('Checked source changed')
    sources = [Path(core.__file__), ROOT / 'attention_gp_rank_v1.py', Path(__file__), Path(base.__file__),
               Path(reference_io.__file__), ROOT / 'test_attention_gp_residual_v2.py', ROOT / 'overlap_block_models.py']
    inputs = [HERE / 'ATTENTION_GP_RESIDUAL_ADAPTATION_V2.md', HERE / 'ATTENTION_GP_RESIDUAL_UNIT_V2.json',
              HERE / 'ATTENTION_GP_RESIDUAL_PROBE_V2.json', HERE / '2603_17569.pdf',
              HERE / 'GATGP_AUTHOR_LANDING_V1.json']
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
    p = base.seal(dict(created_at_utc=now(), family='analytic_attention_gp', version=2,
        source_sha256={str(path): base.digest(path) for path in sources},
        input_sha256={str(path): base.digest(path) for path in inputs},
        primary='full', controls=['gcn_gp'], depth=2, precision=10., residual_bound=2., residual_weight=.1,
        query_policy='direct auxiliary-graph kernel logistic residual, bounded once',
        lnc_policy='exact_reference_identity',
        splits=['61453', '7919', '104729'], historical_reference_seeds=[666, 0, 1],
        method_is_deterministic=True, repeated_reference_contexts_are_not_independent_fits=True,
        gate='Positive mean MiDrug AUC/AP vs reference and GCN-GP; >=2/3 positive reference differences per metric; exact Lnc identity; historical then partitions then frozen outer.',
        candidate_performance_seen=False, prior_development_exposure=True,
        parent_failure_seen=True,
        parent_assessment_sha256=base.digest(ROOT / 'results/attention_gp_rank_v1_20260916/HISTORICAL_ASSESSMENT.json'),
        original_test_source_access=False, confirmed_final_innovations=0))
    OUT.mkdir(parents=True)
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    base.write_json(OUT / 'protocol.json', p)
    print('FROZEN_ATTENTION_GP', p['sha256'], flush=True)

def checked():
    p = base.check_seal(base.read_json(OUT / 'protocol.json'))
    for key in ('source_sha256', 'input_sha256'):
        for name, digest in p[key].items():
            if base.digest(name) != digest:
                raise ValueError('Frozen artifact changed: ' + name)
    return p


def train(split, p, unused=None):
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
    aux = core.auxiliary_arrays(a)
    if np.any(aux['target_codes'] >= 0):
        raise ValueError('A target fact remains in the auxiliary graph')
    ids = a['pairs_train'][0]*216 + a['pairs_train'][1]
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate training pairs')
    mean, records, files = {}, [], []
    for arm in ARMS:
        start = time.perf_counter()
        kr,kd = core.entity_kernels(a, arm, p['depth'], 'cuda:1')
        value,alpha,trace,error = core.fit(kr,kd,a['pairs_train'],a['y_train'],offset,p['precision'])
        mean[arm] = value.cpu().numpy().ravel()
        name = arm+'_model.npz'
        np.savez_compressed(dest / name, rna_kernel=kr.cpu().numpy(), drug_kernel=kd.cpu().numpy(),
                            alpha=alpha.cpu().numpy(), mean_universe=mean[arm])
        base.write_json(dest / (arm+'_trace.json'), trace)
        files += [name, arm+'_trace.json']
        records.append(dict(arm=arm, seconds=time.perf_counter()-start,
            target_edges_remaining=0, iterations=len(trace)-1,
            maximum_dual_residual=trace[-1]['maximum_dual_residual'], replay_error=error))
        print('RESIDUAL_GP_FIT', split, records[-1], flush=True)
        del kr,kd,value,alpha
    np.savez_compressed(dest / 'rank_arrays.npz', pairs_train=a['pairs_train'], y_train=a['y_train'],
        folds=a['folds'], anchor_oof=offset, **{arm+'_mean_universe': mean[arm] for arm in ARMS})
    files.append('rank_arrays.npz')
    record = base.seal(dict(created_at_utc=now(), protocol_sha256=p['sha256'], split=split,
        nested_manifest_sha256=base.digest(manifest), fit_records=records,
        exact_kernel_fits=2, scalar_calibration_fits=0, new_neural_fits=0,
        evaluation_outcomes_used=False, files={name: base.digest(dest / name) for name in files}))
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
    residual = {arm: np.clip(rank[arm+'_mean_universe'][query_ids], -p['residual_bound'], p['residual_bound'])
                for arm in ARMS}
    for arm in ARMS:
        replay = np.clip(rank[arm+'_mean_universe'][query_ids[::-1]], -p['residual_bound'], p['residual_bound'])[::-1]
        chunked = np.concatenate([np.clip(rank[arm+'_mean_universe'][ids], -p['residual_bound'], p['residual_bound'])
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
        np.savez_compressed(dest / ('lnc_identity_'+key+'.npz'), y=ly, pairs=lp, anchor=la, full=la.copy(), gcn_gp=la.copy())
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
    from scipy.special import expit
    dest = OUT / 'runs' / split
    r = base.check_seal(base.read_json(dest / 'result.json'))
    if r['protocol_sha256'] != p['sha256']:
        raise ValueError('Result protocol mismatch')
    for name,digest in r['files'].items():
        if base.digest(dest / name) != digest:
            raise ValueError('Saved score evidence changed')
    saved = read(dest / 'rank_arrays.npz', ['pairs_train','y_train','anchor_oof']+
                 [arm+'_mean_universe' for arm in ARMS])
    mean = {}
    for arm in ARMS:
        v = read(dest / (arm+'_model.npz'), ('rna_kernel','drug_kernel','alpha','mean_universe'))
        weights = np.zeros((605,216))
        np.add.at(weights,tuple(saved['pairs_train']),v['alpha'])
        replay = (v['rna_kernel'] @ weights @ v['drug_kernel'].T).ravel()
        np.testing.assert_allclose(replay,v['mean_universe'],atol=1e-10,rtol=1e-10)
        np.testing.assert_array_equal(v['mean_universe'],saved[arm+'_mean_universe'])
        value = replay[saved['pairs_train'][0]*216+saved['pairs_train'][1]]
        residual = saved['y_train'] - expit(saved['anchor_oof']+value) - p['precision']*v['alpha']
        if abs(residual).max() > 1e-7:
            raise ValueError('Saved kernel MAP function is not stationary')
        mean[arm] = saved[arm+'_mean_universe']
    with np.load(dest / 'scores.npz',allow_pickle=False) as z:
        query_ids = z['pairs'][0]*216+z['pairs'][1]
        for arm in ARMS:
            np.testing.assert_array_equal(np.clip(mean[arm][query_ids],-p['residual_bound'],p['residual_bound']),z[arm+'_residual'])
        for seed in r['contexts']:
            for arm in ('reference',)+ARMS:
                key = ('anchor' if arm=='reference' else arm)+'_'+str(seed)
                if arm!='reference':
                    np.testing.assert_array_equal(z[key],z['anchor_'+str(seed)]+p['residual_weight']*z[arm+'_residual'])
                if base.metrics(z['y'],z[key],z['pairs']) != r['metrics'][str(seed)][arm]:
                    raise ValueError('Stored metrics disagree with raw scores')
    for seed in r['contexts']:
        with np.load(dest / ('lnc_identity_'+str(seed)+'.npz'),allow_pickle=False) as z:
            for arm in ARMS:
                np.testing.assert_array_equal(z[arm],z['anchor'])
            if base.metrics(z['y'],z['anchor'],z['pairs']) != r['lnc_identity_metrics'][str(seed)]:
                raise ValueError('Identity metrics changed')
    return r

def assess(records, p, name):
    target = OUT / (name+'.json')
    cells = [cell for r in records for cell in r['metrics'].values()]
    means = {arm: {m: float(np.mean([cell[arm][m] for cell in cells])) for m in ('auc', 'ap')}
             for arm in ('reference',)+ARMS}
    delta = {arm: {m: means['full'][m]-means[arm][m] for m in ('auc', 'ap')}
             for arm in ('reference', 'gcn_gp')}
    positive = {m: sum(cell['full'][m] > cell['reference'][m] for cell in cells) for m in ('auc', 'ap')}
    performance = all(delta['reference'][m] > 0 and positive[m] >= 2 for m in ('auc', 'ap'))
    mechanism = all(delta['gcn_gp'][m] > 0 for m in ('auc', 'ap'))
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
    print('ATTENTION_GP_RESIDUAL_ASSESSMENT', name, means, delta, positive, result['declared_gate'], flush=True)
    return result


def run():
    p = checked()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train('historical', p)
    historical = assess([evaluate('historical', p)], p, 'HISTORICAL_ASSESSMENT')
    partition = None
    if historical['declared_gate']:
        results = []
        for split in p['splits']:
            train(split, p)
            results.append(evaluate(split, p))
        partition = assess(results, p, 'PARTITION_ASSESSMENT')
    target = HERE / 'ATTENTION_GP_RESIDUAL_QUEUE_RESULT_V2.json'
    if not target.exists():
        trains = [base.read_json(path) for path in (OUT / 'runs').glob('*/TRAINING.json')]
        base.write_json(target, base.seal(dict(created_at_utc=now(), protocol_sha256=p['sha256'],
            historical_gate=historical['declared_gate'],
            partition_gate=partition['declared_gate'] if partition else None,
            exact_kernel_fits=sum(r['exact_kernel_fits'] for r in trains),
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
    output = HERE / 'ATTENTION_GP_RESIDUAL_CONSOLIDATION_V2.json'
    if output.exists():
        raise ValueError('Preserve the existing consolidation')
    base.write_json(output, base.seal(dict(created_at_utc=now(), scores_recomputed=True,
        protocol_sha256=p['sha256'], results=records,
        queue=base.check_seal(base.read_json(HERE / 'ATTENTION_GP_RESIDUAL_QUEUE_RESULT_V2.json')),
        original_test_source_access=False, confirmed_final_innovations=0)))
    print('ATTENTION_GP_RESIDUAL_READ_ONLY_AUDIT', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'audit'))
    globals()[parser.parse_args().command]()
