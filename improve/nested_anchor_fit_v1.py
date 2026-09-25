"""Fit/replay fixed strong-reference predictions for nested training folds.

Saved offsets contain one prediction per training observation, made without
that observation's outer fold. Predictions are never judged on query labels.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from improve import nested_anchor_inputs_v1 as inputs
from improve import context_anchor_verify as anchor
from improve import anchor_prediction_precision as precision

base = inputs.base


def array_hash(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(str((x.shape, x.dtype)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def context_prediction(dest, x, y, q, p, device):
    from tabicl import TabICLClassifier
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {k: array_hash(v) for k, v in dict(x=x, y=y, query=q).items()}
    if (dest / 'result.json').exists():
        r = base.check_seal(base.read_json(dest / 'result.json'))
        assert r['protocol_sha256'] == p['sha256'] and r['input_hashes'] == hashes
        inputs.verify_files(dest, r)
        return load(dest / 'scores.npz')['probability'], r['replay_error']
    path = dest / 'model.pkl'
    if path.exists():
        model = TabICLClassifier.load(path, device=device)
    else:
        model = base.fit_context(base.estimator(dict(n_estimators=p['members']),
            p['reference_seed'], p['icl_protocol'], device), x, y)
        model.save(path, save_model_weights=False, save_training_data=True, save_kv_cache=False)
    np.testing.assert_array_equal(model.ensemble_generator_.X_, model.X_encoder_.transform(x))
    np.testing.assert_array_equal(model.ensemble_generator_.y_, y.astype(int))
    parameter_hash = base.parameter_digest(model)
    probability = base.query_scores(model, q)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    replay_model = TabICLClassifier.load(path, device=device)
    assert base.parameter_digest(replay_model) == parameter_hash
    replay = base.query_scores(replay_model, q[::-1].copy())[::-1]
    np.testing.assert_allclose(probability, replay, atol=1e-6, rtol=1e-6)
    error = float(abs(probability - replay).max())
    np.savez_compressed(dest / 'scores.npz', probability=probability)
    base.write_json(dest / 'result.json', base.seal(dict(protocol_sha256=p['sha256'],
        input_hashes=hashes, parameter_sha256=parameter_hash, replay_error=error,
        fitting_arrays_exactly_matched=True, query_outcomes_read=False,
        files={n: base.digest(dest / n) for n in ('model.pkl', 'scores.npz')})))
    del replay_model
    gc.collect()
    torch.cuda.empty_cache()
    return probability, error


def fit_fold(out, p, cell, fold, device):
    task, split = cell['task'], cell['split']
    source = Path(out) / 'inputs' / task / split / str(fold)
    manifest = base.check_seal(base.read_json(source / 'manifest.json'))
    assert manifest['protocol_sha256'] == p['sha256']
    inputs.verify_files(source, manifest)
    a = load(source / 'arrays.npz')
    if any(k.startswith('y_') and k != 'y_train' for k in a):
        raise ValueError('A nested reference fit may not access query outcomes')
    dest = Path(out) / 'fits' / task / split / str(fold)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'result.json').exists():
        r = base.check_seal(base.read_json(dest / 'result.json'))
        assert r['protocol_sha256'] == p['sha256'] and r['input_manifest_sha256'] == manifest['sha256']
        inputs.verify_files(dest, r)
        return
    start = time.monotonic()
    prefix = 'raw' if p['input_configs'][task]['input_view'] == 'raw' else 'x'
    arm = p['graph_recipes'][task]['arm']
    graph_scores, errors = {}, []
    for variant in ('graph', 'tiered'):
        components = {}
        kinds = ['full'] + (['fold' + str(i) for i in range(p['inner_folds'])] if arm == 'fold_match' else [])
        for kind in kinds:
            full = kind == 'full'
            ids = np.arange(len(a['y_train'])) if full else np.flatnonzero(a['fold_ids'] == int(kind[-1]))
            x = a[prefix + '_train'][ids]
            q = (np.concatenate([a[prefix + '_query'][None], a[prefix + '_query_views']])
                 if full and arm == 'query_bag' else a[prefix + '_query'][None] if full
                 else a[prefix + '_query_views'][int(kind[-1]):int(kind[-1]) + 1])
            if variant == 'tiered':
                x = np.concatenate([x, a['tiered_train'][ids]], axis=1).astype(np.float32)
                q = np.concatenate([q, np.broadcast_to(a['tiered_query'],
                    (len(q), *a['tiered_query'].shape))], axis=2).astype(np.float32)
            values, error = context_prediction(dest / variant / kind, x, a['y_train'][ids],
                q.reshape(-1, x.shape[1]), p, device)
            components[kind] = values.reshape(q.shape[:2])
            errors.append(error)
            print('NESTED_REFERENCE_CONTEXT', task, split, fold, variant, kind, 'replay', error, flush=True)
        context = (components['full'][1:].mean(0) if arm == 'query_bag' else
                   np.stack([components['fold' + str(i)][0] for i in range(p['inner_folds'])]).mean(0))
        graph_scores[variant] = .5 * components['full'][0] + .5 * context
    member_path = dest / 'classic_members.pkl'
    if member_path.exists():
        with member_path.open('rb') as stream:
            members = pickle.load(stream)
    else:
        members = {}
        for name in p['anchor']['members']:
            members[name] = anchor.learner(name, seed=p['anchor']['seed']).fit(a['context76_train'], a['y_train'])
        with member_path.open('wb') as stream:
            pickle.dump(members, stream, protocol=4)
    member_logits = precision.member_mean(members, p['anchor']['members'], a['context76_query'])
    with member_path.open('rb') as stream:
        replay_members = pickle.load(stream)
    replay = precision.member_mean(replay_members, p['anchor']['members'], a['context76_query'][::-1])[::-1]
    np.testing.assert_allclose(member_logits, replay, atol=1e-10, rtol=1e-10)
    tiered = .5 * graph_scores['graph'] + .5 * graph_scores['tiered']
    tabicl_logit = anchor.anchors.logit(tiered)
    offset = .7 * tabicl_logit + .3 * member_logits
    if not np.isfinite(offset).all():
        raise ValueError('Nonfinite nested reference offset')
    np.savez_compressed(dest / 'scores.npz', query_indices=a['query_indices'], pairs=a['pairs_query'],
        anchor=offset, tabicl=tabicl_logit, classic=member_logits)
    record = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=p['sha256'], input_manifest_sha256=manifest['sha256'],
        task=task, split=split, fold=int(fold), seconds=time.monotonic() - start,
        tabicl_replay_max_abs=max(errors), classic_replay_max_abs=float(abs(member_logits - replay).max()),
        query_outcomes_read=False, prediction_metrics_evaluated=False,
        files={str(f.relative_to(dest)): base.digest(f) for f in dest.rglob('*') if f.is_file()}))
    base.write_json(dest / 'result.json', record)
    print('NESTED_REFERENCE_FOLD_COMPLETE', task, split, fold, 'seconds', round(record['seconds'], 1), flush=True)


def run(out, task, device):
    if device != 'cuda:1':
        raise ValueError('This study is authorized on cuda:1 only')
    p = inputs.protocol(out)
    torch.cuda.set_device(device)
    for cell in [c for c in p['cells'] if c['task'] == task]:
        for fold in range(p['inner_folds']):
            fit_fold(out, p, cell, fold, device)


def assemble(out, task):
    out = Path(out)
    p = inputs.protocol(out)
    for cell in [c for c in p['cells'] if c['task'] == task]:
        a = inputs.load_selected(cell['source'], ('pairs_train', 'folds'))
        dest = out / 'offsets' / task / cell['split']
        if (dest / 'manifest.json').exists():
            r = base.check_seal(base.read_json(dest / 'manifest.json'))
            assert r['protocol_sha256'] == p['sha256']
            inputs.verify_files(dest, r)
            continue
        offset = np.full(len(a['folds']), np.nan)
        counts = np.zeros(len(offset), dtype=int)
        files, replay_errors = {}, []
        for fold in range(p['inner_folds']):
            fit = out / 'fits' / task / cell['split'] / str(fold)
            r = base.check_seal(base.read_json(fit / 'result.json'))
            assert r['protocol_sha256'] == p['sha256']
            inputs.verify_files(fit, r)
            z = load(fit / 'scores.npz')
            ids = np.flatnonzero(a['folds'] == fold)
            np.testing.assert_array_equal(ids, z['query_indices'])
            np.testing.assert_array_equal(a['pairs_train'][:, ids], z['pairs'])
            counts[ids] += 1
            offset[ids] = z['anchor']
            files[str(fit / 'result.json')] = base.digest(fit / 'result.json')
            replay_errors.append(r['tabicl_replay_max_abs'])
        if not np.all(counts == 1) or not np.isfinite(offset).all():
            raise ValueError('Nested OOF offsets do not cover each training row exactly once')
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / 'arrays.npz', anchor_oof=offset,
            pairs_train=a['pairs_train'], folds=a['folds'])
        base.write_json(dest / 'manifest.json', base.seal(dict(protocol_sha256=p['sha256'],
            task=task, split=cell['split'], query_outcomes_read=False,
            prediction_metrics_evaluated=False, rows=len(offset), every_row_predicted_once=True,
            max_tabicl_replay_error=max(replay_errors), fit_records=files,
            files={'arrays.npz': base.digest(dest / 'arrays.npz')})))
        print('NESTED_OFFSETS_ASSEMBLED', task, cell['split'], len(offset), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'assemble'))
    parser.add_argument('--out', default=str(inputs.DEFAULT))
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'), required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.out, args.task, args.device)
    else:
        assemble(args.out, args.task)
