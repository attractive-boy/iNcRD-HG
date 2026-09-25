"""Versioned audit correction for TabICL's existing constant-feature filter.

The reference recipe, features, checkpoint, labels, random seed, and prediction
formula are unchanged. This version checks the actual training-only feature
filter and preserves all v2 artifacts. Completed v2 fits are copied with exact
score hashes; an interrupted context resumes from its already saved model.
"""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import torch

from improve import nested_anchor_fit_v2 as previous

inputs, base = previous.inputs, previous.base
OWN = ('nested_anchor_fit_v3.py', 'test_nested_anchor_v3.py')
DEFAULT = inputs.ROOT / 'results/nested_anchor_v3_20260915/partitions'


def audit_training_context(model, x, y):
    """Reconstruct the fitted feature mask using training rows only."""
    encoded = model.X_encoder_.transform(x)
    ensemble = model.ensemble_generator_
    feature_filter = ensemble.unique_filter_
    if feature_filter.threshold != 1:
        raise ValueError('The fixed reference uses the default unique-feature threshold')
    expected = (np.ones(encoded.shape[1], dtype=bool) if len(encoded) <= 1 else
                np.array([len(np.unique(encoded[:, i])) > 1 for i in range(encoded.shape[1])]))
    if not expected.any():
        expected[0] = True
    np.testing.assert_array_equal(feature_filter.features_to_keep_, expected)
    np.testing.assert_array_equal(ensemble.X_, encoded[:, expected])
    np.testing.assert_array_equal(ensemble.y_, y.astype(int))
    return dict(input_columns=int(encoded.shape[1]), kept_columns=int(expected.sum()),
        removed_columns=np.flatnonzero(~expected).tolist(),
        training_rows=int(len(encoded)), mask_recomputed_from_training_only=True,
        filtered_training_arrays_exactly_matched=True)


def freeze(out, parent):
    out, parent = Path(out).resolve(), Path(parent).resolve()
    if out.exists():
        raise ValueError('Use a new corrected reference cache directory')
    old = inputs.protocol(parent)
    manifests = []
    for cell in old['cells']:
        for fold in range(old['inner_folds']):
            path = parent / 'inputs' / cell['task'] / cell['split'] / str(fold) / 'manifest.json'
            record = base.check_seal(base.read_json(path))
            if record['protocol_sha256'] != old['sha256']:
                raise ValueError('Nested feature manifest protocol mismatch')
            inputs.verify_files(path.parent, record)
            manifests.append(path)
    from tabicl._sklearn import preprocessing
    p = {k: v for k, v in old.items() if k != 'sha256'}
    p.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
        version='nested_anchor_v3_constant_filter_audit',
        parent_reference_cache=str(parent), parent_protocol_sha256=old['sha256'],
        correction='v2 compares EnsembleGenerator.X_ before its automatic constant-feature filter. Reconstruct and check the unchanged training-only filter before exact array comparison. No estimator, feature, label, prediction, or seed change.',
        inherited_feature_correction=old['correction'],
        source_sha256={**old['source_sha256'], **{n: base.digest(inputs.ROOT / n) for n in OWN}},
        input_sha256={**old['input_sha256'], str(parent / 'protocol.json'): base.digest(parent / 'protocol.json'),
            str(Path(preprocessing.__file__).resolve()): base.digest(preprocessing.__file__),
            **{str(path): base.digest(path) for path in manifests}},
        reuse_policy='Verify every completed v2 fit and copy its raw scores byte-for-byte. Resume interrupted contexts from saved weights. Never edit parent files.',
        predictor_recipe_unchanged=True, correction_uses_evaluation_outcomes=False)
    p = base.seal(p)
    out.mkdir(parents=True)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for name in p['source_sha256']:
        shutil.copy2(inputs.ROOT / name, snapshot / name)
    base.write_json(out / 'protocol.json', p)
    print('FROZEN_NESTED_REFERENCE_V3', old['scope'], p['sha256'], flush=True)


def context_prediction(dest, legacy, x, y, query, p, device):
    from tabicl import TabICLClassifier
    dest, legacy = Path(dest), Path(legacy)
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {k: previous.array_hash(v) for k, v in dict(x=x, y=y, query=query).items()}
    if (dest / 'result.json').exists():
        record = base.check_seal(base.read_json(dest / 'result.json'))
        if record['protocol_sha256'] != p['sha256'] or record['input_hashes'] != hashes:
            raise ValueError('Existing corrected context does not match')
        inputs.verify_files(dest, record)
        return previous.load(dest / 'scores.npz')['probability'], record['replay_error']
    if (legacy / 'result.json').exists():
        old = base.check_seal(base.read_json(legacy / 'result.json'))
        if old['protocol_sha256'] != p['parent_protocol_sha256'] or old['input_hashes'] != hashes:
            raise ValueError('Completed parent context does not match')
        inputs.verify_files(legacy, old)
        for name in ('model.pkl', 'scores.npz'):
            shutil.copy2(legacy / name, dest / name)
            if base.digest(dest / name) != old['files'][name]:
                raise ValueError('Parent context copy was not exact')
        record = {k: v for k, v in old.items() if k != 'sha256'}
        record.update(protocol_sha256=p['sha256'],
            adopted_from=str(legacy / 'result.json'),
            adopted_result_sha256=base.digest(legacy / 'result.json'),
            new_context_fit=False, new_context_prediction=False,
            legacy_unfiltered_training_equality_already_passed=True)
        base.write_json(dest / 'result.json', base.seal(record))
        return previous.load(dest / 'scores.npz')['probability'], old['replay_error']
    model_path = dest / 'model.pkl'
    inherited = None
    if not model_path.exists() and (legacy / 'model.pkl').exists():
        shutil.copy2(legacy / 'model.pkl', model_path)
        inherited = dict(path=str(legacy / 'model.pkl'), sha256=base.digest(model_path))
    fitted = not model_path.exists()
    if fitted:
        model = base.fit_context(base.estimator(dict(n_estimators=p['members']),
            p['reference_seed'], p['icl_protocol'], device), x, y)
        model.save(model_path, save_model_weights=False, save_training_data=True, save_kv_cache=False)
    else:
        model = TabICLClassifier.load(model_path, device=device)
    training_audit = audit_training_context(model, x, y)
    parameter_hash = base.parameter_digest(model)
    probability = base.query_scores(model, query)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    replay_model = TabICLClassifier.load(model_path, device=device)
    if base.parameter_digest(replay_model) != parameter_hash:
        raise ValueError('Reference parameter replay mismatch')
    replay_audit = audit_training_context(replay_model, x, y)
    if replay_audit != training_audit:
        raise ValueError('Training filter changed after model reload')
    replay = base.query_scores(replay_model, query[::-1].copy())[::-1]
    np.testing.assert_allclose(probability, replay, atol=1e-6, rtol=1e-6)
    error = float(abs(probability - replay).max())
    np.savez_compressed(dest / 'scores.npz', probability=probability)
    base.write_json(dest / 'result.json', base.seal(dict(protocol_sha256=p['sha256'],
        input_hashes=hashes, parameter_sha256=parameter_hash, replay_error=error,
        fitting_arrays_exactly_matched=True, training_filter_audit=training_audit,
        inherited_saved_model=inherited, new_context_fit=fitted, new_context_prediction=True,
        query_outcomes_read=False,
        files={n: base.digest(dest / n) for n in ('model.pkl', 'scores.npz')})))
    del replay_model
    gc.collect()
    torch.cuda.empty_cache()
    return probability, error


def fit_fold(out, p, cell, fold, device):
    task, split = cell['task'], cell['split']
    parent = Path(p['parent_reference_cache'])
    source = parent / 'inputs' / task / split / str(fold)
    manifest = base.check_seal(base.read_json(source / 'manifest.json'))
    if manifest['protocol_sha256'] != p['parent_protocol_sha256']:
        raise ValueError('Wrong inherited feature protocol')
    inputs.verify_files(source, manifest)
    dest, legacy = Path(out) / 'fits' / task / split / str(fold), parent / 'fits' / task / split / str(fold)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'result.json').exists():
        record = base.check_seal(base.read_json(dest / 'result.json'))
        if record['protocol_sha256'] != p['sha256'] or record['input_manifest_sha256'] != manifest['sha256']:
            raise ValueError('Wrong corrected fit protocol')
        inputs.verify_files(dest, record)
        return
    if (legacy / 'result.json').exists():
        old = base.check_seal(base.read_json(legacy / 'result.json'))
        if old['protocol_sha256'] != p['parent_protocol_sha256'] or old['input_manifest_sha256'] != manifest['sha256']:
            raise ValueError('Wrong completed parent fit')
        inputs.verify_files(legacy, old)
        shutil.copy2(legacy / 'scores.npz', dest / 'scores.npz')
        if base.digest(dest / 'scores.npz') != old['files']['scores.npz']:
            raise ValueError('Adopted fold scores changed')
        record = {k: v for k, v in old.items() if k not in ('sha256', 'files')}
        record.update(protocol_sha256=p['sha256'], seconds=0., original_fit_seconds=old['seconds'],
            adopted_complete_fit=True, adopted_from=str(legacy / 'result.json'),
            adopted_result_sha256=base.digest(legacy / 'result.json'),
            adopted_original_artifacts=old['files'],
            files={'scores.npz': base.digest(dest / 'scores.npz')})
        base.write_json(dest / 'result.json', base.seal(record))
        print('ADOPTED_NESTED_REFERENCE_FOLD', task, split, fold, 'exact_score_copy', flush=True)
        return
    a = previous.load(source / 'arrays.npz')
    if any(k.startswith('y_') and k != 'y_train' for k in a):
        raise ValueError('Nested reference fitting cannot read query outcomes')
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
            query = (np.concatenate([a[prefix + '_query'][None], a[prefix + '_query_views']])
                     if full and arm == 'query_bag' else a[prefix + '_query'][None] if full
                     else a[prefix + '_query_views'][int(kind[-1]):int(kind[-1]) + 1])
            if variant == 'tiered':
                x = np.concatenate([x, a['tiered_train'][ids]], axis=1).astype(np.float32)
                query = np.concatenate([query, np.broadcast_to(a['tiered_query'],
                    (len(query), *a['tiered_query'].shape))], axis=2).astype(np.float32)
            values, error = context_prediction(dest / variant / kind, legacy / variant / kind,
                x, a['y_train'][ids], query.reshape(-1, x.shape[1]), p, device)
            components[kind] = values.reshape(query.shape[:2])
            errors.append(error)
            print('NESTED_REFERENCE_CONTEXT_V3', task, split, fold, variant, kind, 'replay', error, flush=True)
        context = (components['full'][1:].mean(0) if arm == 'query_bag' else
                   np.stack([components['fold' + str(i)][0] for i in range(p['inner_folds'])]).mean(0))
        graph_scores[variant] = .5 * components['full'][0] + .5 * context
    member_path = dest / 'classic_members.pkl'
    if not member_path.exists() and (legacy / 'classic_members.pkl').exists():
        shutil.copy2(legacy / 'classic_members.pkl', member_path)
    classic_fitted = not member_path.exists()
    if classic_fitted:
        members = {name: previous.anchor.learner(name, seed=p['anchor']['seed']).fit(
            a['context76_train'], a['y_train']) for name in p['anchor']['members']}
        with member_path.open('wb') as stream:
            pickle.dump(members, stream, protocol=4)
    else:
        with member_path.open('rb') as stream:
            members = pickle.load(stream)
    logits = previous.precision.member_mean(members, p['anchor']['members'], a['context76_query'])
    with member_path.open('rb') as stream:
        reloaded = pickle.load(stream)
    replay = previous.precision.member_mean(reloaded, p['anchor']['members'], a['context76_query'][::-1])[::-1]
    np.testing.assert_allclose(logits, replay, atol=1e-10, rtol=1e-10)
    tabicl = previous.anchor.anchors.logit(.5 * graph_scores['graph'] + .5 * graph_scores['tiered'])
    offset = .7 * tabicl + .3 * logits
    if not np.isfinite(offset).all():
        raise ValueError('Nonfinite corrected reference offset')
    np.savez_compressed(dest / 'scores.npz', query_indices=a['query_indices'], pairs=a['pairs_query'],
                        anchor=offset, tabicl=tabicl, classic=logits)
    base.write_json(dest / 'result.json', base.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=p['sha256'],
        input_manifest_sha256=manifest['sha256'], task=task, split=split, fold=int(fold),
        seconds=time.monotonic() - start, adopted_complete_fit=False, classic_members_new_fit=classic_fitted,
        tabicl_replay_max_abs=max(errors), classic_replay_max_abs=float(abs(logits - replay).max()),
        query_outcomes_read=False, prediction_metrics_evaluated=False,
        files={str(path.relative_to(dest)): base.digest(path) for path in dest.rglob('*') if path.is_file()})))
    print('NESTED_REFERENCE_FOLD_COMPLETE_V3', task, split, fold, flush=True)


def run(out, task, device):
    if device != 'cuda:1':
        raise ValueError('Reference fitting is authorized only on cuda:1')
    p = inputs.protocol(out)
    if p['version'] != 'nested_anchor_v3_constant_filter_audit':
        raise ValueError('Use a frozen v3 reference cache')
    torch.cuda.set_device(device)
    for cell in [c for c in p['cells'] if c['task'] == task]:
        for fold in range(p['inner_folds']):
            fit_fold(out, p, cell, fold, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'assemble'))
    parser.add_argument('--out', default=str(DEFAULT))
    parser.add_argument('--parent')
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.out, args.parent)
    elif args.command == 'run':
        run(args.out, args.task, args.device)
    else:
        previous.assemble(args.out, args.task)
