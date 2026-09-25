"""Rebuild matched graph-query features in the untouched historical environment.

The inference environment does not import the old PyG stack. This executable
only builds features; it never fits a predictor or evaluates a metric.
"""
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import pickle
import sys

import numpy as np

from improve import backbone_replacement as graph_parent
from improve import numeric_embedding as feature_parent
from improve import tabular_foundation as base
from improve.graph_tabm_modules import PairFeatureEncoder
from improve.tabicl_graph_context_modules import support_plan


def build_views(encoder, pairs, labels, fold_ids, query_pairs, raw_train, raw_val):
    plan = support_plan(pairs, labels, fold_ids, query_pairs)
    views, checks = [], []
    names = None
    for item in plan:
        idx, support = item['context'], pairs[:, item['support']]
        combined_pairs = np.concatenate([pairs[:, idx], query_pairs], axis=1)
        frame = encoder.transform(combined_pairs, support)
        values = frame.values
        if names is None:
            names = list(frame.columns)
        if list(frame.columns) != names:
            raise ValueError('Feature column order changed across support graphs')
        np.testing.assert_allclose(values[:len(idx)], raw_train[idx], atol=2e-6, rtol=2e-6)
        views.append(values[len(idx):].copy())
        # Both directions of a target edge are functions of this one explicit
        # support list. The encoder does not read graph-stored target edges.
        forbidden = set(map(tuple, combined_pairs.T))
        overlap = forbidden & set(map(tuple, support.T))
        if overlap:
            raise ValueError('A context or evaluation query leaked into support')
        checks.append(dict(fold=item['fold'], context_rows=len(idx),
            context_positives=int(labels[idx].sum()), support_positives=support.shape[1],
            support_indices_sha256=base.json_digest(item['support'].tolist()),
            context_indices_sha256=base.json_digest(idx.tolist()),
            query_support_overlap=len(overlap),
            context_feature_max_abs=float(np.max(np.abs(values[:len(idx)]-raw_train[idx])))))
    full = encoder.transform(query_pairs, pairs[:, labels == 1])
    if list(full.columns) != names:
        raise ValueError('Full-support feature columns changed')
    np.testing.assert_allclose(full.values, raw_val, atol=2e-6, rtol=2e-6)
    return np.stack(views), names, checks, float(np.max(np.abs(full.values-raw_val)))


def build(args):
    # Header/source and the experiment's OWN outer selection are checked before
    # either historical loader can open an outer NPZ.
    from improve import tabicl_graph_context as experiment
    out = Path(args.out_dir).resolve()
    p = experiment.header(out)
    if args.phase == 'outer':
        experiment.selection(out, p)
    if args.task not in p['tasks'] or args.phase not in ('inner', 'outer'):
        raise ValueError('Unknown feature task or phase')
    fp = feature_parent.protocol(p['feature_reference_dir'])
    if fp['sha256'] != p['feature_reference_protocol_sha256']:
        raise ValueError('Feature reference protocol changed')
    gp = graph_parent.protocol(p['graph_reference_dir'])
    if gp['sha256'] != p['graph_reference_protocol_sha256']:
        raise ValueError('Graph reference protocol changed')
    dest = out/'features'/args.phase/args.task
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (dest/'manifest.json').exists():
            experiment.feature_manifest(out, p, args.task, args.phase)
            print('FEATURES_ALREADY_DONE', args.phase, args.task, flush=True)
            return
        arrays, graph, _ = graph_parent.stage_inputs(p['graph_reference_dir'], args.task, args.phase)
        ref = Path(p['feature_reference_dir'])/'features'/args.phase/args.task
        manifest = base.read_json(ref/'manifest.json')
        for name, expected in manifest['files'].items():
            if base.digest(ref/name) != expected:
                raise ValueError('Reference feature file changed')
        with np.load(ref/'arrays.npz', allow_pickle=False) as archive:
            if any('test' in key.lower() for key in archive.files):
                raise ValueError('Test arrays are forbidden')
            cached = {key:archive[key].copy() for key in archive.files}
        for key in ('pairs_train', 'y_train', 'pairs_val', 'y_val'):
            np.testing.assert_array_equal(arrays[key], cached[key])
        encoder = PairFeatureEncoder(graph, args.task)
        raw, names, checks, full_error = build_views(encoder, arrays['pairs_train'], arrays['y_train'],
            cached['fold_ids'], arrays['pairs_val'], cached['raw_train'], cached['raw_val'])
        if names != manifest['names']:
            raise ValueError('Reference feature names changed')
        with (ref/'transform.pkl').open('rb') as stream:
            transformer = pickle.load(stream)
        transformed = transformer.transform(raw.reshape(-1, raw.shape[-1])).astype(np.float32).reshape(raw.shape)
        train_scaled = transformer.transform(cached['raw_train']).astype(np.float32)
        np.testing.assert_allclose(train_scaled, cached['x_train'], atol=2e-6, rtol=2e-6)
        np.savez_compressed(dest/'arrays.npz', raw_query_views=raw, x_query_views=transformed,
            fold_ids=cached['fold_ids'], pairs_val=cached['pairs_val'])
        audit = dict(protocol_sha256=p['sha256'], phase=args.phase, task=args.task, folds=checks,
            passed=True, full_query_feature_max_abs=full_error,
            transformed_context_max_abs=float(np.max(np.abs(train_scaled-cached['x_train']))),
            no_test_evaluation=True, query_labels_used_by_encoder=False,
            unchanged_other_task_policy=True, feature_python=sys.executable,
            created_at_utc=datetime.now(timezone.utc).isoformat())
        base.write_json(dest/'AUDIT.json', audit)
        base.write_json(dest/'manifest.json', dict(protocol_sha256=p['sha256'], phase=args.phase,
            task=args.task, reference_manifest_sha256=base.digest(ref/'manifest.json'), names=names,
            files={name:base.digest(dest/name) for name in ('arrays.npz', 'AUDIT.json')}))
        print('FEATURES', args.phase, args.task, raw.shape, 'max_context_error',
              max(item['context_feature_max_abs'] for item in checks), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--task', choices=('LncDrug', 'MiDrug'), required=True)
    parser.add_argument('--phase', choices=('inner', 'outer'), required=True)
    build(parser.parse_args())


if __name__ == '__main__':
    main()
