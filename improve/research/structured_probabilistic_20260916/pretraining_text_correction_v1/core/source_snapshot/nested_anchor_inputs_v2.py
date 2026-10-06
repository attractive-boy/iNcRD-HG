"""Versioned, nested training inputs for a strong-reference residual offset.

Only the current task's fitting observations are opened. For each existing
neural fold Q, a new five-fold feature crossfit is constructed inside S = T minus Q.
No Q outcome, evaluation outcome, or existing full-context training prediction
is accepted by the feature builder. This is infrastructure, not an innovation.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from improve import tabular_foundation as base
from improve.graph_tabm_modules import PairFeatureEncoder, crossfit_features, fit_transform
from improve.tabicl_graph_context_modules import support_plan

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'results/nested_anchor_v2_20260915/historical'
TRAIN_KEYS = frozenset(('pairs_train', 'y_train', 'edges', 'relation', 'target_codes',
    'node_counts', 'pair_offset', 'nd', 'folds', 'x_0', 'x_1', 'x_2', 'x_3'))
SOURCES = ('nested_anchor_inputs_v2.py', 'nested_anchor_fit_v2.py',
    'test_nested_anchor_v2.py', 'graph_tabm_modules.py', 'tabicl_graph_context_modules.py',
    'tabicl_context_program_modules.py', 'tabicl_pathway_modules.py',
    'context_anchor_verify.py', 'anchor_shrinkage.py', 'anchor_prediction_precision.py',
    'tabular_foundation.py')


def load_selected(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in keys}


class FeatureGraph:
    """Minimal read-only graph view; does not import historical PyG pickles."""
    def __init__(self, arrays, description, support_positive):
        self.node_types = description['node_types']
        counts = arrays['node_counts']
        offsets = dict(zip(self.node_types, np.r_[0, np.cumsum(counts)[:-1]]))
        self.nodes = {nt: SimpleNamespace(num_nodes=int(counts[i]),
            x=torch.from_numpy(arrays['x_' + str(i)].copy()))
            for i, nt in enumerate(self.node_types)}
        self.edge_index_dict = {}
        target_ids = np.unique(arrays['relation'][arrays['target_codes'] >= 0])
        if len(target_ids) != 1:
            raise ValueError('Exactly one target physical relation is required')
        for i, key in enumerate(description['physical_relations']):
            src, rel, dst = key
            if i == target_ids[0]:
                edges = support_positive.copy()
            else:
                edges = arrays['edges'][:, arrays['relation'] == i].copy()
                edges -= np.array([offsets[src], offsets[dst]])[:, None]
                if src == dst:
                    edges = np.concatenate([edges, edges[::-1]], axis=1)
            if src == dst:
                # The archived feature graph has two loops per similarity
                # node and one per CircHost node. Physical arrays omit them.
                multiplicity = 1 if rel == 'CircHost' else 2
                loops = np.tile(np.arange(self.nodes[src].num_nodes), multiplicity)
                edges = np.concatenate([edges, np.stack([loops, loops])], axis=1)
            self.edge_index_dict[tuple(key)] = torch.from_numpy(edges)
        # Matches the observed historical edge iteration order: all physical
        # relations first, then cross-type reverse relations in the same order.
        for key, edges in list(self.edge_index_dict.items()):
            src, rel, dst = key
            if src != dst:
                self.edge_index_dict[(dst, 'rev_' + rel, src)] = edges.flip(0)

    def __getitem__(self, key):
        return self.nodes[key] if isinstance(key, str) else SimpleNamespace(
            edge_index=self.edge_index_dict[key])


def build_nested(graph, task, support_pairs, support_y, query_pairs,
                 n_folds=5, feature_seed=20260910):
    """The signature intentionally has no query labels or outer-fold labels."""
    if n_folds != 5:
        raise ValueError('The frozen reference recipe requires five inner folds')
    codes = lambda x: x[0] * graph['drug'].num_nodes + x[1]
    if np.intersect1d(codes(support_pairs), codes(query_pairs)).size:
        raise ValueError('Reference fitting/query pairs overlap')
    rna = 'miRNA' if task == 'MiDrug' else 'lncRNA'
    physical = graph[rna, task, 'drug'].edge_index.numpy()
    if set(codes(physical)) != set(codes(support_pairs[:, support_y == 1])):
        raise ValueError('Graph target relation must be exactly support positives')
    raw, raw_query, names, folds = crossfit_features(graph, task, support_pairs,
        support_y, query_pairs, n_folds, feature_seed)
    encoder = PairFeatureEncoder(graph, task)
    views, audits = [], []
    for item in support_plan(support_pairs, support_y, folds, query_pairs):
        idx, pos = item['context'], item['support']
        combined = np.concatenate([support_pairs[:, idx], query_pairs], axis=1)
        frame = encoder.transform(combined, support_pairs[:, pos])
        if list(frame.columns) != names:
            raise ValueError('Feature column ordering changed')
        np.testing.assert_allclose(frame.values[:len(idx)], raw[idx], atol=2e-6, rtol=2e-6)
        overlap = np.intersect1d(codes(combined), codes(support_pairs[:, pos])).size
        if overlap:
            raise ValueError('Nested fitting/evaluation query fact is in graph support')
        views.append(frame.values[len(idx):].copy())
        audits.append(dict(fold=int(item['fold']), context_rows=len(idx),
            support_positive_rows=len(pos), forbidden_fact_overlap=int(overlap),
            context_feature_error=float(abs(frame.values[:len(idx)] - raw[idx]).max())))
    views = np.stack(views)
    x, xq, transformer = fit_transform(raw, raw_query)
    xviews = transformer.transform(views.reshape(-1, len(names))).astype(np.float32).reshape(views.shape)
    arrays = dict(pairs_train=support_pairs, y_train=support_y, pairs_query=query_pairs,
        raw_train=raw, raw_query=raw_query, raw_query_views=views,
        x_train=x, x_query=xq, x_query_views=xviews, fold_ids=folds)
    return arrays, transformer, dict(feature_names=names, folds=audits,
        query_outcomes_accepted=False, scaler_fitted_on='Nested OOF support features only')


def freeze(out, scope):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new cache directory')
    dp_path = ROOT / 'results/mechanism_validation_20260915/protocol.json'
    dp = base.check_seal(base.read_json(dp_path))
    cells = []
    for task in ('MiDrug', 'LncDrug'):
        splits = ['historical'] if scope == 'historical' else dp['split_seeds']
        if scope == 'outer':
            splits = ['outer']
        for split in splits:
            if scope == 'historical':
                path = ROOT / 'results/observation_offset_inputs_20260915' / task / 'arrays.npz'
                description = ROOT / 'results/typed_sheaf_20260914/inputs' / task / 'graph.json'
                feature = ROOT / 'results/numeric_embedding_20260912/features/inner' / task / 'arrays.npz'
            elif scope == 'outer':
                path = ROOT / 'results/mechanism_outer_recheck_20260915/inputs' / task / 'arrays.npz'
                description = ROOT / 'results/mechanism_validation_20260915/inputs' / task / '61453/graph.json'
                feature = ROOT / 'results/numeric_embedding_20260912/features/outer' / task / 'arrays.npz'
            else:
                folder = ROOT / 'results/mechanism_validation_20260915'
                cell = folder / 'inputs' / task / str(split)
                path, description, feature = cell / 'arrays.npz', cell / 'graph.json', cell / 'arrays.npz'
            for required in (path, description, feature):
                if not required.exists():
                    raise FileNotFoundError(required)
            cells.append(dict(task=task, split=str(split), source=str(path),
                description=str(description), feature_source=str(feature),
                source_sha256=base.digest(path), description_sha256=base.digest(description),
                feature_source_sha256=base.digest(feature)))
    evidence = Path(dp['evidence_dir'])
    evidence_manifest = base.read_json(evidence / 'INPUTS.json')
    if evidence_manifest['labels_read']:
        raise ValueError('Static context evidence may not use benchmark outcomes')
    inputs = [dp_path, evidence / 'INPUTS.json', Path(dp['icl_protocol']['checkpoint']),
              ROOT / 'research/new_mechanisms_20260915/residual_frontier/SELF_LOOP_AUDIT.json']
    inputs.extend(evidence / name for name in evidence_manifest['files'])
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        version='nested_anchor_v2', scope=scope, cells=cells,
        correction='v1 feature reconstruction stopped before fitting: physical arrays omit archived self loops. Restore audited per-node loop multiplicities to reproduce original degree columns.',
        members=32, reference_seed=20260914, inner_folds=5, feature_fold_seed=20260910,
        graph_recipes=dp['graph_recipes'], input_configs=dp['input_configs'],
        evidence_dir=str(evidence), anchor=dp['anchor'],
        icl_protocol={k: dp['icl_protocol'][k] for k in ('checkpoint', 'implementation', 'inference')},
        source_sha256={n: base.digest(ROOT / n) for n in SOURCES},
        input_sha256={str(f): base.digest(f) for f in inputs},
        allowed_training_source_keys=sorted(TRAIN_KEYS),
        reconstruction_audit_keys=['pairs_train', 'pairs_val', 'raw_train', 'raw_val', 'fold_ids'],
        fit_policy='For neural fold Q, fit the same fixed strong reference on S = T minus Q. All target-dependent feature training rows are inner-five-fold OOF inside S; Q features use S support only. Fit quantile preprocessing on S OOF features. Static evidence and non-target relation policy unchanged.',
        composition='.7 * logit(.5 * graph_context + .5 * tiered_context) + .3 * mean(classic_member_logits)',
        infrastructure_not_innovation=True, query_outcome_access=False,
        source_evaluation_outcomes_opened=False, original_test_source_access=False,
        historical_outer_results_previously_seen=True, existing_default_replaced=False))
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT / name, out / 'source_snapshot' / name)
    base.write_json(out / 'protocol.json', p)
    print('FROZEN_NESTED_REFERENCE', scope, p['sha256'], flush=True)


def protocol(out):
    p = base.check_seal(base.read_json(Path(out) / 'protocol.json'))
    for name, expected in p['source_sha256'].items():
        if base.digest(ROOT / name) != expected:
            raise ValueError('Frozen nested reference source changed: ' + name)
    for name, expected in p['input_sha256'].items():
        if base.digest(name) != expected:
            raise ValueError('Frozen reference input changed: ' + name)
    for cell in p['cells']:
        for key in ('source', 'description', 'feature_source'):
            if base.digest(cell[key]) != cell[key + '_sha256']:
                raise ValueError('Frozen cell input changed')
    return p


def verify_files(dest, record):
    for name, expected in record['files'].items():
        if base.digest(Path(dest) / name) != expected:
            raise ValueError('Nested reference artifact changed: ' + name)


def features(out, task):
    from improve.tabicl_context_program_modules import ContextPrograms
    out = Path(out)
    p = protocol(out)
    evidence = ContextPrograms.load(p['evidence_dir'])
    for cell in [c for c in p['cells'] if c['task'] == task]:
        a = load_selected(cell['source'], TRAIN_KEYS)
        description = base.read_json(cell['description'])
        dest = out / 'inputs' / task / cell['split']
        dest.mkdir(parents=True, exist_ok=True)
        audit_path = dest / 'RECONSTRUCTION.json'
        if not audit_path.exists():
            old = load_selected(cell['feature_source'], p['reconstruction_audit_keys'])
            np.testing.assert_array_equal(a['pairs_train'], old['pairs_train'])
            full_graph = FeatureGraph(a, description, a['pairs_train'][:, a['y_train'] == 1])
            encoder = PairFeatureEncoder(full_graph, task)
            reconstructed = encoder.transform(old['pairs_val'], a['pairs_train'][:, a['y_train'] == 1])
            np.testing.assert_allclose(reconstructed.values, old['raw_val'], atol=2e-6, rtol=2e-6)
            errors = []
            for f in range(5):
                rows = np.flatnonzero(old['fold_ids'] == f)
                positive = (old['fold_ids'] != f) & (a['y_train'] == 1)
                raw = encoder.transform(a['pairs_train'][:, rows], a['pairs_train'][:, positive]).values
                np.testing.assert_allclose(raw, old['raw_train'][rows], atol=2e-6, rtol=2e-6)
                errors.append(float(abs(raw - old['raw_train'][rows]).max()))
            base.write_json(audit_path, base.seal(dict(protocol_sha256=p['sha256'],
                full_feature_max_abs=float(abs(reconstructed.values - old['raw_val']).max()),
                oof_feature_max_abs=max(errors), feature_names=list(reconstructed.columns),
                all_five_original_folds_reconstructed=True, query_labels_read=False,
                edge_type_order=[list(k) for k in full_graph.edge_index_dict])))
            print('FULL_SUPPORT_RECONSTRUCTION', task, cell['split'], max(errors), flush=True)
        for f in sorted(np.unique(a['folds'])):
            fold_dest = dest / str(f)
            if (fold_dest / 'manifest.json').exists():
                record = base.check_seal(base.read_json(fold_dest / 'manifest.json'))
                assert record['protocol_sha256'] == p['sha256']
                verify_files(fold_dest, record)
                continue
            query = np.flatnonzero(a['folds'] == f)
            support = np.flatnonzero(a['folds'] != f)
            pairs, y = a['pairs_train'][:, support], a['y_train'][support]
            graph = FeatureGraph(a, description, pairs[:, y == 1])
            nested, transformer, audit = build_nested(graph, task, pairs, y,
                a['pairs_train'][:, query], p['inner_folds'], p['feature_fold_seed'])
            prefix = 'raw' if p['input_configs'][task]['input_view'] == 'raw' else 'x'
            for suffix, pair_key in (('train', 'pairs_train'), ('query', 'pairs_query')):
                nested['tiered_' + suffix] = evidence.transform(task, nested[pair_key], 'tiered')
                nested['context76_' + suffix] = np.concatenate([nested[prefix + '_' + suffix],
                    evidence.transform(task, nested[pair_key], 'context')], axis=1).astype(np.float32)
            nested.update(support_indices=support, query_indices=query)
            if nested['context76_train'].shape[1] != 76:
                raise ValueError('Reference context column count changed')
            fold_dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fold_dest / 'arrays.npz', **nested)
            with (fold_dest / 'transform.pkl').open('wb') as stream:
                pickle.dump(transformer, stream, protocol=4)
            base.write_json(fold_dest / 'AUDIT.json', dict(**audit, outer_fold=int(f),
                support_rows=len(support), query_rows=len(query), query_outcomes_read=False,
                full_target_support_overlap=0, static_source_uses_labels=False))
            record = base.seal(dict(protocol_sha256=p['sha256'], task=task,
                split=cell['split'], fold=int(f), files={n: base.digest(fold_dest / n)
                for n in ('arrays.npz', 'transform.pkl', 'AUDIT.json')}))
            base.write_json(fold_dest / 'manifest.json', record)
            print('NESTED_FEATURES', task, cell['split'], int(f), len(support), len(query), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'features'))
    parser.add_argument('--out', default=str(DEFAULT))
    parser.add_argument('--scope', choices=('historical', 'partitions', 'outer'), default='historical')
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.out, args.scope)
    else:
        features(args.out, args.task)
