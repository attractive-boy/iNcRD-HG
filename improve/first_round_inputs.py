"""Shared, frozen inputs for the three first-round feature comparisons."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from improve.network_feature_inputs import (digest_array, graph_weights,
    minmax_rows, similarity_matrices, split_fingerprints, write_json)
from improve.run import TASKS


FOLDS = ('train', 'val', 'test')
ARMS = ('base', 'drug_unmasked', 'drug_masked', 'expr_pca64_stats', 'edge_weight')
COMPARISONS = {
    'drug_pooling': ('drug_unmasked', 'drug_masked'),
    'expression': ('base', 'expr_pca64_stats'),
    'continuous_edges': ('base', 'edge_weight'),
}
SIM_RELATIONS = {
    'lncRNA': ('lncRNA', 'LncLnc', 'lncRNA'),
    'miRNA': ('miRNA', 'MiMi', 'miRNA'),
    'drug': ('drug', 'DrugDrug', 'drug'),
}
STRUCTURE_COLUMNS = (
    'log_train_degree_rna', 'log_train_degree_drug', 'degree_product',
    'rna_support', 'rna_support_mean', 'drug_support', 'drug_support_mean',
    'both_support', 'log_train_path3', 'log_cross_rna_path2',
    'cross_rna_similarity_support', 'cross_rna_drug_similarity_support',
    'mean_neighbour_degree_rna', 'mean_neighbour_degree_drug',
)


def clean_training_pairs(rows):
    """Preserve evaluation rows; remove train duplicates and eval overlap once."""
    evaluation = set()
    for fold in ('val', 'test'):
        pairs_set = set(map(tuple, rows[fold][0].T.tolist()))
        if len(pairs_set) != len(rows[fold][1]) or evaluation.intersection(pairs_set):
            raise ValueError('Evaluation pairs overlap or repeat; an explicit new split is needed')
        evaluation.update(pairs_set)
    pairs, y = rows['train']
    labels = {}
    keep, removed = [], []
    for index, (pair, label) in enumerate(zip(pairs.T.tolist(), y)):
        pair = tuple(pair)
        if pair in labels and labels[pair] != label:
            raise ValueError('Conflicting training labels for the same pair')
        reason = 'overlaps_evaluation' if pair in evaluation else 'duplicate_training_pair' if pair in labels else None
        labels[pair] = label
        if reason:
            removed.append(dict(row=index, pair=list(pair), label=int(label), reason=reason))
        else:
            keep.append(index)
    keep = np.asarray(keep, dtype=np.int64)
    cleaned = dict(rows, train=(pairs[:, keep].copy(), y[keep].copy()))
    audit = {
        'policy': 'Remove training copies of evaluation pairs and retain the first row of each '
                  'remaining training pair; validation/test rows unchanged; no resplitting',
        'original_training_rows': len(y), 'clean_training_rows': len(keep),
        'removed_rows': removed, 'retained_original_indices_sha256': digest_array(keep),
    }
    return cleaned, audit


def load_frozen_splits(run_dir, task):
    """Keep released eval pairs/negatives; freeze an audited clean training graph."""
    key = (TASKS[task]['rna'], task, 'drug')
    paths = [Path(run_dir) / '3.heter_data' / (task + '_' + f + '_data.pth')
             for f in FOLDS]
    splits = [torch.load(p, map_location='cpu') for p in paths]
    train = splits[0]
    rows = {}
    for fold, data in zip(FOLDS, splits):
        if data.metadata() != train.metadata():
            raise ValueError('Node/relation metadata differ across frozen splits')
        for nt in train.node_types:
            torch.testing.assert_close(data[nt].x, train[nt].x, rtol=0, atol=0)
            if 'node_id' in train[nt]:
                torch.testing.assert_close(data[nt].node_id, train[nt].node_id)
        pairs = data[key].edge_label_index.numpy().copy()
        y = data[key].edge_label.numpy().copy()
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError('Both binary classes must be present in ' + fold)
        rows[fold] = (pairs, y)
    train_edges = set(map(tuple, train[key].edge_index.T.tolist()))
    pairs, y = rows['train']
    if train_edges != set(map(tuple, pairs[:, y == 1].T.tolist())):
        raise ValueError('Training graph must contain exactly the training positives')
    reverse = ('drug', 'rev_' + task, key[0])
    if train_edges != set(map(tuple, train[reverse].edge_index.flip(0).T.tolist())):
        raise ValueError('Forward and reverse training target edges disagree')
    rows, cleanup = clean_training_pairs(rows)
    train = train.clone()
    pairs, y = rows['train']
    train[key].edge_label_index = torch.from_numpy(pairs)
    train[key].edge_label = torch.from_numpy(y)
    positive = set(map(tuple, pairs[:, y == 1].T.tolist()))
    retained, seen = [], set()
    for index, pair in enumerate(train[key].edge_index.T.tolist()):
        pair = tuple(pair)
        if pair in positive and pair not in seen:
            retained.append(index)
            seen.add(pair)
    if seen != positive:
        raise ValueError('Clean training positives do not match graph')
    train[key].edge_index = train[key].edge_index[:, retained]
    train[reverse].edge_index = train[key].edge_index.flip(0).clone()
    train_edges = positive
    for fold in ('val', 'test'):
        if train_edges.intersection(map(tuple, rows[fold][0].T.tolist())):
            raise ValueError('Held-out pair appears in the training target graph')
    audit = {
        'released_split_fingerprints': split_fingerprints(splits, key),
        'training_cleanup': cleanup,
        'message_graph': 'training graph for train, validation and test',
        'message_graph_sha256': {
            '|'.join(k): digest_array(v.numpy()) for k, v in train.edge_index_dict.items()},
        'released_target_edge_counts': {
            f: int(d[key].edge_index.shape[1]) for f, d in zip(FOLDS, splits)},
        'label_counts': {
            f: {'total': len(y), 'positive': int(y.sum())} for f, (_, y) in rows.items()},
    }
    return train, key, rows, audit


def fit_expression_array(raw, fit_ids, components=64):
    """PCA and all scalers see training nodes only, including output scaling."""
    raw = np.asarray(raw, dtype=np.float32)
    fit_ids = np.asarray(fit_ids, dtype=np.int64)
    if not np.isfinite(raw).all() or min(len(fit_ids), raw.shape[1]) <= components:
        raise ValueError('Insufficient training nodes/features or non-finite expression')
    scaler = StandardScaler().fit(raw[fit_ids])
    scaled = scaler.transform(raw)
    pca = PCA(n_components=components, svd_solver='randomized', random_state=0)
    pca.fit(scaled[fit_ids])
    z = pca.transform(scaled)
    pc_scaler = StandardScaler().fit(z[fit_ids])
    stats = np.stack([raw.mean(1), raw.std(1), (raw == 0).mean(1)], axis=1)
    stat_scaler = StandardScaler().fit(stats[fit_ids])
    result = np.concatenate([pc_scaler.transform(z), stat_scaler.transform(stats)], axis=1)
    parameters = {
        'fit_ids': fit_ids, 'input_mean': scaler.mean_, 'input_scale': scaler.scale_,
        'pca_mean': pca.mean_, 'pca_components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'component_mean': pc_scaler.mean_, 'component_scale': pc_scaler.scale_,
        'statistics_mean': stat_scaler.mean_, 'statistics_scale': stat_scaler.scale_,
    }
    return result.astype(np.float32), parameters


def expression_inputs(train, key, rows, net_dir, components=64):
    replacements, parameters, audit = {}, {}, {}
    for nt, filename in (('lncRNA', 'LncExp.csv'), ('miRNA', 'MiExp.csv')):
        frame = pd.read_csv(Path(net_dir) / filename, index_col=0).T
        names = pd.read_csv(Path(net_dir) / (nt + '.csv'))['x'].str.upper().tolist()
        frame.index = frame.index.str.upper()
        if frame.index.has_duplicates or len(set(names)) != len(names):
            raise ValueError('Ambiguous RNA identifiers')
        frame = frame.loc[names]
        if nt == 'miRNA':
            frame = frame[sorted(frame.columns, key=lambda c: int(c.split('-')[-1]))]
        raw = frame.values.astype(np.float32)
        np.testing.assert_allclose(minmax_rows(raw), train[nt].x.numpy(),
                                   rtol=2e-5, atol=2e-6)
        if nt == key[0]:
            fit_ids = np.unique(rows['train'][0][0])
        else:
            relation = 'LncDrug' if nt == 'lncRNA' else 'MiDrug'
            fit_ids = np.unique(train[nt, relation, 'drug'].edge_index[0].numpy())
        replacements[nt], params = fit_expression_array(raw, fit_ids, components)
        parameters.update({nt + '_' + k: v for k, v in params.items()})
        audit[nt] = {
            'fit_nodes': len(fit_ids), 'fit_ids_sha256': digest_array(fit_ids),
            'raw_sha256': digest_array(raw), 'input_dim': raw.shape[1],
            'output_dim': components + 3,
            'explained_variance_ratio_sum': float(params['explained_variance_ratio'].sum()),
            'statistics': ['raw_expression_mean', 'raw_expression_std_ddof0', 'raw_zero_fraction'],
        }
    return replacements, parameters, audit


def dense_relation(data, key):
    matrix = np.zeros((data[key[0]].num_nodes, data[key[2]].num_nodes), dtype=np.float32)
    src, dst = data[key].edge_index.numpy()
    matrix[src, dst] = 1.0
    return matrix


def incoming_similarity(data, key, weights):
    """Rows are destinations, columns sources; use exactly the GNN weights.

    Exclude self terms from label-derived pair scores. GNN self edges remain.
    Destination normalization can make this matrix asymmetric.
    """
    matrix = np.zeros((data[key[2]].num_nodes, data[key[0]].num_nodes), dtype=np.float32)
    src, dst = data[key].edge_index.numpy()
    np.add.at(matrix, (dst, src), weights[key].cpu().numpy())
    np.fill_diagonal(matrix, 0.0)
    return matrix


def structure_features(pairs, A, R, D, bridge, weighted_bridge):
    """Fixed statistics of one graph; no dense off-graph similarities."""
    r, d = pairs
    deg_r, deg_d = A.sum(1), A.sum(0)
    lr, ld = np.log1p(deg_r[r]), np.log1p(deg_d[d])
    ra, ad = R @ A, A @ D.T
    path3 = (A @ A.T) @ A
    columns = [
        lr, ld, lr * ld, ra[r, d], ra[r, d] / np.maximum(R.sum(1)[r], 1e-8),
        ad[r, d], ad[r, d] / np.maximum(D.sum(1)[d], 1e-8),
        (ra @ D.T)[r, d], np.log1p(path3[r, d]), np.log1p(bridge[r, d]),
        weighted_bridge[r, d], (weighted_bridge @ D.T)[r, d],
        (A @ deg_d)[r] / np.maximum(deg_r[r], 1),
        (A.T @ deg_r)[d] / np.maximum(deg_d[d], 1),
    ]
    return np.column_stack(columns).astype(np.float32)


def crossfit_structure(rows, A, R, D, bridge, weighted_bridge, n_folds=5):
    pairs, y = rows['train']
    assignment = np.empty(len(y), dtype=np.int64)
    features = {'train': np.empty((len(y), len(STRUCTURE_COLUMNS)), dtype=np.float32)}
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for fold_id, (_, held_ids) in enumerate(splitter.split(np.zeros(len(y)), y)):
        assignment[held_ids] = fold_id
        held = pairs[:, held_ids[y[held_ids] == 1]]
        Ak = A.copy()
        Ak[held[0], held[1]] = 0
        if Ak[held[0], held[1]].any():
            raise AssertionError('Queried training positives leaked into pair statistics')
        features['train'][held_ids] = structure_features(
            pairs[:, held_ids], Ak, R, D, bridge, weighted_bridge)
    for fold in ('val', 'test'):
        features[fold] = structure_features(rows[fold][0], A, R, D, bridge, weighted_bridge)
    return features, assignment


def pair_features(rows, nodes, rna, structures):
    """Give both simple models the COMPLETE endpoint vectors, then graph stats."""
    raw = {
        f: np.concatenate([nodes[rna][pairs[0]], nodes['drug'][pairs[1]], structures[f]], axis=1)
        for f, (pairs, _) in rows.items()
    }
    scaler = StandardScaler().fit(raw['train'])
    features = {f: scaler.transform(x).astype(np.float32) for f, x in raw.items()}
    layout = {
        'rna_vector': [0, nodes[rna].shape[1]],
        'drug_vector': [nodes[rna].shape[1], nodes[rna].shape[1] + nodes['drug'].shape[1]],
        'structure': list(STRUCTURE_COLUMNS),
    }
    return features, {'mean': scaler.mean_, 'scale': scaler.scale_}, layout


def pooling_inputs(path, net_dir, n_drugs):
    expected = pd.read_csv(Path(net_dir) / 'Drug.csv')['CID'].astype(str).to_numpy()
    with np.load(path, allow_pickle=False) as pools:
        np.testing.assert_array_equal(pools['ids'].astype(str), expected)
        if pools['masked'].shape != pools['unmasked'].shape or len(expected) != n_drugs:
            raise ValueError('Pooling controls must have identical drugs and dimensions')
        result = {name: minmax_rows(pools[name]) for name in ('unmasked', 'masked')}
    if any(not np.isfinite(x).all() for x in result.values()):
        raise ValueError('Non-finite drug embeddings')
    return result


def prepare_task(run_dir, net_dir, out_dir, task, components=64):
    out = Path(out_dir)
    dest = out / 'inputs' / task
    dest.mkdir(parents=True, exist_ok=True)
    train, key, rows, audit = load_frozen_splits(run_dir, task)
    original = {nt: x.numpy().copy() for nt, x in train.x_dict.items()}
    for x in original.values():
        if not np.isfinite(x).all():
            raise ValueError('Non-finite original node features')
    pools = pooling_inputs(out / 'drug_pooling.npz', net_dir, train['drug'].num_nodes)
    expr, expr_params, expr_audit = expression_inputs(train, key, rows, net_dir, components)
    np.savez_compressed(dest / 'expression_transform.npz', **expr_params)
    nodes = {
        'base': original, 'drug_unmasked': dict(original, drug=pools['unmasked']),
        'drug_masked': dict(original, drug=pools['masked']),
        'expr_pca64_stats': dict(original, **expr), 'edge_weight': original,
    }
    # A single serialized training graph is shared by every arm and eval fold.
    torch.save(train, dest / 'train_graph.pth')
    matrices = similarity_matrices(net_dir)
    weight_sets = {weighted: graph_weights(train, matrices, weighted) for weighted in (False, True)}
    for edge, values in weight_sets[True].items():
        src, dst = train[edge].edge_index
        totals = torch.zeros(train[edge[2]].num_nodes).index_add_(0, dst, values)
        counts = torch.bincount(dst, minlength=len(totals)).float()
        torch.testing.assert_close(totals, counts, rtol=2e-5, atol=2e-5)
    other = 'miRNA' if key[0] == 'lncRNA' else 'lncRNA'
    other_task = 'MiDrug' if other == 'miRNA' else 'LncDrug'
    B = dense_relation(train, ('lncRNA', 'LncMi', 'miRNA'))
    if key[0] == 'miRNA':
        B = B.T
    other_A = dense_relation(train, (other, other_task, 'drug'))
    bridge = B @ other_A
    A = dense_relation(train, key)
    structure_sets, assignments = {}, {}
    for weighted, weights in weight_sets.items():
        R = incoming_similarity(train, SIM_RELATIONS[key[0]], weights)
        D = incoming_similarity(train, SIM_RELATIONS['drug'], weights)
        O = incoming_similarity(train, SIM_RELATIONS[other], weights)
        structure_sets[weighted], assignments[weighted] = crossfit_structure(
            rows, A, R, D, bridge, B @ O @ other_A)
    np.testing.assert_array_equal(assignments[False], assignments[True])
    audit.update(task=task, expression=expr_audit, arms={},
                 crossfit_folds=5, crossfit_seed=0,
                 crossfit_assignment_sha256=digest_array(assignments[False]),
                 edge_types=[list(k) for k in train.edge_types],
                 pair_input='Complete RNA/drug vectors plus 14 fixed train-graph statistics; '
                            'pair StandardScaler fit only on training rows',
                 continuous_weights='Absolute similarity on existing LncLnc/MiMi/DrugDrug edges; '
                                    'incoming mass per relation preserved; all other edges have weight 1')
    np.save(dest / 'crossfit_assignment.npy', assignments[False])
    for arm in ARMS:
        weighted = arm == 'edge_weight'
        X, scaler, layout = pair_features(rows, nodes[arm], key[0], structure_sets[weighted])
        payload = {'x_' + nt: x for nt, x in nodes[arm].items()}
        payload.update({'phi_' + f: x for f, x in X.items()})
        payload.update({'pairs_' + f: pairs for f, (pairs, _) in rows.items()})
        payload.update({'y_' + f: y for f, (_, y) in rows.items()})
        payload.update({'weight_' + str(i): weight_sets[weighted][k].numpy()
                        for i, k in enumerate(train.edge_types)})
        payload.update({'pair_scaler_' + k: v for k, v in scaler.items()})
        if any(not np.isfinite(x).all() for x in payload.values()):
            raise ValueError('Non-finite prepared input')
        np.savez_compressed(dest / (arm + '.npz'), **payload)
        audit['arms'][arm] = {
            'node_dims': {nt: int(x.shape[1]) for nt, x in nodes[arm].items()},
            'node_sha256': {nt: digest_array(x) for nt, x in nodes[arm].items()},
            'pair_dim': int(X['train'].shape[1]), 'pair_layout': layout,
            'pair_sha256': {f: digest_array(x) for f, x in X.items()},
            'structure_sha256': {f: digest_array(x) for f, x in structure_sets[weighted].items()},
            'weight_sha256': {'|'.join(k): digest_array(w.numpy())
                              for k, w in weight_sets[weighted].items()},
        }
        print('prepared', task, arm, 'pair_dim', X['train'].shape[1], flush=True)
    # Fail closed if an arm accidentally bundles an unrelated change.
    aa = audit['arms']
    for name, (control, treatment) in COMPARISONS.items():
        if name == 'continuous_edges':
            assert aa[control]['node_sha256'] == aa[treatment]['node_sha256']
            assert aa[control]['weight_sha256'] != aa[treatment]['weight_sha256']
            assert aa[control]['structure_sha256'] != aa[treatment]['structure_sha256']
        else:
            assert aa[control]['weight_sha256'] == aa[treatment]['weight_sha256']
            assert aa[control]['structure_sha256'] == aa[treatment]['structure_sha256']
            changed = {'drug'} if name == 'drug_pooling' else {'lncRNA', 'miRNA'}
            for nt in original:
                assert (aa[control]['node_sha256'][nt] != aa[treatment]['node_sha256'][nt]) == (nt in changed)
    write_json(dest / 'audit.json', audit)
    return audit
