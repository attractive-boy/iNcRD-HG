"""Audit min-max information loss and test its proposed validation mechanism.

Only RNA input vectors vary. The original GNN, seed 666, frozen training graph,
labels, optimizer, epoch budgets and single-checkpoint selection are retained.
This runner accepts training/validation arrays only; it never predicts on test.
The source and all inputs are frozen before any mechanism run starts.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from improve.first_round_features import (json_digest, load_protocol, sha256,
    source_hashes, verify_files, write_json)
from improve.first_round_inputs import fit_expression_array
from improve.network_feature_ablation import ExperimentModel, score
from improve.network_feature_inputs import digest_array
from improve.run import set_seed


TASKS = ('LncDrug', 'MiDrug')
RNA_TYPES = ('lncRNA', 'miRNA')
VAL_METRICS = ('val_auc', 'val_aupr', 'val_criterion')
ARMS = {
    'base': 'original row-minmax expression, 938 dimensions',
    'norm_zero': 'row-minmax expression + 3 zeros, 941 dimensions',
    'norm_raw_stats': 'row-minmax expression + raw mean/std/zero fraction, 941 dimensions',
    'norm_shape_stats': 'row-minmax expression + mean/std/zero fraction OF that normalized expression, 941 dimensions',
    'norm_shuffled_stats': 'row-minmax expression + jointly row-shuffled raw statistics, 941 dimensions',
    'rawpca_zero': '64 scaled PCs of column-standardized RAW expression + 3 zeros, 67 dimensions',
    'rawpca_stats': '64 scaled PCs of column-standardized RAW expression + raw statistics, 67 dimensions; exact first-round input',
    'rawpca_shuffled_stats': 'the same raw-expression PCs + jointly row-shuffled raw statistics, 67 dimensions',
    'normpca_zero': '64 scaled PCs of column-standardized ROW-NORMALIZED expression + 3 zeros, 67 dimensions',
    'normpca_stats': 'the same row-normalized-expression PCs + raw statistics, 67 dimensions',
}
CONTRASTS = {
    'full_shape_add_raw_statistics': ('norm_zero', 'norm_raw_stats'),
    'full_shape_add_redundant_summaries': ('norm_zero', 'norm_shape_stats'),
    'raw_vs_shape_statistics': ('norm_shape_stats', 'norm_raw_stats'),
    'full_shape_correct_vs_shuffled_statistics': ('norm_shuffled_stats', 'norm_raw_stats'),
    'raw_pca_add_statistics': ('rawpca_zero', 'rawpca_stats'),
    'raw_pca_correct_vs_shuffled_statistics': ('rawpca_shuffled_stats', 'rawpca_stats'),
    'normalized_pca_add_raw_statistics': ('normpca_zero', 'normpca_stats'),
    'pca_raw_vs_normalized_without_statistics': ('normpca_zero', 'rawpca_zero'),
}


def own_source_hashes():
    return dict(source_hashes(), expression_scale_mechanism=sha256(__file__))


def load_raw(net_dir, nt):
    filename = {'lncRNA': 'LncExp.csv', 'miRNA': 'MiExp.csv'}[nt]
    frame = pd.read_csv(Path(net_dir) / filename, index_col=0).T
    frame.index = frame.index.str.upper()
    names = pd.read_csv(Path(net_dir) / (nt + '.csv'))['x'].str.upper().tolist()
    if frame.index.has_duplicates or len(set(names)) != len(names):
        raise ValueError('Ambiguous RNA names')
    frame = frame.loc[names]
    if nt == 'miRNA':
        frame = frame[sorted(frame.columns, key=lambda c: int(c.split('-')[-1]))]
    raw = frame.values.astype(np.float32)
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite raw expression')
    return raw


def row_statistics(x):
    # Same float32 reduction and zero-fraction definition as the frozen arm.
    return np.stack([x.mean(1), x.std(1), (x == 0).mean(1)], axis=1)


def reconstruct_from_moments(normalized, mean, std):
    """Invert any positive affine row normalization, including epsilon min-max.

    Full normalized profiles and two ORIGINAL moments are required. This is
    not an inverse of a lossy PCA embedding. Constant rows use their mean.
    """
    normalized = np.asarray(normalized, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1, 1)
    std = np.asarray(std, dtype=np.float64).reshape(-1, 1)
    yn_mean = normalized.mean(1, keepdims=True)
    yn_std = normalized.std(1, keepdims=True)
    if np.any((yn_std == 0) & (std > 1e-12)):
        raise ValueError('A constant normalized row cannot recover positive raw variance')
    scale = np.divide(std, yn_std, out=np.zeros_like(std), where=yn_std > 0)
    return mean + scale * (normalized - yn_mean)


def reconstruction_errors(raw, recovered):
    raw = raw.astype(np.float64)
    error = recovered - raw
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {'max_absolute_error': float(np.abs(error).max()),
            'rmse': rmse,
            'relative_rmse': rmse / float(np.sqrt(np.mean(raw ** 2)))}


def math_audit(reference_dir, reference):
    ref = Path(reference_dir)
    result = {
        'normalization_source': 'heter_data_prepare.py:172-173',
        'normalization_source_sha256': sha256(Path(__file__).parent.parent / 'heter_data_prepare.py'),
        'raw_units': 'numerical units in the input expression CSV; not a claim of absolute molecule counts',
        'ideal_identity': 'N(a*x+b) = N(x), a>0, nonconstant x, no epsilon',
        'epsilon_caveat': 'A known nonzero epsilon retains a tiny scale signal in exact arithmetic; translation is still removed. Audit float32 denominators below.',
        'inverse': 'x = mean_raw + std_raw/std_normalized * (normalized - mean_normalized); constants use mean_raw',
        'inverse_limit': 'requires the full normalized profile, not only its 64 PCs',
        'pca_confound': 'the first-round PCA is fitted on RAW expression after per-column StandardScaler, so it can already carry raw level/amplitude information',
        'rna': {},
    }
    example = np.array([[1., 2., 3.], [12., 14., 16.]], dtype=np.float64)
    result['counterexample'] = {
        'raw': example.tolist(), 'normalized': ((example - example.min(1, keepdims=True)) / example.ptp(1, keepdims=True)).tolist(),
        'mean': example.mean(1).tolist(), 'std_ddof0': example.std(1).tolist(),
    }
    with np.load(ref / 'inputs' / 'LncDrug' / 'base.npz') as z:
        norm = {nt: z['x_' + nt].copy() for nt in RNA_TYPES}
    with np.load(ref / 'inputs' / 'MiDrug' / 'base.npz') as z:
        for nt in RNA_TYPES:
            np.testing.assert_array_equal(z['x_' + nt], norm[nt])
    for nt in RNA_TYPES:
        raw = load_raw(reference['net_dir'], nt)
        x64, yn = raw.astype(np.float64), norm[nt]
        ranges = raw.max(1) - raw.min(1)
        active = ranges > 0
        eps = np.float32(1e-8 if nt == 'lncRNA' else 0.)
        denominator = ranges + eps
        calculated = (raw - raw.min(1, keepdims=True)) / denominator[:, None]
        np.testing.assert_allclose(calculated, yn, rtol=2e-5, atol=2e-6)
        exact = reconstruct_from_moments(yn, x64.mean(1), x64.std(1))
        error = reconstruction_errors(raw, exact)
        if error['max_absolute_error'] > 2e-6:
            raise AssertionError('Full-profile inverse is unexpectedly inaccurate')
        item = {
            'shape': list(raw.shape), 'raw_sha256': digest_array(raw),
            'normalized_sha256': digest_array(yn),
            'raw_min': float(raw.min()), 'raw_max': float(raw.max()),
            'constant_rows': int((~active).sum()),
            'rows_with_raw_min_zero': int((raw.min(1) == 0).sum()),
            'raw_zero_fraction_min': float((raw == 0).mean(1).min()),
            'raw_zero_fraction_max': float((raw == 0).mean(1).max()),
            'normalized_vs_raw_zero_fraction_max_abs': float(np.abs((yn == 0).mean(1) - (raw == 0).mean(1)).max()),
            'nonconstant_rows': int(active.sum()),
            'minimum_positive_row_range': float(ranges[active].min()),
            'epsilon': float(eps),
            'nonconstant_float32_denominators_unchanged_by_epsilon': int((denominator[active] == ranges[active]).sum()),
            'normalization_formula_vs_cached_max_abs': float(np.abs(calculated - yn).max()),
            'inverse_with_float64_raw_moments': error,
            'inverse_with_cached_standardized_statistics': {},
        }
        # Measure the epsilon effect separately from rounding transformed raw data.
        def normalize64(x, eps64):
            return (x - x.min(1, keepdims=True)) / (x.ptp(1, keepdims=True) + eps64)
        x = x64[active]
        item['affine_a2_b5_max_abs_ideal_float64'] = float(np.abs(normalize64(2*x + 5, 0) - normalize64(x, 0)).max())
        item['affine_a2_b5_max_abs_epsilon_float64'] = float(np.abs(normalize64(2*x + 5, float(eps)) - normalize64(x, float(eps))).max())
        for task in TASKS:
            with np.load(ref / 'inputs' / task / 'expression_transform.npz') as p, np.load(ref / 'inputs' / task / 'expr_pca64_stats.npz') as z:
                stored = z['x_' + nt][:, -3:]
                moments = stored.astype(np.float64) * p[nt + '_statistics_scale'] + p[nt + '_statistics_mean']
            recovered = reconstruct_from_moments(yn, moments[:, 0], np.maximum(moments[:, 1], 0))
            stored_error = reconstruction_errors(raw, recovered)
            if stored_error['max_absolute_error'] > 1e-5:
                raise AssertionError('Cached raw moments fail reconstruction')
            item['inverse_with_cached_standardized_statistics'][task] = stored_error
        result['rna'][nt] = item
    return result


def block_shuffle_indices(n, fit_ids):
    """Joint shuffle within fit/non-fit nodes; never use held-out labels."""
    fit_ids = np.asarray(fit_ids, dtype=np.int64)
    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), fit_ids)
    rng = np.random.default_rng(0)
    index = np.arange(n, dtype=np.int64)
    for block in (fit_ids, remaining):
        index[block] = rng.permutation(block)
    return index


def val_only_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files if k.startswith(('x_', 'weight_'))
                or k in ('pairs_train', 'pairs_val', 'y_train', 'y_val')}


def reference_record(ref, reference, task, arm):
    path = ref / 'runs' / (task + '_gnn_' + arm + '_s666.json')
    record = json.loads(path.read_text())
    if record['protocol_sha256'] != reference['protocol_sha256']:
        raise ValueError('Reference result has the wrong input protocol')
    for name, digest in record['artifacts_sha256'].items():
        if sha256(ref / name) != digest:
            raise ValueError('Reference artifact changed: ' + name)
    return {'path': str(path), 'sha256': sha256(path),
            **{k: record[k] for k in VAL_METRICS + ('selected_epoch',)}}


def identity_for(reference, task, smoke):
    return {
        'task': task, 'seed': 666, 'model': 'original_gnn', 'smoke': bool(smoke),
        'arms': ARMS, 'contrasts_control_then_candidate': CONTRASTS,
        'reference_protocol_sha256': reference['protocol_sha256'],
        'source_sha256': own_source_hashes(),
        'training': dict(reference['spec']['task_settings'][task],
                         train_epochs=2 if smoke else reference['spec']['task_settings'][task]['train_epochs']),
        'optimizer': reference['spec']['gnn'],
        'selection': 'first strict maximum of (val AUC + val AUPR)/2; one checkpoint; full fixed budget',
        'evaluation_folds': ['val'],
        'preprocessing': 'PCA/scalers fitted only on frozen training nodes; PCA random_state=0; normalized-PC control uses the same recipe as raw PCs',
        'shuffle': 'numpy default_rng(0), joint 3-statistic row shuffle separately within fit and remaining nodes; no label access',
        'same_width_control': 'initial parameter hash and parameter count must match for all equal-width arms of a task',
        'decision': 'compare both validation metrics on both tasks; fixed split and one training seed, no claim of significance or independent generalization',
    }


def prepare_task(args, reference, task):
    ref, out = Path(args.reference_dir).resolve(), Path(args.out_dir).resolve()
    verify_files(ref, reference, prefix='inputs/' + task + '/')
    dest = out / task
    dest.mkdir(parents=True, exist_ok=True)
    identity = identity_for(reference, task, args.smoke)
    # Convert tuples to their JSON representation before identity comparison.
    identity = json.loads(json.dumps(identity))
    if (dest / 'config.json').exists():
        return load_config(args, reference, task)
    if any((dest / arm / 'result.json').exists() for arm in ARMS):
        raise ValueError('Refusing to overwrite completed runs without their config')
    base = val_only_arrays(ref / 'inputs' / task / 'base.npz')
    pca = val_only_arrays(ref / 'inputs' / task / 'expr_pca64_stats.npz')
    graph = torch.load(ref / 'inputs' / task / 'train_graph.pth', map_location='cpu')
    variants = {arm: dict(base) for arm in ARMS}
    parameters, audit = {}, {}
    with np.load(ref / 'inputs' / task / 'expression_transform.npz') as p:
        for nt in RNA_TYPES:
            fit_ids = p[nt + '_fit_ids']
            if nt == reference['spec']['task_settings'][task]['rna']:
                expected_ids = np.unique(base['pairs_train'][0])
            else:
                relation = 'LncDrug' if nt == 'lncRNA' else 'MiDrug'
                expected_ids = np.unique(graph[nt, relation, 'drug'].edge_index[0].numpy())
            np.testing.assert_array_equal(fit_ids, expected_ids)
            normalized = base['x_' + nt]
            raw = load_raw(reference['net_dir'], nt)
            raw_stats = pca['x_' + nt][:, -3:]
            stats_scaler = StandardScaler().fit(row_statistics(raw)[fit_ids])
            np.testing.assert_allclose(stats_scaler.mean_, p[nt + '_statistics_mean'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(stats_scaler.transform(row_statistics(raw)), raw_stats, atol=1e-6, rtol=1e-6)
            shape_repr, shape_params = fit_expression_array(normalized, fit_ids, components=64)
            permutation = block_shuffle_indices(len(raw), fit_ids)
            shuffled = raw_stats[permutation]
            zeros = np.zeros_like(raw_stats)
            raw_pc, shape_pc = pca['x_' + nt][:, :64], shape_repr[:, :64]
            features = {
                'base': normalized,
                'norm_zero': np.concatenate([normalized, zeros], 1),
                'norm_raw_stats': np.concatenate([normalized, raw_stats], 1),
                'norm_shape_stats': np.concatenate([normalized, shape_repr[:, -3:]], 1),
                'norm_shuffled_stats': np.concatenate([normalized, shuffled], 1),
                'rawpca_zero': np.concatenate([raw_pc, zeros], 1),
                'rawpca_stats': pca['x_' + nt],
                'rawpca_shuffled_stats': np.concatenate([raw_pc, shuffled], 1),
                'normpca_zero': np.concatenate([shape_pc, zeros], 1),
                'normpca_stats': np.concatenate([shape_pc, raw_stats], 1),
            }
            for arm, x in features.items():
                variants[arm]['x_' + nt] = x.astype(np.float32)
            parameters.update({nt + '_' + k: v for k, v in shape_params.items()})
            parameters[nt + '_shuffle_index'] = permutation
            audit[nt] = {
                'fit_nodes': len(fit_ids), 'fit_ids_sha256': digest_array(fit_ids),
                'raw_sha256': digest_array(raw),
                'raw_pc_explained_variance_ratio_sum': float(p[nt + '_explained_variance_ratio'].sum()),
                'normalized_pc_explained_variance_ratio_sum': float(shape_params['explained_variance_ratio'].sum()),
                'shuffled_assignment_changed_rows': int((permutation != np.arange(len(raw))).sum()),
                'shuffle_index_sha256': digest_array(permutation),
                'raw_stats_column_std': raw_stats.std(0).tolist(),
            }
    files, nodes = {}, {}
    for arm, arrays in variants.items():
        validate_array_keys(arrays)
        for name, value in base.items():
            if name not in ('x_lncRNA', 'x_miRNA'):
                np.testing.assert_array_equal(arrays[name], value)
        for nt in graph.node_types:
            if arrays['x_' + nt].shape[0] != graph[nt].num_nodes:
                raise ValueError('Node alignment differs')
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Non-finite input')
        for name, value in arrays.items():
            if name.startswith('weight_'):
                np.testing.assert_array_equal(value, np.ones_like(value))
        np.savez_compressed(dest / (arm + '.npz'), **arrays)
        files[arm + '.npz'] = sha256(dest / (arm + '.npz'))
        nodes[arm] = {nt: digest_array(arrays['x_' + nt]) for nt in graph.node_types}
    for k in base:
        np.testing.assert_array_equal(variants['base'][k], base[k])
        np.testing.assert_array_equal(variants['rawpca_stats'][k], pca[k])
    np.savez_compressed(dest / 'normalized_expression_transform.npz', **parameters)
    files['normalized_expression_transform.npz'] = sha256(dest / 'normalized_expression_transform.npz')
    shutil.copy2(ref / 'inputs' / task / 'train_graph.pth', dest / 'train_graph.pth')
    files['train_graph.pth'] = sha256(dest / 'train_graph.pth')
    config = {
        'identity': identity, 'created_at': datetime.now(timezone.utc).isoformat(),
        'input_files_sha256': files, 'node_sha256': nodes, 'audit': audit,
        'references': {arm: reference_record(ref, reference, task, old_arm)
                       for arm, old_arm in (('base', 'base'), ('rawpca_stats', 'expr_pca64_stats'))},
        'versions': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'pandas', 'scikit-learn', 'torch', 'torch-geometric')},
    }
    if identity['source_sha256'] != own_source_hashes():
        raise ValueError('Source changed during preparation')
    snapshot = dest / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in source_hashes():
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    config['sha256'] = json_digest(config)
    write_json(dest / 'config.json', config)
    print('FROZEN', task, len(ARMS), 'validation-only arms', config['sha256'], flush=True)
    return config


def load_config(args, reference, task):
    dest = Path(args.out_dir).resolve() / task
    config = json.loads((dest / 'config.json').read_text())
    if config['sha256'] != json_digest({k: v for k, v in config.items() if k != 'sha256'}):
        raise ValueError('Frozen config changed')
    identity = json.loads(json.dumps(identity_for(reference, task, args.smoke)))
    if config['identity'] != identity:
        raise ValueError('Frozen source/protocol differs; use a new output directory')
    for filename, digest in config['input_files_sha256'].items():
        if sha256(dest / filename) != digest:
            raise ValueError('Frozen input changed: ' + filename)
    return config


def prepare(args):
    ref, out = Path(args.reference_dir).resolve(), Path(args.out_dir).resolve()
    reference = load_protocol(ref)
    for name, digest in reference['raw_network_files_sha256'].items():
        if sha256(Path(reference['net_dir']) / name) != digest:
            raise ValueError('Raw network input changed: ' + name)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'MATH_AUDIT.json', math_audit(ref, reference))
    for task in TASKS:
        prepare_task(args, reference, task)


def validate_array_keys(arrays):
    if any('test' in k or k.startswith(('phi_', 'pair_scaler_')) for k in arrays):
        raise ValueError('Only GNN training/validation inputs are accepted')
    for name in ('pairs_train', 'pairs_val', 'y_train', 'y_val'):
        if name not in arrays:
            raise ValueError('Missing ' + name)


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(digest_array(tensor.detach().cpu().numpy()).encode())
    return h.hexdigest()


def build_model(arrays, data, task, device):
    validate_array_keys(arrays)
    set_seed(666)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    graph = data.clone()
    for nt in graph.node_types:
        graph[nt].x = torch.from_numpy(arrays['x_' + nt])
    graph = graph.to(device)
    weights = {key: torch.from_numpy(arrays['weight_' + str(i)]).to(device)
               for i, key in enumerate(graph.edge_types)}
    indices = {fold: torch.from_numpy(arrays['pairs_' + fold]).to(device) for fold in ('train', 'val')}
    model = ExperimentModel(graph, task).to(device)

    def forward(fold):
        if fold not in ('train', 'val'):
            raise ValueError('Only training/validation forward passes are permitted')
        return model(graph.x_dict, graph.edge_index_dict, indices[fold], weights)

    with torch.no_grad():
        forward('train')  # Same lazy parameter initialization as the frozen runner.
    return model, forward


def train_val_only(arrays, data, task, config, device, arm):
    start = time.monotonic()
    model, forward = build_model(arrays, data, task, device)
    initial_hash = state_digest(model)
    settings = config['identity']['training']
    optimizer = torch.optim.RMSprop(model.parameters(), lr=settings['lr'], alpha=0.99)
    target = torch.as_tensor(arrays['y_train'], device=device)
    best, best_epoch, best_state, best_prediction = -np.inf, None, None, None
    trace = []
    for epoch in range(1, settings['train_epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(forward('train'), target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = forward('val').cpu().numpy()
        auc, ap = score(arrays['y_val'], val)
        criterion = (auc + ap) / 2
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), val_auc=auc,
                          val_aupr=ap, criterion=criterion))
        if criterion > best:
            best, best_epoch = criterion, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_prediction = val.copy()
        if epoch == 1 or epoch % 100 == 0 or epoch == settings['train_epochs']:
            print(task, arm, 'epoch', epoch, 'val %.6f/%.6f' % (auc, ap), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored = forward('val').cpu().numpy()
    np.testing.assert_allclose(restored, best_prediction, atol=2e-5, rtol=2e-5)
    auc, ap = score(arrays['y_val'], best_prediction)
    restored_auc, restored_ap = score(arrays['y_val'], restored)
    result = {
        'val_auc': auc, 'val_aupr': ap, 'val_criterion': (auc + ap) / 2,
        'selected_epoch': best_epoch, 'trained_epochs': settings['train_epochs'],
        'selection': 'maximum_mean_validation_auc_aupr_first_tie',
        'prediction_kind': 'logit', 'prediction_source': 'saved at the selected epoch',
        'initial_state_sha256': initial_hash,
        'params': sum(p.numel() for p in model.parameters()),
        'restore_max_logit_abs_error': float(np.max(np.abs(restored - best_prediction))),
        'restore_val_auc_delta': restored_auc - auc, 'restore_val_aupr_delta': restored_ap - ap,
        'elapsed_seconds': time.monotonic() - start,
    }
    return result, best_prediction, trace, best_state


def check_completed(target, config):
    result = json.loads((target / 'result.json').read_text())
    if result['config_sha256'] != config['sha256']:
        raise ValueError('Result belongs to another configuration')
    for filename, digest in result['artifacts_sha256'].items():
        if sha256(target / filename) != digest:
            raise ValueError('Output artifact changed: ' + filename)
    return result


def run(args):
    reference = load_protocol(Path(args.reference_dir).resolve())
    for task in [args.task] if args.task else TASKS:
        config = load_config(args, reference, task)
        dest = Path(args.out_dir).resolve() / task
        with (dest / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            graph = torch.load(dest / 'train_graph.pth', map_location='cpu')
            initial_by_shape = {}
            for arm in ARMS:
                target = dest / arm
                target.mkdir(exist_ok=True)
                if (target / 'result.json').exists():
                    result = check_completed(target, config)
                    print('already complete', task, arm, flush=True)
                else:
                    with np.load(dest / (arm + '.npz'), allow_pickle=False) as z:
                        arrays = {k: z[k].copy() for k in z.files}
                    print('START', task, arm, 'original GNN, seed 666, val only', flush=True)
                    result, val, trace, state = train_val_only(arrays, graph, task, config, args.device, arm)
                    result.update(task=task, arm=arm, model='gnn', seed=666,
                                  device=args.device, cpu_threads=args.cpu_threads,
                                  config_sha256=config['sha256'],
                                  node_dims={nt: arrays['x_' + nt].shape[1] for nt in graph.node_types})
                    np.savez_compressed(target / 'scores.npz', val=val,
                                        y_val=arrays['y_val'], pairs_val=arrays['pairs_val'])
                    pd.DataFrame(trace).to_csv(target / 'trace.csv', index=False)
                    torch.save(state, target / 'checkpoint.pt')
                    result['artifacts_sha256'] = {name: sha256(target / name)
                                                 for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                    write_json(target / 'result.json', result)
                    print('DONE', task, arm, 'val %.6f/%.6f at epoch %d' % (
                        result['val_auc'], result['val_aupr'], result['selected_epoch']), flush=True)
                shape = tuple(result['node_dims'].items())
                match = (result['initial_state_sha256'], result['params'])
                if shape in initial_by_shape and initial_by_shape[shape] != match:
                    raise AssertionError('Same-width arms did not share initialization')
                initial_by_shape[shape] = match


def summarize(args):
    reference = load_protocol(Path(args.reference_dir).resolve())
    out = Path(args.out_dir).resolve()
    rows, contrasts, verification = [], [], {}
    for task in TASKS:
        config = load_config(args, reference, task)
        dest, results, initial = out / task, {}, {}
        graph = torch.load(dest / 'train_graph.pth', map_location='cpu')
        for arm in ARMS:
            target = dest / arm
            result = check_completed(target, config)
            with np.load(dest / (arm + '.npz')) as z:
                arrays = {k: z[k].copy() for k in z.files}
            validate_array_keys(arrays)
            with np.load(target / 'scores.npz') as z:
                if set(z.files) != {'val', 'y_val', 'pairs_val'}:
                    raise ValueError('Unexpected prediction folds')
                for key in ('y_val', 'pairs_val'):
                    np.testing.assert_array_equal(z[key], arrays[key])
                saved_val = z['val'].copy()
            auc, ap = score(arrays['y_val'], saved_val)
            for metric, value in zip(VAL_METRICS, (auc, ap, (auc + ap)/2)):
                np.testing.assert_allclose(result[metric], value, atol=1e-12, rtol=0)
            trace = pd.read_csv(target / 'trace.csv')
            epochs = config['identity']['training']['train_epochs']
            np.testing.assert_array_equal(trace.epoch, np.arange(1, epochs + 1))
            chosen = trace.loc[trace.criterion.idxmax()]
            if int(chosen.epoch) != result['selected_epoch']:
                raise AssertionError('Wrong single-checkpoint selection')
            np.testing.assert_allclose(chosen.val_auc, auc, atol=1e-12, rtol=0)
            np.testing.assert_allclose(chosen.val_aupr, ap, atol=1e-12, rtol=0)
            model, forward = build_model(arrays, graph, task, args.device)
            if state_digest(model) != result['initial_state_sha256']:
                raise AssertionError('Initial parameter state no longer matches')
            model.load_state_dict(torch.load(target / 'checkpoint.pt', map_location='cpu'))
            model.eval()
            with torch.no_grad():
                restored = forward('val').cpu().numpy()
            np.testing.assert_allclose(restored, saved_val, atol=2e-5, rtol=2e-5)
            shape = tuple(result['node_dims'].items())
            pair = (result['initial_state_sha256'], result['params'])
            if shape in initial and initial[shape] != pair:
                raise AssertionError('Mismatched same-width initial states')
            initial[shape] = pair
            results[arm] = result
        base = results['base']
        verification[task] = {
            'completed_runs': len(results), 'initialization_groups': len(initial),
            'reference_reproduction': {
                arm: {metric: results[arm][metric] - old[metric] for metric in VAL_METRICS + ('selected_epoch',)}
                for arm, old in config['references'].items()},
        }
        for arm, result in results.items():
            row = {k: result[k] for k in ('task', 'arm', 'seed', 'selected_epoch', 'params') + VAL_METRICS}
            row.update({metric + '_delta': result[metric] - base[metric] for metric in VAL_METRICS})
            rows.append(row)
        for name, (control, candidate) in CONTRASTS.items():
            contrasts.append(dict(task=task, contrast=name, control=control, candidate=candidate,
                                  **{metric + '_delta': results[candidate][metric] - results[control][metric]
                                     for metric in VAL_METRICS}))
    frame = pd.DataFrame(rows)
    contrast_frame = pd.DataFrame(contrasts)
    frame.to_csv(out / 'results.csv', index=False)
    contrast_frame.to_csv(out / 'contrasts.csv', index=False)
    passing = {arm: bool(((frame.loc[frame.arm == arm, ['val_auc_delta', 'val_aupr_delta']]) > 0).all().all())
               for arm in ARMS if arm != 'base'}
    contrast_passing = {name: bool(((contrast_frame.loc[contrast_frame.contrast == name, ['val_auc_delta', 'val_aupr_delta']]) > 0).all().all())
                        for name in CONTRASTS}
    verification.update(status='passed', smoke=args.smoke, seed=666, completed_runs=len(rows),
                        evaluation_folds=['val'], four_metrics_above_base=passing,
                        four_positive_metrics_for_contrast=contrast_passing,
                        checks=['frozen sources and input hashes', 'training-only preprocessing fit IDs',
                                'unchanged labels/graph/non-RNA inputs/unit weights',
                                'same-width initialization and parameter-count matches',
                                'validation-only arrays and score files', 'saved-score metric recomputation',
                                'full fixed epoch budgets', 'one first-max validation checkpoint',
                                'checkpoint prediction replay'])
    write_json(out / 'VALIDATION.json', verification)
    lines = ['# 表达尺度机制对照：原 GNN，seed 666，只看 val', '',
             '固定首轮清理后的训练图与划分、原 GNN、训练预算和优化器。每项取验证 AUC/AUPR 均值首次达到最大时的同一个检查点。'
             '两种 RNA 同时替换输入，其他节点、边及训练标签完全相同。所有新缓存和预测仅含 train/val；不计算 test。', '',
             '同输入宽度的对照具有相同参数数量和逐参数相同的初始权重；统计量与 PCA 的缩放器只拟合训练节点。'
             '打乱对照对三项统计量联合换行，保持联合分布，分别在训练拟合节点和其余节点内完成，固定预处理种子 0。', '',
             '| 数据集 | 输入 | val AUC | val AUPR | Δ AUC 对 base | Δ AUPR 对 base | 轮次 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| %s | %s | %.6f | %.6f | %+.6f | %+.6f | %d |' % (
            r['task'], r['arm'], r['val_auc'], r['val_aupr'], r['val_auc_delta'], r['val_aupr_delta'], r['selected_epoch']))
    lines += ['', '## 预设对照的差值（candidate − control）', '',
              '| 对照 | LncDrug Δ AUC / AUPR | MiDrug Δ AUC / AUPR | 四项全正 |',
              '|---|---:|---:|---|']
    for name in CONTRASTS:
        selected = {r['task']: r for r in contrasts if r['contrast'] == name}
        lnc, mi = selected['LncDrug'], selected['MiDrug']
        lines.append('| %s | %+.6f / %+.6f | %+.6f / %+.6f | %s |' % (
            name, lnc['val_auc_delta'], lnc['val_aupr_delta'], mi['val_auc_delta'], mi['val_aupr_delta'],
            '是' if contrast_passing[name] else '否'))
    lines += ['', '## 输入定义', '']
    lines += ['- `%s`：%s。' % (arm, desc) for arm, desc in ARMS.items()]
    lines += ['', '## 解释边界', '',
              '数学上的不可唯一逆变换与 GNN 的预测增益是两个问题。原始均值/标准差加完整归一化向量可恢复尺度，'
              '并不保证统计量能提高下游指标。原首轮 PCA 使用归一化前表达，本身可保留尺度；不能把 PCA＋统计量相对基线的提升全部归于统计量。', '',
              '零填充控制输入维度/初始化，归一化后统计量控制无需新增信息的摘要作用，打乱统计量控制特征分布与节点对应关系。'
              '单次打乱和单个 seed 只提供当前固定划分上的消融证据，不构成统计显著性或生物学因果证明。', '',
              '数值审计：[MATH_AUDIT.json](MATH_AUDIT.json)；完整结果：[results.csv](results.csv)；'
              '对照：[contrasts.csv](contrasts.csv)；核验：[VALIDATION.json](VALIDATION.json)。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(frame[['task', 'arm', 'val_auc', 'val_aupr', 'val_auc_delta', 'val_aupr_delta']].to_string(index=False), flush=True)
    print('VERIFIED', len(rows), 'validation-only runs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'summarize'))
    parser.add_argument('--reference-dir', default='improve/results/first_round_features_20260909')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true', help='Separate output, two training epochs per arm')
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    {'prepare': prepare, 'run': run, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
