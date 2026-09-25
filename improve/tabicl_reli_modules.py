"""Shared local-evidence matching and a final-layer relational ICL adapter.

The first probe freezes every neural weight. One shared, validation-selected
attention strength is applied to both RNA tasks. No dataset-specific bypass,
query label, direct RNA-drug binding assumption, or installed-source edit.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import types

import numpy as np
import torch


STRENGTHS = (0.03125, 0.125, 0.5, 2.0)


def recipes():
    return [dict(id='zero', mode='zero', strength=0.)] + [
        dict(id=f'{mode}_{str(value).replace(".", "p")}', mode=mode, strength=value)
        for mode in ('pooled', 'local') for value in STRENGTHS]


def normalize_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def projected_evidence(local, weights, projection, center):
    local, weights = np.asarray(local, np.float32), np.asarray(weights, np.float32)
    if local.ndim != 3 or weights.shape != local.shape[:2] or np.any(weights < 0):
        raise ValueError('Invalid local evidence')
    if not np.isfinite(local).all() or not np.isfinite(weights).all():
        raise ValueError('Nonfinite local evidence')
    totals = weights.sum(-1)
    if not np.all(np.isclose(totals, 1., atol=1e-5) | (totals == 0)):
        raise ValueError('Evidence weights must sum to one or zero for missing entities')
    projected = normalize_rows((local-center) @ projection)
    projected[weights == 0] = 0
    return projected.astype(np.float32), normalize_rows((projected*weights[..., None]).sum(1)).astype(np.float32)


@torch.no_grad()
def soft_local_kernel(local, weights, device='cpu', temperature=.15, null_score=.2, chunk=8):
    """Symmetric many-to-many matching with normalized target and null mass."""
    x = torch.as_tensor(local, dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    if x.ndim != 3 or w.shape != x.shape[:2] or temperature <= 0 or chunk < 1:
        raise ValueError('Invalid matching shapes or settings')
    n, sites, dim = x.shape
    keys = x.reshape(n*sites, dim).T
    result = np.empty((n, n), np.float32)
    null_mass = math.exp(null_score/temperature)
    for begin in range(0, n, chunk):
        query = x[begin:begin+chunk]
        scores = (query.reshape(-1, dim) @ keys).reshape(len(query), sites, n, sites).permute(0, 2, 1, 3)
        mass = torch.exp(scores/temperature) * w[None, :, None, :]
        matches = (mass*scores).sum(-1)/(mass.sum(-1)+null_mass)
        result[begin:begin+len(query)] = (matches*w[begin:begin+len(query), None, :]).sum(-1).cpu().numpy()
    return ((result+result.T)*.5).astype(np.float32)


def build_kernels(input_dir, device):
    """Only entity attributes; no split, graph, association label or prediction IO."""
    from improve.tabicl_reli_inputs import digest, write_json
    input_dir = Path(input_dir)
    target = input_dir/'kernels.npz'
    if target.exists():
        raise ValueError('Evidence kernels already exist')
    assets = {}
    source_sha = {}
    for task in ('LncDrug', 'MiDrug'):
        path = input_dir/('rna_'+task+'.npz')
        with np.load(path, allow_pickle=False) as archive:
            assets[task] = {key: archive[key].copy() for key in ('local', 'weights', 'names')}
        source_sha[path.name] = digest(path)
    globals_ = []
    for asset in assets.values():
        keep = asset['weights'].sum(-1) > 0
        globals_.append((asset['local']*asset['weights'][..., None]).sum(1)[keep])
    # Entity-balanced unsupervised centering, identical for both RNA types.
    center = np.concatenate(globals_).mean(0)
    rng = np.random.default_rng(20260912)
    rna_projection = (rng.standard_normal((640, 128))/math.sqrt(128)).astype(np.float32)
    drug_projection = (rng.standard_normal((1024, 128))/math.sqrt(128)).astype(np.float32)
    output = {}
    for task, asset in assets.items():
        local, pooled = projected_evidence(asset['local'], asset['weights'], rna_projection, center)
        output[task+'_local'] = soft_local_kernel(local, asset['weights'], device)
        output[task+'_pooled'] = (pooled@pooled.T).astype(np.float32)
        output[task+'_present'] = asset['weights'].sum(-1) > 0
        output[task+'_names'] = asset['names']
        print('KERNEL', task, output[task+'_local'].shape, flush=True)
    path = input_dir/'drug.npz'
    with np.load(path, allow_pickle=False) as archive:
        local, weights, names = archive['local'], archive['weights'], archive['names']
    projected, pooled = projected_evidence(local, weights, drug_projection, np.zeros(1024, np.float32))
    output['drug_local'] = soft_local_kernel(projected, weights, device)
    output['drug_pooled'] = (pooled@pooled.T).astype(np.float32)
    output['drug_present'] = weights.sum(-1) > 0
    output['drug_names'] = names
    source_sha[path.name] = digest(path)
    output.update(rna_center=center, rna_projection=rna_projection, drug_projection=drug_projection)
    np.savez_compressed(target, **output)
    write_json(input_dir/'kernels.json', dict(
        files=source_sha, output_sha256=digest(target), implementation_sha256=digest(__file__),
        local_temperature=.15, local_null_score=.2, projection_seed=20260912, projection_dimensions=128,
        centering='RNA: all resolved named entities, without labels, equal mass per entity. Drug: no centering.',
        pooled_control='Same projected, unit-normalized local evidence and coverage weights, pooled before comparison.',
        transductive_attributes=True, no_association_labels=True))
    print('KERNELS_SAVED', target, flush=True)


def _context_zscore(values, present):
    present = np.asarray(present, bool)
    counts = present.sum(-1, keepdims=True)
    mean = (values*present).sum(-1, keepdims=True)/np.maximum(counts, 1)
    var = (((values-mean)*present)**2).sum(-1, keepdims=True)/np.maximum(counts, 1)
    result = (values-mean)/np.maximum(np.sqrt(var), 1e-6)
    result[~present] = 0.
    result = np.clip(result, -3., 3.)
    return result.astype(np.float32)


def pair_bias(kernels, task, query_pairs, support_pairs, mode):
    if mode not in ('local', 'pooled'):
        raise ValueError('Unknown evidence comparison')
    q, s = np.asarray(query_pairs), np.asarray(support_pairs)
    if q.ndim != 2 or s.ndim != 2 or q.shape[0] != 2 or s.shape[0] != 2:
        raise ValueError('Expected RNA/drug pair columns')
    if q.dtype.kind not in 'iu' or s.dtype.kind not in 'iu' or np.any(q < 0) or np.any(s < 0):
        raise ValueError('Pair indices must be nonnegative integers')
    scores = []
    for row, prefix in enumerate((task, 'drug')):
        kernel, available = kernels[prefix+'_'+mode], kernels[prefix+'_present']
        if q[row].max(initial=0) >= len(kernel) or s[row].max(initial=0) >= len(kernel):
            raise ValueError('Pair index exceeds entity evidence')
        values = kernel[q[row, :, None], s[row, None, :]]
        present = available[q[row, :, None]] & available[s[row, None, :]]
        scores.append(_context_zscore(values, present))
    # Both endpoint modalities contribute, with one rule for both RNA tasks.
    bias = np.clip((scores[0]+scores[1])/math.sqrt(2), -3., 3.).astype(np.float32)
    if not np.isfinite(bias).all():
        raise FloatingPointError('Invalid relation bias')
    return bias


def decode(predictor, hidden, return_logits, temperature):
    if predictor.norm_first:
        hidden = predictor.ln(hidden)
    out = predictor.decoder(hidden)[..., :2]
    return out if return_logits else torch.softmax(out/temperature, dim=-1)


@torch.no_grad()
def replay_last_layer(predictor, rows, labels, biases, configs, *, return_logits=True, temperature=.9):
    """Reuse unmodified prefix layers and cached final-layer support keys/values.

    Labels are known support labels only. No query can update a support state.
    Changes are made inside the actual pretrained ICL attention, not post-hoc
    probability blending. The zero branch is computed on the native full shape.
    """
    from tabicl._model.kv_cache import KVCacheEntry
    if predictor.training or predictor.max_classes < 2:
        raise ValueError('This binary probe requires evaluation mode')
    train_size = labels.shape[1]
    query_size = rows.shape[1]-train_size
    if query_size <= 0 or any(b.shape != (query_size, train_size) for b in biases.values()):
        raise ValueError('Evidence/context/query alignment differs')
    hidden = rows.clone()
    hidden[:, :train_size] += predictor.y_encoder(labels.float())
    encoder = predictor.tf_icl
    for block in encoder.blocks[:-1]:
        hidden = block(q=hidden, train_size=train_size, rope=encoder.rope)
    block = encoder.blocks[-1]
    base_hidden, key, value = block(q=hidden, train_size=train_size, rope=encoder.rope, need_kv=True)
    outputs = {'zero': decode(predictor, base_hidden[:, train_size:], return_logits, temperature)}
    cache = KVCacheEntry(key=key, value=value)
    for cfg in configs:
        if cfg['mode'] == 'zero':
            continue
        mask = biases[cfg['mode']]*cfg['strength']
        changed = block(q=hidden[:, train_size:], cached_kv=cache, rope=encoder.rope, attn_mask=mask)
        outputs[cfg['id']] = decode(predictor, changed, return_logits, temperature)
    return outputs


def aggregate_members(clf, outputs):
    shuffles = [p for group in clf.ensemble_generator_.class_shuffles_.values() for p in group]
    if len(shuffles) != len(outputs):
        raise ValueError('Captured members do not match class permutations')
    average = np.zeros_like(outputs[0])
    for member, permutation in zip(outputs, shuffles):
        average += member[..., permutation]
    average /= len(shuffles)
    if clf.average_logits:
        average = clf.softmax(average, axis=-1, temperature=clf.softmax_temperature)
    return average/average.sum(axis=1, keepdims=True)


def predict_all(clf, query_features, biases, configs=None, check_native=True):
    """Public experiment API; always restores the original estimator method."""
    configs = recipes() if configs is None else configs
    if any(c not in recipes() for c in configs) or not any(c['id'] == 'zero' for c in configs):
        raise ValueError('Unregistered or missing zero reference recipe')
    if getattr(clf, 'model_kv_cache_', None) is not None:
        raise ValueError('Expected unmodified no-cache sklearn fit')
    predictor = clf.model_.icl_predictor
    native_forward = predictor.forward
    device = torch.device(clf.device_)
    side = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in biases.items()}
    captured = {cfg['id']: [] for cfg in configs}
    checks = []

    def forward(this, R, y_train, return_logits=True, softmax_temperature=.9, mgr_config=None):
        this.to(device)
        R, y_train = R.to(device), y_train.to(device)
        native = native_forward(R=R.clone(), y_train=y_train, return_logits=return_logits,
                                softmax_temperature=softmax_temperature, mgr_config=mgr_config)
        variants = replay_last_layer(this, R, y_train, side, configs,
                                     return_logits=return_logits, temperature=softmax_temperature)
        error = float((variants['zero']-native.to(device)).abs().max().cpu())
        if check_native:
            torch.testing.assert_close(variants['zero'], native.to(device), atol=1e-5, rtol=1e-5)
        checks.append(error)
        for key in captured:
            # Use the actual official result as zero, after checking native replay.
            value = native if key == 'zero' else variants[key]
            captured[key].append(value.float().cpu().numpy())
        return native

    predictor.forward = types.MethodType(forward, predictor)
    try:
        baseline = np.asarray(clf.predict_proba(query_features))
    finally:
        predictor.forward = native_forward
    probabilities = {key: aggregate_members(clf, np.concatenate(parts)) for key, parts in captured.items()}
    np.testing.assert_allclose(probabilities['zero'], baseline, atol=1e-7, rtol=0)
    index = int(np.flatnonzero(np.asarray(clf.classes_) == 1)[0])
    scores = {key: value[:, index].astype(np.float64) for key, value in probabilities.items()}
    if any(not np.isfinite(value).all() or np.any((value < 0) | (value > 1)) for value in scores.values()):
        raise FloatingPointError('Invalid adapter predictions')
    return scores, dict(native_icl_replay_max_abs=max(checks), forward_calls=len(checks),
                        actual_members=sum(len(part) for part in captured['zero']),
                        aggregation_max_abs=float(np.max(np.abs(probabilities['zero']-baseline))))
