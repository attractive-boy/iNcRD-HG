"""Inputs for the fixed-split network/feature ablation.

All files are written below a new output directory. The released CSVs and
serialized graph splits are never changed. Pair features are cross-fitted;
expression transforms are fitted on nodes available in the training split.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from improve.feature_extra import extra_columns
from improve.gbdt import build_inputs, featurise
from improve.run import TASKS


FEATURE_VARIANTS = ('base', 'drug_unmasked', 'drug_masked',
                    'expr_pca64_stats', 'expr_ae64', 'expr_stats3')


def digest_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def minmax_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min(1, keepdims=True)) / np.maximum(
        x.max(1, keepdims=True) - x.min(1, keepdims=True), 1e-8)


def masked_pool(hidden, mask):
    """Mean over non-padding positions, retaining BOS/EOS in both controls."""
    w = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * w).sum(1) / w.sum(1).clamp_min(1)


def extract_drugs(net_dir, model_dir, out_dir, device='cuda:0', batch_size=8):
    from transformers import AutoModel, AutoTokenizer

    net_dir, out_dir = Path(net_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    smiles = pd.read_csv(net_dir / 'drug_smiles.csv')
    drug_ids = pd.read_csv(net_dir / 'Drug.csv')['CID'].astype(str).tolist()
    smiles['drug_cid'] = smiles['drug_cid'].astype(str)
    smiles = smiles.set_index('drug_cid').loc[drug_ids]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device).eval()
    strings = smiles['smiles'].tolist()
    full_lengths = [len(tokenizer.encode(s, truncation=False)) for s in strings]
    # Reproduce padding of the original all-drug call even when microbatching.
    padded_length = min(512, max(full_lengths))
    unmasked, masked, valid_lengths = [], [], []
    with torch.inference_mode():
        for start in range(0, len(strings), batch_size):
            tokens = tokenizer(strings[start:start + batch_size],
                               return_tensors='pt', padding='max_length',
                               truncation=True, max_length=padded_length)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            unmasked.append(hidden.mean(1).cpu().numpy())
            masked.append(masked_pool(hidden, tokens['attention_mask']).cpu().numpy())
            valid_lengths.extend(tokens['attention_mask'].sum(1).cpu().tolist())
    unmasked, masked = np.concatenate(unmasked), np.concatenate(masked)
    assert np.isfinite(unmasked).all() and np.isfinite(masked).all()
    np.savez_compressed(out_dir / 'drug_pooling.npz',
                        unmasked=unmasked, masked=masked,
                        ids=np.asarray(drug_ids), lengths=np.asarray(valid_lengths))
    old = pd.read_csv(net_dir / 'drug_features-chemBertA.csv', index_col=0)
    audit = dict(model_path=str(Path(model_dir).resolve()),
                 model_config=model.config.to_dict(), n_drugs=len(strings),
                 dimensions=int(masked.shape[1]), original_dimensions=old.shape[1],
                 padded_length=padded_length,
                 truncated_drugs=sum(n > 512 for n in full_lengths),
                 padding_fraction_mean=float(1 - np.mean(valid_lengths) / padded_length),
                 pooling_relative_change=float(np.linalg.norm(masked - unmasked)
                                               / np.linalg.norm(unmasked)),
                 note='Compare masked against unmasked to isolate pooling. '
                      'Both use the same local checkpoint and full-batch padding length. '
                      'Differences against base also include checkpoint replacement.')
    if old.shape == unmasked.shape:
        audit['unmasked_vs_original_rmse'] = float(np.sqrt(
            np.mean((old.values - unmasked) ** 2)))
    # A real-model check: masked pooling is invariant to extra padding.
    sample = strings[0]
    short_len = min(512, len(tokenizer.encode(sample)))
    lengths = [short_len, min(512, short_len + 17)]
    pooled = []
    with torch.inference_mode():
        for length in lengths:
            t = tokenizer([sample], return_tensors='pt', padding='max_length',
                          truncation=True, max_length=length)
            t = {k: v.to(device) for k, v in t.items()}
            pooled.append(masked_pool(model(**t).last_hidden_state,
                                      t['attention_mask']).cpu().numpy())
    audit['padding_invariance_max_abs'] = float(np.max(np.abs(pooled[0] - pooled[1])))
    if not np.allclose(pooled[0], pooled[1], atol=2e-5, rtol=2e-5):
        raise AssertionError('Masked pooling failed the padding invariance check')
    write_json(out_dir / 'drug_pooling_audit.json', audit)
    return audit


def split_fingerprints(splits, key):
    return {
        fold: {
            'labels': digest_array(data[key].edge_label.numpy()),
            'pairs': digest_array(data[key].edge_label_index.numpy()),
            'graph': {'|'.join(k): digest_array(v.numpy())
                      for k, v in data.edge_index_dict.items()},
        } for fold, data in zip(('train', 'val', 'test'), splits)
    }


def crossfit_matrices(pairs, y, A, n_folds=5):
    fold = np.random.RandomState(0).randint(0, n_folds, pairs.shape[1])
    for k in range(n_folds):
        take = fold == k
        Ak = A.copy()
        held = pairs[:, take & (y == 1)]
        Ak[held[0], held[1]] = 0
        assert not Ak[held[0], held[1]].any()
        yield take, Ak


def expression_features(train, net_dir, key, components=64, kind='pca'):
    """Train-node column standardization, compression and three RNA summaries.

    `kind='pca'` uses a 64-component PCA (the original ablation arm);
    `kind='ae'` replaces it with a 938->256->64 autoencoder bottleneck trained
    on the training-fold node rows (MSE reconstruction), so the non-linear
    compression axis is measured against the linear one.  Both are fit on the
    training nodes only (no leakage into val/test) and both output the
    bottleneck + the three expression statistics, standardized.
    """
    if kind not in ('pca', 'ae'):
        raise ValueError(kind)
    replacements, summary, audit = {}, {}, {}
    for nt, filename, idfile in [('lncRNA', 'LncExp.csv', 'lncRNA.csv'),
                                ('miRNA', 'MiExp.csv', 'miRNA.csv')]:
        frame = pd.read_csv(Path(net_dir) / filename, index_col=0).T
        names = pd.read_csv(Path(net_dir) / idfile)['x'].str.upper().tolist()
        frame.index = frame.index.str.upper()
        frame = frame.loc[names]
        if nt == 'miRNA':
            frame = frame[sorted(frame.columns, key=lambda c: int(c.split('-')[-1]))]
        raw = frame.values.astype(np.float32)
        if nt == key[0]:
            ids = np.unique(train[key].edge_label_index[0].numpy())
        else:
            relation = 'LncDrug' if nt == 'lncRNA' else 'MiDrug'
            ids = np.unique(train[nt, relation, 'drug'].edge_index[0].numpy())
        assert len(ids) > components and np.isfinite(raw).all()
        scaler = StandardScaler().fit(raw[ids])
        scaled = scaler.transform(raw)
        if kind == 'ae':
            enc = _ae_encoder(scaled[ids], dims=(256, components), epochs=300)
            z = enc(scaled)
            audit[nt] = dict(method='ae',
                             fit_nodes=len(ids), fit_ids_sha256=digest_array(ids),
                             input_dim=raw.shape[1], output_dim=components + 3,
                             bottleneck_dims=(256, components), epochs=300,
                             statistics=['expression_mean', 'expression_std',
                                         'zero_fraction'])
        else:
            pca = PCA(n_components=components, svd_solver='randomized',
                      random_state=0)
            pca.fit(scaled[ids])
            z = pca.transform(scaled)
            audit[nt] = dict(method='pca',
                             fit_nodes=len(ids), fit_ids_sha256=digest_array(ids),
                             input_dim=raw.shape[1], output_dim=components + 3,
                             explained_variance=float(
                                 pca.explained_variance_ratio_.sum()),
                             statistics=['expression_mean', 'expression_std',
                                         'zero_fraction'])
        pc = StandardScaler().fit(z[ids]).transform(z)
        stats = np.stack([raw.mean(1), raw.std(1), (raw == 0).mean(1)], axis=1)
        stats = StandardScaler().fit(stats[ids]).transform(stats).astype(np.float32)
        replacements[nt] = np.concatenate([pc, stats], axis=1).astype(np.float32)
        summary[nt] = stats
    return replacements, summary, audit


def _ae_encoder(X, dims=(256, 64), epochs=300, seed=0):
    """A small deterministic autoencoder; returns an `encode(Y)` callable.

    Fits on `X` (training-fold rows only, already standardized) with full-batch
    MSE.  The bottleneck is `Linear->ReLU->Linear`, exposed by slicing the
    Sequential.  CPU-only and seed-fixed so the node features are identical
    across the experiment's five seeds.
    """
    import torch
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], dims[0]), torch.nn.ReLU(),
        torch.nn.Linear(dims[0], dims[1]),
        torch.nn.Linear(dims[1], dims[0]), torch.nn.ReLU(),
        torch.nn.Linear(dims[0], X.shape[1]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(X)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(xt) - xt) ** 2).mean()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def encode(Y):
        model.eval()
        return model[:3](torch.from_numpy(Y)).numpy().astype(np.float32)

    return encode


def attribute_columns(pairs, A, Xr, Xd):
    """The four existing attribute pair features, with vectorized maxima."""
    def unit(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

    r, d = pairs
    ur, ud = unit(Xr), unit(Xd)
    dr, dd = A.sum(1), A.sum(0)
    pr = unit((A.T @ ur) / (dd[:, None] + 1e-9))
    pd_ = unit((A @ ud) / (dr[:, None] + 1e-9))
    cr, cd = ur @ ur.T, ud @ ud.T
    mr, md = np.zeros_like(A), np.zeros_like(A)
    for j in range(A.shape[1]):
        hits = A[:, j] > 0
        if hits.any():
            mr[:, j] = cr[:, hits].max(1)
    for i in range(A.shape[0]):
        hits = A[i] > 0
        if hits.any():
            md[i] = cd[:, hits].max(1)
    return np.stack([(pd_[r] * ud[d]).sum(1), (pr[d] * ur[r]).sum(1),
                     md[r, d], mr[r, d]], axis=1)


def prepare_task(run_dir, net_dir, out_dir, task):
    out_dir = Path(out_dir)
    cache = out_dir / 'inputs'
    cache.mkdir(parents=True, exist_ok=True)
    splits, key, A, Sr, Sd, bridge, attrs = build_inputs(run_dir, net_dir, task, 'cpu')
    fingerprints = split_fingerprints(splits, key)
    train = splits[0]
    original = {nt: x.numpy().copy() for nt, x in train.x_dict.items()}
    replacements, stats, expr_audit = expression_features(train, net_dir, key,
                                                          kind='pca')
    ae_replacements, ae_stats, ae_audit = expression_features(train, net_dir,
                                                              key, kind='ae')
    variants = {'base': original,
                'expr_pca64_stats': dict(original, **replacements),
                'expr_ae64': dict(original, **ae_replacements),
                # Isolation control: the 3 expression statistics alone, without
                # any compression, so the compression-vs-summary attribution of
                # expr_pca64_stats / expr_ae64 can be separated.
                'expr_stats3': dict(original, lncRNA=stats['lncRNA'],
                                    miRNA=stats['miRNA'])}
    pools_path = out_dir / 'drug_pooling.npz'
    if pools_path.exists():
        with np.load(pools_path) as pools:
            for name in ('unmasked', 'masked'):
                variants['drug_' + name] = dict(original, drug=minmax_rows(pools[name]))
    rows = {f: (d[key].edge_label_index.numpy(), d[key].edge_label.numpy())
            for f, d in zip(('train', 'val', 'test'), splits)}
    pairs, y = rows['train']
    # Structural features are identical across all feature arms.
    structures, matrices = {}, {}
    for fold in ('val', 'test'):
        idx = rows[fold][0]
        f = featurise(idx, A, Sr, Sd, bridge)
        e = extra_columns(idx, A, Sr, Sd)
        structures[fold] = (f.values, e.values)
    fold_data = []
    for take, Ak in crossfit_matrices(pairs, y, A):
        idx = pairs[:, take]
        f = featurise(idx, Ak, Sr, Sd, bridge)
        e = extra_columns(idx, Ak, Sr, Sd)
        fold_data.append((take, Ak, f.values, e.values))
    for variant, x in variants.items():
        X = {}
        xr, xd = x[key[0]], x['drug']
        for fold in ('val', 'test'):
            idx = rows[fold][0]
            a = attribute_columns(idx, A, xr, xd)
            f, e = structures[fold]
            X[fold] = np.concatenate([f, a, e], axis=1)
        X['train'] = np.empty((len(y), X['val'].shape[1]), dtype=np.float32)
        for take, Ak, f, e in fold_data:
            a = attribute_columns(pairs[:, take], Ak, xr, xd)
            X['train'][take] = np.concatenate([f, a, e], axis=1)
        if variant == 'expr_pca64_stats':
            for fold in X:
                X[fold] = np.concatenate([X[fold], stats[key[0]][rows[fold][0][0]]], axis=1)
        if variant == 'expr_ae64':
            for fold in X:
                X[fold] = np.concatenate([X[fold], ae_stats[key[0]][rows[fold][0][0]]],
                                         axis=1)
        if variant == 'expr_stats3':
            for fold in X:
                X[fold] = np.concatenate([X[fold], stats[key[0]][rows[fold][0][0]]],
                                         axis=1)
        scaler = StandardScaler().fit(X['train'])
        payload = {'x_' + nt: val for nt, val in x.items()}
        payload.update({'phi_' + f: scaler.transform(X[f]).astype(np.float32) for f in X})
        payload.update({'y_' + f: rows[f][1] for f in rows})
        payload.update({'pairs_' + f: rows[f][0] for f in rows})
        for a in payload.values():
            assert np.isfinite(a).all()
        np.savez_compressed(cache / (task + '_' + variant + '.npz'), **payload)
        print('prepared', task, variant, 'pair_dim', X['train'].shape[1], flush=True)
    # Confirm the optimized attribute computation has the released definitions.
    subset = rows['val'][0][:, :40]
    reference = featurise(subset, A, Sr, Sd, bridge, Xr=attrs[0], Xd=attrs[1])
    check = attribute_columns(subset, A, attrs[0], attrs[1])
    np.testing.assert_allclose(reference.iloc[:, -4:].values, check, rtol=2e-5, atol=2e-6)
    assert fingerprints == split_fingerprints(splits, key)
    audit = dict(task=task, split_fingerprints=fingerprints,
                 expression=expr_audit, expression_ae=ae_audit,
                 node_types=list(train.node_types), edge_types=[list(k) for k in train.edge_types],
                 variants=list(variants),
                 protocol='Frozen released split, including existing circRNA relations. '
                          'Validation message graph is train; released test message graph '
                          'includes validation positives. Pair features use training A for both.')
    write_json(cache / (task + '_audit.json'), audit)
    return audit


def similarity_matrices(net_dir):
    files = {('lncRNA', 'LncLnc', 'lncRNA'): 'LncSimMat_ken.csv',
             ('miRNA', 'MiMi', 'miRNA'): 'MiSimMat_ken.csv',
             ('drug', 'DrugDrug', 'drug'): 'drugSimMat.csv'}
    return {k: np.nan_to_num(np.abs(pd.read_csv(Path(net_dir) / f, header=None).values),
                             nan=0.0).astype(np.float32) for k, f in files.items()}


def graph_weights(data, matrices, weighted=False):
    """Preserve the edge list and total incoming weight (including self loops)."""
    result = {}
    for key, edges in data.edge_index_dict.items():
        w = torch.ones(edges.size(1), device=edges.device)
        if weighted and key in matrices:
            r, c = edges.cpu().numpy()
            values = matrices[key][r, c].copy()
            values[r == c] = 1.0
            assert (values > 0).all()
            w = torch.as_tensor(values, device=edges.device)
            total = torch.zeros(data[key[2]].num_nodes, device=edges.device)
            count = torch.zeros_like(total)
            total.index_add_(0, edges[1], w)
            count.index_add_(0, edges[1], torch.ones_like(w))
            w = w * count[edges[1]] / total[edges[1]].clamp_min(1e-8)
        result[key] = w
    return result
