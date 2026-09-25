"""Training-only input snapshots and drug-disjoint LinkPFN episodes."""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import torch

from improve.link_pfn_modules import ATTR_DIM, LinkFeatures, observed_matrix


TASKS = ('LncDrug', 'MiDrug')
RNA_TYPES = ('lncRNA', 'miRNA')


def project_attributes(raw, fit_nodes, seed):
    fit_nodes = np.unique(fit_nodes)
    if len(fit_nodes) <= ATTR_DIM:
        raise ValueError('Too few fitting endpoints for the frozen PCA width')
    raw = np.asarray(raw, dtype=np.float64)
    mean, scale = raw[fit_nodes].mean(0), raw[fit_nodes].std(0)
    scale = np.maximum(scale, 1e-6)
    z = np.clip((raw-mean)/scale, -10, 10)
    pca = PCA(ATTR_DIM, svd_solver='randomized', random_state=seed)
    pca.fit(z[fit_nodes])
    values = pca.transform(z)
    pm, ps = values[fit_nodes].mean(0), np.maximum(values[fit_nodes].std(0), 1e-6)
    return np.clip((values-pm)/ps, -8, 8).astype(np.float32), dict(
        fit_nodes=fit_nodes, input_mean=mean, input_scale=scale,
        components=pca.components_, pca_mean=pca.mean_, output_mean=pm, output_scale=ps)


def drug_groups(pairs, labels, nd, seed):
    """Five fixed groups, stratified by combined training-positive support."""
    count = sum(np.bincount(p[1, y == 1], minlength=nd) for p, y in zip(pairs, labels))
    bins = np.digitize(count, [1, 3, 10, 50, 200])
    rng = np.random.RandomState(seed)
    groups = np.empty(nd, dtype=np.int64)
    for value in np.unique(bins):
        ids = np.flatnonzero(bins == value)
        rng.shuffle(ids)
        groups[ids] = (np.arange(len(ids))+rng.randint(5)) % 5
    return groups, count


def reserve_support(pairs, labels, indices, seed, maximum=3):
    """Reserve at most half of each held drug's positives, leaving queries.

    The SAME reserved rows are excluded from evaluation at every support
    budget. Their labels become visible only when a budget reveals them.
    """
    rng = np.random.RandomState(seed)
    reserved = []
    for d in np.unique(pairs[1, indices]):
        pos = indices[(pairs[1, indices] == d) & (labels[indices] == 1)]
        rng.shuffle(pos)
        reserved.extend(pos[:min(maximum, len(pos)//2)].tolist())
    return np.asarray(reserved, dtype=np.int64)


def snapshot_inputs(reference, seed):
    reference = Path(reference)
    data, provenance, raw = {}, {}, []
    for i, task in enumerate(TASKS):
        path = reference/'inputs'/task/'outer.npz'
        # Never load historical validation or test arrays.
        with np.load(path) as archive:
            if 'pairs_test' in archive.files or 'y_test' in archive.files:
                raise ValueError('Unexpected test archive')
            allowed = ('pairs_train', 'y_train', 'x_lncRNA', 'x_miRNA', 'x_drug')
            current = {k:archive[k].copy() for k in allowed}
        raw.append(current)
        data['pairs_'+str(i)], data['labels_'+str(i)] = current['pairs_train'], current['y_train']
        provenance[task] = dict(archive=str(path), arrays_read=list(allowed))
    for key in ('x_lncRNA', 'x_miRNA', 'x_drug'):
        np.testing.assert_array_equal(raw[0][key], raw[1][key])
    nd = len(raw[0]['x_drug'])
    pairs, labels = [data['pairs_'+str(i)] for i in range(2)], [data['labels_'+str(i)] for i in range(2)]
    groups, count = drug_groups(pairs, labels, nd, seed)
    data.update(drug_group=groups, combined_training_positive_count=count)
    fit_drugs = np.flatnonzero(groups < 3)
    for i in range(2):
        p, y = pairs[i], labels[i]
        if p.shape != (2, len(y)) or len(set(map(tuple,p.T))) != len(y):
            raise ValueError('Invalid or repeated training pair')
        for phase, mask in (('fit',groups < 3), ('dev',groups == 3), ('assess',groups == 4)):
            ids = np.flatnonzero(mask[p[1]])
            if len(np.unique(y[ids])) != 2:
                raise ValueError('Drug partition lacks a class: '+phase)
            data[phase+'_idx_'+str(i)] = ids
            if phase != 'fit':
                data[phase+'_reserved_'+str(i)] = reserve_support(p,y,ids,seed+31*i+(phase=='assess'))
        fit_rna = p[0, data['fit_idx_'+str(i)]]
        projected, scaler = project_attributes(raw[0]['x_'+RNA_TYPES[i]], fit_rna, seed+17*i)
        data['rna_'+str(i)] = projected
        for key, value in scaler.items():
            data['projection_rna_'+str(i)+'_'+key] = value
    data['drug'], scaler = project_attributes(raw[0]['x_drug'], fit_drugs, seed+73)
    for key, value in scaler.items():
        data['projection_drug_'+key] = value
    graph_path = reference/'inputs'/'LncDrug'/'train_graph.pth'
    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    side_keys = [(('lncRNA','LncLnc','lncRNA'),'side_0'),
                 (('miRNA','MiMi','miRNA'),'side_1'),
                 (('drug','DrugDrug','drug'),'side_drug'),
                 (('lncRNA','LncMi','miRNA'),'cross')]
    for key, name in side_keys:
        a = np.zeros((graph[key[0]].num_nodes, graph[key[2]].num_nodes), dtype=np.float32)
        edge = graph[key].edge_index.cpu().numpy()
        a[tuple(edge)] = 1.
        data[name] = a
    provenance['side_graph'] = dict(path=str(graph_path), relations_read=[list(k) for k,_ in side_keys],
        excluded='Original LncDrug, MiDrug, CircDrug and all their reverse relations are discarded. Associations are reconstructed only from the two outer TRAIN arrays.')
    return data, provenance


def builder(data, task, fitting_drugs, extra_indices=None):
    i = TASKS.index(task) if isinstance(task, str) else int(task)
    j, nd = 1-i, len(data['drug'])
    matrices = [observed_matrix(data['pairs_'+str(k)], data['labels_'+str(k)],
                               len(data['rna_'+str(k)]), nd, fitting_drugs) for k in range(2)]
    if extra_indices is not None and len(extra_indices):
        extra = data['pairs_'+str(i)][:, extra_indices]
        if not np.all(data['labels_'+str(i)][extra_indices] == 1):
            raise ValueError('The support-budget protocol reveals positive records only')
        matrices[i][tuple(extra)] = 1.
    forbidden = np.setdiff1d(np.arange(nd), fitting_drugs)
    # All other-task labels of held drugs must stay absent, even in few-shot.
    if np.any(matrices[j][:, forbidden]):
        raise AssertionError('Other RNA type leaked into a held-out drug')
    f = LinkFeatures(data['rna_'+str(i)], data['drug'], data['side_'+str(i)], data['side_drug'],
        data['cross'] if i == 0 else data['cross'].T, matrices[i], matrices[j], i)
    return f, matrices


def scenario(data, task, phase, fold=None, support_budget=0, full_candidates=False):
    i = TASKS.index(task) if isinstance(task, str) else int(task)
    pairs, y, groups = data['pairs_'+str(i)], data['labels_'+str(i)], data['drug_group']
    fit_drugs = np.flatnonzero((groups < 3) & (groups != fold)) if fold is not None else np.flatnonzero(groups < 3)
    context_ids = np.flatnonzero(np.isin(pairs[1], fit_drugs))
    extra = np.array([], dtype=np.int64)
    if phase == 'fit':
        if fold not in (0,1,2):
            raise ValueError('Fitting queries need an entire held drug fold')
        query_ids = np.flatnonzero(groups[pairs[1]] == fold)
    else:
        if phase not in ('dev','assess'):
            raise ValueError(phase)
        query_ids = np.setdiff1d(data[phase+'_idx_'+str(i)], data[phase+'_reserved_'+str(i)])
        if support_budget:
            selected = []
            reserved = data[phase+'_reserved_'+str(i)]
            for d in np.unique(pairs[1,reserved]):
                selected.extend(reserved[pairs[1,reserved] == d][:support_budget].tolist())
            extra = np.asarray(selected, dtype=np.int64)
    if np.intersect1d(context_ids,query_ids).size or np.intersect1d(extra,query_ids).size:
        raise AssertionError('Query/context overlap')
    priority = np.arange(len(context_ids), len(context_ids)+len(extra), dtype=np.int64)
    context_ids = np.concatenate([context_ids,extra])
    f, matrices = builder(data, i, fit_drugs, extra)
    cp, qp = pairs[:,context_ids].T.copy(), pairs[:,query_ids].T.copy()
    result = dict(cx=f.transform(cp.T,context=True), cy=y[context_ids].astype(np.int64),cp=cp,
                  qx=f.transform(qp.T),qy=y[query_ids].astype(np.float32),qp=qp,
                  context_indices=context_ids,query_indices=query_ids,priority=priority,
                  support_counts=matrices[i].sum(0).astype(np.int64))
    if full_candidates:
        query_drugs = np.unique(qp[:,1])
        r,d = np.meshgrid(np.arange(len(data['rna_'+str(i)])),query_drugs,indexing='ij')
        keep = matrices[i][r,d] == 0
        all_pairs = np.stack([r[keep],d[keep]],axis=1)
        result.update(all_x=f.transform(all_pairs.T),all_p=all_pairs)
    result['association_counts'] = np.asarray([a.sum() for a in matrices],dtype=np.int64)
    result['held_drugs'] = np.setdiff1d(np.unique(qp[:,1]),fit_drugs)
    # For budget 0, neither RNA type supplies any association for queried drugs.
    if not support_budget and any(np.any(a[:,result['held_drugs']]) for a in matrices):
        raise AssertionError('Cold-drug association leakage')
    return result


def summary(data):
    rows = []
    for i, task in enumerate(TASKS):
        pairs, y = data['pairs_'+str(i)],data['labels_'+str(i)]
        for phase in ('fit','dev','assess'):
            ids = data[phase+'_idx_'+str(i)]
            reserve = data.get(phase+'_reserved_'+str(i),np.array([],dtype=np.int64))
            queries = np.setdiff1d(ids,reserve)
            eligible = [int(d) for d in np.unique(pairs[1,queries]) if len(np.unique(y[queries[pairs[1,queries]==d]]))==2]
            rows.append(dict(task=task,phase=phase,drugs=int(len(np.unique(pairs[1,ids]))),
                rows=len(ids),positives=int(y[ids].sum()),reserved_positives=len(reserve),
                query_rows=len(queries),query_positives=int(y[queries].sum()),macro_drugs=len(eligible)))
    return rows
