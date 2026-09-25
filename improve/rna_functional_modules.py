"""Support-only compression and coverage-preserving RNA ablations."""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from improve import tabular_foundation as base
from improve import rna_functional_inputs as inputs


VARIANTS = ('simple', 'generic', 'functional_mean', 'functional_moments', 'functional_shuffled')
DIMENSION = 16
SHUFFLE_SEED = 20260914


def stratified_permutation(simple, present, support_ids, seed=SHUFFLE_SEED):
    """Derange within coverage, support membership and length/GC/count tertiles."""
    present = np.asarray(present, bool)
    support = np.zeros(len(present), bool)
    support[np.unique(support_ids)] = True
    fit = np.flatnonzero(present & support)
    bins = np.zeros((len(present), 3), int)
    cuts = []
    for j, column in enumerate((2, 3, 6)):
        edges = np.unique(np.quantile(simple[fit,column],[1/3,2/3])) if len(fit) else np.array([])
        bins[:,j] = np.digitize(simple[:,column],edges)
        cuts.append(edges.tolist())
    groups = {}
    for i in np.flatnonzero(present):
        groups.setdefault((int(support[i]),*bins[i]),[]).append(int(i))
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(present))
    for key in sorted(groups):
        ids = np.array(groups[key],int)
        if len(ids)>1:
            ids = rng.permutation(ids)
            permutation[ids] = np.roll(ids,1)
    np.testing.assert_array_equal(present[permutation],present)
    np.testing.assert_array_equal(support[permutation],support)
    np.testing.assert_array_equal(bins[permutation[present]],bins[present])
    metadata = dict(seed=seed,cutpoints=cuts,strata=len(groups),covered=int(present.sum()),
        moved=int(np.sum((permutation!=np.arange(len(present))) & present)),
        moved_fraction=float(np.mean(permutation[present]!=np.flatnonzero(present))) if present.any() else 0.,
        singleton_strata=sum(len(g)==1 for g in groups.values()),
        limitations='One deterministic within-stratum derangement, not a uniform null distribution. Simple metadata and RNA coverage remain attached to the original entity.')
    return permutation,metadata


def project(payload, present, support_ids, dimension=DIMENSION):
    """Fit one row per covered support RNA, never a query-only RNA or a label."""
    payload = np.asarray(payload,np.float64)
    present = np.asarray(present,bool)
    if payload.ndim!=2 or len(present)!=len(payload) or not np.isfinite(payload).all():
        raise ValueError('Invalid representation matrix')
    fit_ids = np.unique(np.asarray(support_ids,int))
    fit_ids = fit_ids[present[fit_ids]]
    output = np.zeros((len(payload),dimension),np.float32)
    state = dict(fit_ids=fit_ids)
    if len(fit_ids)<2:
        return output,state,dict(fit_entities=len(fit_ids),components=0,explained_variance_fraction=0.)
    scaler = StandardScaler().fit(payload[fit_ids])
    x = scaler.transform(payload[fit_ids])
    n = min(dimension,len(fit_ids)-1,payload.shape[1])
    pca = PCA(n_components=n,svd_solver='full').fit(x)
    scale = np.sqrt(np.maximum(pca.explained_variance_,1e-10))
    q = pca.transform(scaler.transform(payload[present]))/scale
    output[present,:n] = np.clip(q,-10,10).astype(np.float32)
    state.update(mean=scaler.mean_,scale=scaler.scale_,pca_mean=pca.mean_,components=pca.components_,
                 component_scale=scale,explained_variance=pca.explained_variance_)
    meta = dict(fit_entities=len(fit_ids),components=n,
                explained_variance_fraction=float(pca.explained_variance_ratio_.sum()),
                clipped_coordinates=int((np.abs(q)>10).sum()))
    return output,state,meta


class Representations:
    def __init__(self,tables):
        self.tables = tables
        for task,data in tables.items():
            n = len(data['present'])
            if data['simple'].shape != (n,8) or len(data['names']) != n:
                raise ValueError('Representation table shape mismatch: '+task)
            for key in ('generic','functional_mean','functional_moments'):
                if len(data[key]) != n or not np.isfinite(data[key]).all() or np.any(data[key][~data['present']] != 0):
                    raise ValueError('Uncovered or nonfinite representation: '+key)

    @classmethod
    def load(cls,directory):
        directory = Path(directory)
        inputs.verify(directory/'assembled')
        tables = {}
        for task in ('MiDrug','LncDrug'):
            with np.load(directory/'assembled'/(task+'.npz'),allow_pickle=False) as z:
                tables[task] = {k:z[k] for k in z.files}
        return cls(tables)

    def transform(self,task,variant,support_ids,dimension=DIMENSION):
        if variant not in VARIANTS:
            raise ValueError('Unregistered variant')
        data = self.tables[task]
        permutation = np.arange(len(data['present']))
        shuffle = {}
        if variant=='simple':
            compressed = np.zeros((len(permutation),dimension),np.float32)
            state,metadata = {},dict(fit_entities=0,components=0,explained_variance_fraction=0.)
        else:
            key = 'functional_moments' if variant=='functional_shuffled' else variant
            if variant=='functional_shuffled':
                permutation,shuffle = stratified_permutation(data['simple'],data['present'],support_ids)
            compressed,state,metadata = project(data[key][permutation],data['present'],support_ids,dimension)
        state['permutation'] = permutation
        metadata['shuffle'] = shuffle
        metadata['support_entity_ids_sha256'] = base.json_digest(np.unique(support_ids).tolist())
        return np.concatenate([data['simple'],compressed],axis=1),state,metadata
