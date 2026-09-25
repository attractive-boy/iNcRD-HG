"""Matched frozen drug representations and training-only compression."""
from pathlib import Path

import numpy as np

from improve import tabular_foundation as base
from improve.rna_functional_modules import project


DEFAULT_OUT = Path(__file__).resolve().parent/'results/drug_multimodal_inputs_20260914_v3'
VARIANTS = ('descriptors','fingerprint','chemberta','m2umol','m2umol_shuffled')
SEED = 20260914


def drug_permutation(simple,present,support_ids,seed=SEED):
    """Preserve coverage, support membership, size/TPSA tertiles and fragments."""
    present = np.asarray(present,bool)
    support = np.zeros(len(present),bool)
    support[np.unique(support_ids)] = True
    fit = np.flatnonzero(present & support)
    bins,cuts = [],[]
    for column in (1,3):
        cut = np.unique(np.quantile(simple[fit,column],[1/3,2/3])) if len(fit) else np.array([])
        bins.append(np.digitize(simple[:,column],cut))
        cuts.append(cut.tolist())
    groups = {}
    for i in np.flatnonzero(present):
        key = (int(support[i]),int(bins[0][i]),int(bins[1][i]),float(simple[i,7]))
        groups.setdefault(key,[]).append(int(i))
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(present))
    for key in sorted(groups):
        ids = np.asarray(groups[key],int)
        if len(ids)>1:
            ids = rng.permutation(ids)
            permutation[ids] = np.roll(ids,1)
    np.testing.assert_array_equal(present[permutation],present)
    np.testing.assert_array_equal(support[permutation],support)
    return permutation,dict(seed=seed,covered=int(present.sum()),groups=len(groups),
        moved=int(np.sum(permutation!=np.arange(len(present)))),
        moved_fraction=float(np.mean(permutation[present]!=np.flatnonzero(present))) if present.any() else 0.,
        cutpoints=cuts,null='One deterministic derangement within fixed strata; simple descriptors stay attached.')


def verify(directory):
    directory = Path(directory)
    p = base.check_seal(base.read_json(directory/'INPUTS.json'))
    for name,digest in p['files'].items():
        if base.digest(directory/name)!=digest:
            raise ValueError('Frozen drug feature changed: '+name)
    for name,digest in p['inputs'].items():
        if base.digest(name)!=digest:
            raise ValueError('Frozen drug input changed: '+name)
    if not p['checks']['used_parameters_complete'] or not p['checks']['parameters_unchanged']:
        raise ValueError('Incomplete or changed pretrained encoder')
    return p


class DrugFeatures:
    def __init__(self,directory,task=None):
        with np.load(Path(directory)/'representations.npz',allow_pickle=False) as z:
            self.table = {k:z[k] for k in z.files}
        self.present = self.table['present'].astype(bool)
        self.n_drugs = len(self.present)
        if self.table['simple'].shape != (self.n_drugs,8):
            raise ValueError('Expected eight shared drug descriptors')
        for name in ('m2umol','chemberta','fingerprint'):
            x = self.table[name]
            if len(x)!=self.n_drugs or not np.isfinite(x).all() or np.any(x[~self.present]!=0):
                raise ValueError('Unmatched coverage or nonfinite drug representation')

    def transform(self,variant,support_pairs):
        if variant not in VARIANTS:
            raise ValueError('Unregistered drug representation')
        support = np.unique(np.asarray(support_pairs,int)[1])
        state,meta,shuffle = {},{},{}
        permutation = np.arange(self.n_drugs)
        if variant=='descriptors':
            compressed = np.zeros((self.n_drugs,16),np.float32)
            meta = dict(fit_entities=0,components=0,explained_variance_fraction=0.)
        else:
            key = 'm2umol' if variant=='m2umol_shuffled' else variant
            if variant=='m2umol_shuffled':
                permutation,shuffle = drug_permutation(self.table['simple'],self.present,support)
            compressed,state,meta = project(self.table[key][permutation],self.present,support,16)
        state['permutation'] = permutation
        meta.update(shuffle=shuffle,support_ids_sha256=base.json_digest(support.tolist()),
            output_columns=24,projection_columns=16,covered_drugs=int(self.present.sum()))
        return np.concatenate([self.table['simple'],compressed],axis=1),state,meta

    def select(self,table,pairs,variant):
        return table[pairs[1]]
