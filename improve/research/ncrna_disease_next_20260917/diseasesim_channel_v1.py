"""Disease-disease similarity channel.

Two diseases are similar if the same drugs are annotated to them (CTD
drug-disease). For a pair (r, d): U = sum of normalised disease profiles of the
RNA's diseases, V = the same for the drug's diseases, feature = U . V. This
captures related-but-not-identical diseases, unlike exact-overlap features.
"""
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
import importlib.util  # noqa: E402

HERE = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/ncrna_disease_next_20260917'
spec = importlib.util.spec_from_file_location('dav', HERE + '/disease_anchor_verify_v1.py')
dav = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dav)


class DiseaseSimChannel:
    def __init__(self, permute=False, seed=20260917):
        self.base = dav.Channel(permute=permute, seed=seed)
        drug_dis = self.base.drug_dis
        self.drugs = sorted(drug_dis)
        diseases = sorted({x for v in drug_dis.values() for x in v})
        self.didx = {d: i for i, d in enumerate(diseases)}
        rows, cols = [], []
        for j, drug in enumerate(self.drugs):
            for dis in drug_dis[drug]:
                rows.append(self.didx[dis])
                cols.append(j)
        W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(diseases), len(self.drugs)))
        scale = 1.0 / (np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel()) + 1e-9)
        self.Wn = sp.diags(scale) @ W  # disease x drug, rows normalised

    def _U(self, keys):
        inds = [self.didx[k] for k in keys if k in self.didx]
        if not inds:
            return np.zeros(len(self.drugs))
        return np.asarray(self.Wn[inds].sum(0)).ravel()

    def features(self, task, pairs):
        sets = self.base.rna_sets(task)
        rows = []
        for r, d in zip(pairs[0], pairs[1]):
            rs = sets[r]
            ds = self.base.drug_dis.get(self.base.drug_names[d], set())
            u, v = self._U(rs), self._U(ds)
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            rows.append([float(u @ v), float(u @ v) / (nu * nv + 1e-9), len(rs), len(ds)])
        return np.asarray(rows, dtype=np.float32)
