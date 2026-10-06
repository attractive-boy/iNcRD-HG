"""Direction B (v2): external literature-exposure covariates from PubMed, two feature sets.

  bib      : general volumes  log1p(n_all), log1p(n_cancer) per RNA and drug, cancer fractions,
             product, and two curation-saturation columns log1p(train degree) - log1p(n_cancer)
             (degree columns are the bank's own cross-fitted degree_r / degree_d).
  bib_task : bib + task-relevant exposure: drug x ncRNA-literature volume, drug x resistance volume,
             RNA x resistance/chemotherapy volume, and the RNA x drug product of resistance volumes.
Controls (`_perm`): counts permuted across entities of each type with a fixed seed (marginals kept,
entity map destroyed).  Members hgb/et/mlp/lr on context76; fused into the frozen reference at w.
"""
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance_channel_v1 import surfaces, member_logits

HERE = Path(__file__).resolve().parent
BIB = Path('/data1/licheng/incrd-hg/data_external/bibliometrics_20260917')
SEED = 20260917


def load_counts(which, n):
    frames = [pd.read_csv(p) for p in glob.glob(str(BIB / ('pubmed_%s*.csv' % which))) if 'class' not in p]
    d = pd.concat(frames).drop_duplicates('index').sort_values('index')
    d = d[d.n_all >= 0]
    assert len(d) == n and (d['index'].values == np.arange(n)).all(), ('incomplete counts', which, len(d))
    return np.log1p(d.n_all.values.astype(float)), np.log1p(d.n_cancer.values.astype(float))


def load_class(which, n):
    frames = [pd.read_csv(p) for p in glob.glob(str(BIB / ('pubmed_class_%s_*.csv' % which)))]
    d = pd.concat(frames).drop_duplicates('index').sort_values('index')
    d = d[d.n1 >= 0]
    assert len(d) == n and (d['index'].values == np.arange(n)).all(), ('incomplete class counts', which, len(d))
    n1 = np.log1p(d.n1.values.astype(float))
    n2 = np.log1p(pd.to_numeric(d.n2, errors='coerce').fillna(0).values.astype(float)) if 'n2' in d else None
    return n1, n2


class Exposure:
    def __init__(self, task, permute, with_task):
        nr = 605 if task == 'MiDrug' else 1322
        rkey = 'mirna' if task == 'MiDrug' else 'lnc'
        self.ra, self.rc = load_counts(rkey, nr)
        self.da, self.dc = load_counts('drug', 216)
        self.with_task = with_task
        if with_task:
            self.d_nc, self.d_res = load_class('drug', 216)
            self.r_res, _ = load_class(rkey, nr)
        if permute:
            rng = np.random.default_rng(SEED)
            pr, pd_ = rng.permutation(nr), rng.permutation(216)
            self.ra, self.rc, self.da, self.dc = self.ra[pr], self.rc[pr], self.da[pd_], self.dc[pd_]
            if with_task:
                self.d_nc, self.d_res, self.r_res = self.d_nc[pd_], self.d_res[pd_], self.r_res[pr]

    def features(self, pairs, bank_x):
        r, d = pairs[0], pairs[1]
        deg_r, deg_d = bank_x[:, 0], bank_x[:, 1]
        cols = [self.ra[r], self.rc[r], self.da[d], self.dc[d], self.rc[r] - self.ra[r], self.dc[d] - self.da[d], self.rc[r] * self.dc[d],
                np.log1p(deg_r) - self.rc[r], np.log1p(deg_d) - self.dc[d]]
        if self.with_task:
            cols += [self.d_nc[d], self.d_res[d], self.r_res[r], self.r_res[r] * self.d_res[d], self.d_nc[d] - self.da[d], self.r_res[r] - self.ra[r]]
        return np.stack(cols, 1).astype(np.float32)


if __name__ == '__main__':
    task, sname, mode = sys.argv[1], sys.argv[2], sys.argv[3]   # mode: bib | bib_task
    out_dir = HERE / ('cells_' + mode); out_dir.mkdir(exist_ok=True)
    dest = out_dir / ('%s_%s.npz' % (task, sname))
    if dest.exists():
        sys.exit(0)
    s = next(x for x in surfaces(task) if x['name'] == sname)
    with np.load(s['bank']('context76'), allow_pickle=False) as z:
        x_tr, y_tr, x_q, ptr, pq = z['x_train'], z['y_train'], z['x_val'], z['pairs_train'], z['pairs_val']
    with np.load(HERE / ('cells/%s_%s_context76.npz' % (task, sname)), allow_pickle=False) as z:
        prov_tr, prov_q = z['feat_train_real'], z['feat_val_real']
    store = dict(pairs_val=pq)
    for label, permute in ((mode, False), (mode + '_perm', True)):
        enc = Exposure(task, permute, mode == 'bib_task')
        ft, fq = enc.features(ptr, x_tr), enc.features(pq, x_q)
        store['feat_train_' + label], store['feat_val_' + label] = ft, fq
        for name, v in member_logits(np.c_[x_tr, ft], y_tr, np.c_[x_q, fq]).items():
            store['logit_%s_%s' % (label, name)] = v
        for name, v in member_logits(np.c_[x_tr, prov_tr, ft], y_tr, np.c_[x_q, prov_q, fq]).items():
            store['logit_prov+%s_%s' % (label, name)] = v
        print('fitted', task, sname, label, flush=True)
    np.savez_compressed(dest, **store)
