"""Direction B: external literature-exposure covariates (PubMed entity-level volumes).

Observation model: a curated association is recorded only if the pair was studied, and the
chance a pair is studied scales with each entity's literature volume.  PubMed esearch
counts (all / cancer-context) per RNA and per drug are label-free external covariates of
that exposure process; they are appended to the accepted context76 bank together with two
derived "curation saturation" columns (log1p(train degree) - log1p(cancer volume)) that
use the bank's own cross-fitted degree columns.  Control: PubMed counts permuted across
entities of each type (fixed seed), preserving marginals but destroying the entity map.
Members hgb/et/mlp/lr, fused into the frozen reference at weight w.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance_channel_v1 import surfaces, member_logits

HERE = Path(__file__).resolve().parent
BIB = Path('/data1/licheng/incrd-hg/data_external/bibliometrics_20260917')
NAMES = ('bib_rna_log_all', 'bib_rna_log_cancer', 'bib_drug_log_all', 'bib_drug_log_cancer', 'bib_rna_cancer_frac', 'bib_drug_cancer_frac',
         'bib_product', 'bib_rna_saturation', 'bib_drug_saturation')
SEED = 20260917


def counts(task, permute):
    d = pd.read_csv(BIB / 'pubmed_drug.csv').sort_values('index')
    r = pd.read_csv(BIB / ('pubmed_mirna.csv' if task == 'MiDrug' else 'pubmed_lnc.csv')).sort_values('index')
    assert (d.n_all >= 0).all() and (r.n_all >= 0).all(), 'failed PubMed queries present'
    da, dc = np.log1p(d.n_all.values.astype(float)), np.log1p(d.n_cancer.values.astype(float))
    ra, rc = np.log1p(r.n_all.values.astype(float)), np.log1p(r.n_cancer.values.astype(float))
    if permute:
        rng = np.random.default_rng(SEED)
        pd_, pr = rng.permutation(len(da)), rng.permutation(len(ra))
        da, dc, ra, rc = da[pd_], dc[pd_], ra[pr], rc[pr]
    return ra, rc, da, dc


def features(task, pairs, bank_x, permute):
    ra, rc, da, dc = counts(task, permute)
    r, d = pairs[0], pairs[1]
    deg_r, deg_d = bank_x[:, 0], bank_x[:, 1]   # cross-fitted degree_r / degree_d columns of the accepted bank
    out = np.c_[ra[r], rc[r], da[d], dc[d], rc[r] - ra[r], dc[d] - da[d], rc[r] * dc[d],
                np.log1p(deg_r) - rc[r], np.log1p(deg_d) - dc[d]]
    return out.astype(np.float32)


if __name__ == '__main__':
    task, sname = sys.argv[1], sys.argv[2]
    out_dir = HERE / 'cells_bib'
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / ('%s_%s.npz' % (task, sname))
    if dest.exists():
        sys.exit(0)
    s = next(x for x in surfaces(task) if x['name'] == sname)
    with np.load(s['bank']('context76'), allow_pickle=False) as z:
        x_tr, y_tr, x_q, ptr, pq = z['x_train'], z['y_train'], z['x_val'], z['pairs_train'], z['pairs_val']
    store = dict(pairs_val=pq)
    for label, permute in (('bib', False), ('bib_perm', True)):
        ft, fq = features(task, ptr, x_tr, permute), features(task, pq, x_q, permute)
        store['feat_train_' + label], store['feat_val_' + label] = ft, fq
        for name, v in member_logits(np.c_[x_tr, ft], y_tr, np.c_[x_q, fq]).items():
            store['logit_%s_%s' % (label, name)] = v
        print('fitted', task, sname, label, flush=True)
    np.savez_compressed(dest, **store)
