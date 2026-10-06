"""Provenance column-subset decomposition (diagnostic + candidate techniques).

The real provenance channel has two conceptual parts:
  collab   : pair-level co-membership. A PMID hyperedge that already links r and d
             through OTHER training positives (shared_pmid, inverse-breadth weighted),
             plus experimental-context token co-occurrence (cell line / phenotype /
             disease, IDF weighted).  This is evidence that the two entities are
             studied *together*, not that either is popular.
  exposure : entity-level study volume.  Distinct training PMIDs, record counts and
             high-throughput fraction per RNA and per drug.  This is a popularity /
             sampling-propensity covariate.

If only `exposure` helps, the channel is another degree prior.  If `collab` helps on
its own, the gain is genuine cross-entity study information.  Labels written into a
new cells dir so evaluate_generic_v1.py can score them against the same references.

usage: provenance_subset_v1.py <task> <surface> <src_cells> <out_cells>
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance_channel_v1 import surfaces, member_logits  # noqa: E402

HERE = Path(__file__).resolve().parent
# feat columns: 0 shared_pmid, 1 shared_pmid_w, 2 rna_npmid, 3 drug_npmid,
#               4 rna_nrec, 5 drug_nrec, 6 rna_ht_frac, 7 drug_ht_frac,
#               8 cond_idf, 9 pheno_idf, 10 disease_idf, 11 cond_n, 12 pheno_n, 13 disease_n
COLLAB = [0, 1, 8, 9, 10, 11, 12, 13]
EXPOSURE = [2, 3, 4, 5, 6, 7]
SHARED = [0, 1]
CTX = [8, 9, 10, 11, 12, 13]
RECIPES = {
    'prov_collab': COLLAB,
    'prov_exposure': EXPOSURE,
    'prov_shared': SHARED,
    'prov_ctxonly': CTX,
}


def main(task, sname, src, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / ('%s_%s.npz' % (task, sname))
    if dest.exists():
        return
    s = next(x for x in surfaces(task) if x['name'] == sname)
    with np.load(s['bank']('context76'), allow_pickle=False) as z:
        x_tr, y_tr, x_q = z['x_train'], z['y_train'], z['x_val']
    with np.load(src / ('%s_%s_context76.npz' % (task, sname)), allow_pickle=False) as z:
        store = dict(z.files and {k: z[k] for k in z.files} or {})
    ft, fq = store['feat_train_real'], store['feat_val_real']
    for label, cols in RECIPES.items():
        if ('logit_%s_hgb' % label) in store:
            continue
        xt = np.c_[x_tr, ft[:, cols]]
        xq = np.c_[x_q, fq[:, cols]]
        for name, v in member_logits(xt, y_tr, xq).items():
            store['logit_%s_%s' % (label, name)] = v
        print('fitted', task, sname, label, flush=True)
    np.savez_compressed(dest, **store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
