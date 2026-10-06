"""Direction B (v3): external PubMed literature-exposure covariates, corrected.

v2 read the accepted bank's columns 0/1 as raw entity degrees; those columns are
standardised, so log1p() of them is NaN and the ExtraTrees member rejects the
matrix.  v3 computes the entity degrees itself from the TRAINING pairs, cross-fitted
with the frozen fold ids exactly like the provenance channel.

Features per query (r, d):
  exp_r_log, exp_r_cancer_log, exp_d_log, exp_d_cancer_log,
  exp_r_cancer_frac, exp_d_cancer_frac, exp_r_cancer_x_d_cancer,
  sat_r = log1p(train_deg_r) - exp_r_cancer_log,
  sat_d = log1p(train_deg_d) - exp_d_cancer_log
where train_deg are counts of training positives per entity (fold-positive removal
for training rows).  Controls permute the PubMed counts across entities of each type.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance_channel_v1 import surfaces, member_logits  # noqa: E402
from bibliometric_exposure_v2 import Exposure  # noqa: E402

HERE = Path(__file__).resolve().parent


def entity_degrees(ptr, ytr, fold_ids, pq):
    """Cross-fitted positive-degree per RNA and per drug for train rows; full for query."""
    n_r = int(max(ptr[0].max(), pq[0].max())) + 1
    n_d = int(max(ptr[1].max(), pq[1].max())) + 1
    deg_r = np.zeros((ptr.shape[1], 2), dtype=np.float32)  # per train row (deg_r, deg_d)
    for k in np.unique(fold_ids):
        m = fold_ids != k
        pos = ptr[:, m][:, ytr[m] == 1]
        cr = np.bincount(pos[0], minlength=n_r).astype(np.float32)
        cd = np.bincount(pos[1], minlength=n_d).astype(np.float32)
        q = np.flatnonzero(fold_ids == k)
        deg_r[q, 0] = cr[ptr[0, q]]
        deg_r[q, 1] = cd[ptr[1, q]]
    pos = ptr[:, ytr == 1]
    cr = np.bincount(pos[0], minlength=n_r).astype(np.float32)
    cd = np.bincount(pos[1], minlength=n_d).astype(np.float32)
    deg_q = np.c_[cr[pq[0]], cd[pq[1]]].astype(np.float32)
    return deg_r, deg_q


def main(task, sname, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / ('%s_%s.npz' % (task, sname))
    if dest.exists():
        return
    s = next(x for x in surfaces(task) if x['name'] == sname)
    with np.load(s['bank']('context76'), allow_pickle=False) as z:
        x_tr, y_tr, x_q, ptr, pq = z['x_train'], z['y_train'], z['x_val'], z['pairs_train'], z['pairs_val']
    with np.load(s['folds'], allow_pickle=False) as z:
        fold_ids = z['fold_ids']
    deg_tr, deg_q = entity_degrees(ptr, y_tr, fold_ids, pq)
    store = dict(pairs_val=pq)
    for label, permute in (('bib', False), ('bib_perm', True)):
        enc = Exposure(task, permute, False)

        def base(pairs, deg):
            r, d = pairs[0], pairs[1]
            ra, rc, da, dc = enc.ra[r], enc.rc[r], enc.da[d], enc.dc[d]
            return np.c_[ra, rc, da, dc, rc - ra, dc - da, rc * dc,
                         np.log1p(deg[:, 0]) - rc, np.log1p(deg[:, 1]) - dc]

        ft = base(ptr, deg_tr)
        fq = base(pq, deg_q)
        ft = np.nan_to_num(ft, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        fq = np.nan_to_num(fq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        store['feat_train_' + label], store['feat_val_' + label] = ft, fq
        for name, v in member_logits(np.c_[x_tr, ft], y_tr, np.c_[x_q, fq]).items():
            store['logit_%s_%s' % (label, name)] = v
        print('fitted', task, sname, label, flush=True)
    np.savez_compressed(dest, **store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
