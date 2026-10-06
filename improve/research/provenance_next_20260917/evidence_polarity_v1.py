"""Evidence-polarity channel: resistance vs sensitivity curation per entity.

ncRNADrug curates both *resistance* associations (the prediction target) and
*sensitivity* associations.  The accepted heterogeneous graph keeps only the
resistance edges, so the polarity of an entity's evidence is entirely unused.
DMGAT (Briefings in Bioinformatics 2025) exploits exactly this: sensitivity
curation is reliable negative information.

For a query (r, d), built from TRAINING pairs only (cross-fitted with the frozen
fold ids):
  * entity-level curation profile: number / fraction / log-count of resistant vs
    sensitive records per RNA and per drug, and the resistance-minus-sensitivity
    polarity.
  * pair-level polarity is intentionally omitted: the evaluated pair's own
    records are held out, so it would be zero.

Control: the Effect column is permuted across records of the task with a fixed
seed, preserving marginals but destroying the entity <-> polarity map.

usage: evidence_polarity_v1.py <task> <surface> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance_channel_v1 import surfaces, member_logits, RECORDS, CONTROL_SEED  # noqa: E402

NAMES = ('pol_r_res', 'pol_r_sen', 'pol_d_res', 'pol_d_sen',
         'pol_r_resfrac', 'pol_d_resfrac', 'pol_r_pol', 'pol_d_pol',
         'pol_r_lres', 'pol_d_lres')


def load_records(task, permute):
    rec = pd.read_csv(RECORDS)
    rec = rec[rec.task == task].reset_index(drop=True)
    rec['effect'] = rec.Effect.fillna('').astype(str).str.lower()
    if permute:
        rng = np.random.default_rng(CONTROL_SEED + 11)
        rec['effect'] = rec.effect.values[rng.permutation(len(rec))]
    return rec


class Polarity:
    def __init__(self, rec, support_pairs):
        keep = np.array([(int(a), int(b)) in support_pairs for a, b in zip(rec.ri, rec.di)])
        R = rec[keep]
        self.res_r, self.sen_r = {}, {}
        self.res_d, self.sen_d = {}, {}
        for r, d, e in zip(R.ri, R.di, R.effect):
            r, d = int(r), int(d)
            if e == 'resistant':
                self.res_r[r] = self.res_r.get(r, 0) + 1
                self.res_d[d] = self.res_d.get(d, 0) + 1
            elif e == 'sensitive':
                self.sen_r[r] = self.sen_r.get(r, 0) + 1
                self.sen_d[d] = self.sen_d.get(d, 0) + 1

    def features(self, pairs):
        rows = []
        for r, d in zip(pairs[0], pairs[1]):
            r, d = int(r), int(d)
            rr, sr = self.res_r.get(r, 0), self.sen_r.get(r, 0)
            rd, sd = self.res_d.get(d, 0), self.sen_d.get(d, 0)
            tr, td = rr + sr, rd + sd
            rows.append([rr, sr, rd, sd, rr / max(tr, 1), rd / max(td, 1),
                         np.log1p(rr) - np.log1p(sr), np.log1p(rd) - np.log1p(sd),
                         np.log1p(rr), np.log1p(rd)])
        return np.asarray(rows, dtype=np.float32)


def channel_arrays(rec, ptr, ytr, fold_ids, pq):
    all_support = set(zip(ptr[0].tolist(), ptr[1].tolist()))
    x_train = np.zeros((ptr.shape[1], len(NAMES)), dtype=np.float32)
    for k in np.unique(fold_ids):
        q = np.flatnonzero(fold_ids == k)
        fold_pairs = set(zip(ptr[0, q].tolist(), ptr[1, q].tolist()))
        x_train[q] = Polarity(rec, all_support - fold_pairs).features(ptr[:, q])
    x_val = Polarity(rec, all_support).features(pq)
    return x_train, x_val


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
    store = dict(pairs_val=pq)
    for label, permute in (('pol', False), ('pol_perm', True)):
        rec = load_records(task, permute)
        ft, fq = channel_arrays(rec, ptr, y_tr, fold_ids, pq)
        store['feat_train_' + label], store['feat_val_' + label] = ft, fq
        for name, v in member_logits(np.c_[x_tr, ft], y_tr, np.c_[x_q, fq]).items():
            store['logit_%s_%s' % (label, name)] = v
        print('fitted', task, sname, label, flush=True)
    np.savez_compressed(dest, **store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
