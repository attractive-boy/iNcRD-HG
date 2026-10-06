"""Pan-ncRNA extension of the provenance channel: study hyperedges also include the OTHER
RNA type's training-positive records on the same surface (the accepted protocol already
exposes the other task's training associations as graph side information).  The RNA side
of the query still uses only its own task's records; the drug side gains PMIDs and context
tokens from both RNA types.  Control: PMID/context columns permuted across records of both
tasks.  Members hgb/et/mlp/lr on context76, fused into the reference at weight w.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance_channel_v1 import surfaces, load_records, Provenance, member_logits, PMID_NAMES, CTX_NAMES, RECORDS, CONTROL_SEED

HERE = Path(__file__).resolve().parent
OTHER = {'MiDrug': 'LncDrug', 'LncDrug': 'MiDrug'}


def load_both(task, variant):
    rec = pd.read_csv(RECORDS)
    for col in ('cond', 'pheno', 'diseases'):
        rec[col] = rec[col].fillna('').astype(str)
    rng = np.random.default_rng(CONTROL_SEED + 1)
    if variant == 'perm':
        rec['PMID'] = rec.PMID.values[rng.permutation(len(rec))]
        rec['ht'] = rec.ht.values[rng.permutation(len(rec))]
        for col in ('cond', 'pheno', 'diseases'):
            rec[col] = rec[col].values[rng.permutation(len(rec))]
    return rec


class PanProvenance(Provenance):
    """Own-task records for RNA side; both tasks' records for the drug side."""
    def __init__(self, rec, task, support_own, support_other):
        own = rec[rec.task == task]
        oth = rec[rec.task == OTHER[task]]
        keep_own = np.array([(int(a), int(b)) in support_own for a, b in zip(own.ri, own.di)])
        keep_oth = np.array([(int(a), int(b)) in support_other for a, b in zip(oth.ri, oth.di)])
        R_own, R_oth = own[keep_own], oth[keep_oth]
        # RNA side from own; drug side from own + other (other-type RNAs are offset so they never collide)
        offset = 10 ** 6
        R_oth = R_oth.assign(ri=R_oth.ri + offset)
        R = pd.concat([R_own, R_oth])
        support = set(zip(R.ri.astype(int), R.di.astype(int)))
        super().__init__(R, support)


def channel_arrays(rec, task, ptr, ytr, fold_ids, pq, other_support):
    pos = ptr[:, ytr == 1]
    all_support = set(zip(pos[0].tolist(), pos[1].tolist()))
    x_train = np.zeros((ptr.shape[1], len(PMID_NAMES) + len(CTX_NAMES)), dtype=np.float32)
    for k in np.unique(fold_ids):
        q = np.flatnonzero(fold_ids == k)
        fold_pos = set(zip(ptr[0, q][ytr[q] == 1].tolist(), ptr[1, q][ytr[q] == 1].tolist()))
        enc = PanProvenance(rec, task, all_support - fold_pos, other_support)
        x_train[q] = enc.features(ptr[:, q])
    enc = PanProvenance(rec, task, all_support, other_support)
    return x_train, enc.features(pq)


if __name__ == '__main__':
    task, sname = sys.argv[1], sys.argv[2]
    out_dir = HERE / 'cells_pan'
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / ('%s_%s.npz' % (task, sname))
    if dest.exists():
        sys.exit(0)
    s = next(x for x in surfaces(task) if x['name'] == sname)
    so = next(x for x in surfaces(OTHER[task]) if x['name'] == sname)
    with np.load(s['bank']('context76'), allow_pickle=False) as z:
        x_tr, y_tr, x_q, ptr, pq = z['x_train'], z['y_train'], z['x_val'], z['pairs_train'], z['pairs_val']
    with np.load(s['folds'], allow_pickle=False) as z:
        fold_ids = z['fold_ids']
    with np.load(so['bank']('context76'), allow_pickle=False) as z:
        po, yo = z['pairs_train'], z['y_train']
    other_support = set(zip(po[0][yo == 1].tolist(), po[1][yo == 1].tolist()))
    store = dict(pairs_val=pq)
    for variant in ('real', 'perm'):
        rec = load_both(task, variant)
        ft, fq = channel_arrays(rec, task, ptr, y_tr, fold_ids, pq, other_support)
        store['feat_train_' + variant], store['feat_val_' + variant] = ft, fq
        ml = member_logits(np.c_[x_tr, ft], y_tr, np.c_[x_q, fq])
        for name, v in ml.items():
            store['logit_pan_%s_%s' % (variant, name)] = v
        print('fitted', task, sname, variant, flush=True)
    np.savez_compressed(dest, **store)
