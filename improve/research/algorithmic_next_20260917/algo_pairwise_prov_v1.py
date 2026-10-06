"""Scheme E: cross-drug ranking objective on the provenance-augmented bank.

Combines the two things that survived independently: the curation-provenance channel
(new information) and the cross-drug pairwise objective (algorithm).  Features are
[context76 bank | 14 provenance columns]; the scorer is an MLP trained with the
cross-drug pairwise logistic loss (Scheme A) versus the same net with BCE.  If the
objective adds on top of the channel the ranking must improve; if the channel already
carries everything, the two are equal.

usage: algo_pairwise_prov_v1.py <task> <surface> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917')
from algo_common import load_bank, save  # noqa: E402
from algo_pairwise_v1 import fit  # noqa: E402

PROV = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917/cells')


def main(task, sname, out_dir):
    _, d = load_bank(task, sname)
    with np.load(PROV / ('%s_%s_context76.npz' % (task, sname)), allow_pickle=False) as z:
        ptr, pv = z['feat_train_real'], z['feat_val_real']
    x_tr = np.c_[d['x_train'], ptr].astype(np.float32)
    x_val = np.c_[d['x_val'], pv].astype(np.float32)
    drug = d['pairs_train'][1]
    rng = np.random.default_rng(20260917)
    store = dict(pairs_val=d['pairs_val'])
    for label, (yy, obj) in {'pv_rank': (d['y_train'], 'pair'), 'pv_bce': (d['y_train'], 'bce'),
                             'pv_rank_perm': (rng.permutation(d['y_train']), 'pair')}.items():
        store['logit_%s_member' % label] = fit(x_tr, yy, drug, x_val, obj).astype(np.float32)
        print('fitted', task, sname, label, flush=True)
    save(out_dir, task, sname, store)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
