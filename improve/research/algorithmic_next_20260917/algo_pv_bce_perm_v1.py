"""Adds the permuted-label BCE control (`pv_bce_perm`) for scheme 1 into cells_algoE.

usage: algo_pv_bce_perm_v1.py <task> <surface> <cells_dir>
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917')
from algo_common import load_bank  # noqa: E402
from algo_pairwise_v1 import fit  # noqa: E402

PROV = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917/cells')
SEED = 20260917


def main(task, sname, cells_dir):
    dest = Path(cells_dir) / ('%s_%s.npz' % (task, sname))
    with np.load(dest, allow_pickle=False) as z:
        store = {k: z[k] for k in z.files}
    if 'logit_pv_bce_perm_member' in store:
        return
    _, d = load_bank(task, sname)
    with np.load(PROV / ('%s_%s_context76.npz' % (task, sname)), allow_pickle=False) as z:
        ptr, pv = z['feat_train_real'], z['feat_val_real']
    x_tr = np.c_[d['x_train'], ptr].astype(np.float32)
    x_val = np.c_[d['x_val'], pv].astype(np.float32)
    rng = np.random.default_rng(SEED)
    store['logit_pv_bce_perm_member'] = fit(x_tr, rng.permutation(d['y_train']), d['pairs_train'][1], x_val, 'bce').astype(np.float32)
    np.savez_compressed(dest, **store)
    print('fitted', task, sname, 'pv_bce_perm', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
