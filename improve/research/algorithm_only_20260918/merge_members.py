"""Average logits of new algorithm-only members into a combined member.

usage: merge_members.py <out_dir> <combo_label> <src_dir:label> <src_dir:label> ...
Each source contributes logit_<label>_member (and _perm if present).
"""
import sys
from pathlib import Path

import numpy as np

SURFACES = ('inner', 's42', 's2026', 's31415', 'x7', 'x77', 'x777')
TASKS = ('MiDrug', 'LncDrug')


def main(out_dir, combo, srcs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for s in SURFACES:
            ref = None
            reals, perms = [], []
            for src in srcs:
                d, lab = src.split(':')
                z = np.load(Path(d) / ('%s_%s.npz' % (task, s)), allow_pickle=False)
                ref = z['pairs_val']
                if 'logit_%s_member' % lab in z.files:
                    reals.append(z['logit_%s_member' % lab])
                if 'logit_%s_perm_member' % lab in z.files:
                    perms.append(z['logit_%s_perm_member' % lab])
            store = dict(pairs_val=ref)
            store['logit_%s_member' % combo] = np.mean(reals, 0).astype(np.float32)
            if perms:
                store['logit_%s_perm_member' % combo] = np.mean(perms, 0).astype(np.float32)
            np.savez_compressed(out_dir / ('%s_%s.npz' % (task, s)), **store)
    print('merged', combo, 'from', srcs)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
