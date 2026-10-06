"""Shared helpers for the algorithmic-level experiments (2026-09-17).

Each scheme produces a per-(task, surface) npz holding
    logit_<label>_<member> arrays  +  pairs_val
exactly like the provenance channel, so `evaluate_generic_v1.py` can score it
against the frozen reference on the seven splits.

No new features, no frozen-model edit: the change is in the objective, the model
structure or the fusion rule.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve')
PROV = ROOT / 'research/provenance_next_20260917'
sys.path.insert(0, str(PROV))
from provenance_channel_v1 import surfaces  # noqa: E402

SURFACES = ('inner', 's42', 's2026', 's31415', 'x7', 'x77', 'x777')
TASKS = ('MiDrug', 'LncDrug')


def load_bank(task, sname, bank='context76'):
    s = next(x for x in surfaces(task) if x['name'] == sname)
    with np.load(s['bank'](bank), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(s['folds'], allow_pickle=False) as z:
        d['fold_ids'] = z['fold_ids']
    return s, d


def save(dest_dir, task, sname, store):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest_dir / ('%s_%s.npz' % (task, sname)), **store)


def reference_scores(surface):
    """[(key, y, pairs, score), ...] for the frozen tiered_0p5000 reference."""
    out = []
    for seed, path, key, ykey, pkey in surface['refs']:
        with np.load(path, allow_pickle=False) as z:
            out.append((seed, z[ykey], z[pkey], z[key]))
    return out
