"""Scheme D (fallback): ensemble aggregation operators.

The accepted recipe averages the four member logits with an arithmetic mean.  This
scheme swaps in different, label-free aggregation algorithms on the *same* four
members: median (robust to one member's outlier scores), trimmed mean, probability
mean, and a per-query rank average (Borda-style, scale-free).  Nothing is trained and
no label is used; the change is the aggregation operator itself.

Writes `logit_<name>_member` next to the cached member logits so the standard gate
evaluator can score them.

usage: algo_aggregate_v1.py <task> <surface> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917')
from algo_common import save  # noqa: E402

MEMBERS = ('hgb', 'et', 'mlp', 'lr')
CACHE = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917/cells')


def rankify(a):
    from scipy.stats import rankdata
    return rankdata(a) / (len(a) + 1.0)


def main(task, sname, out_dir):
    src = CACHE / ('%s_%s_context76.npz' % (task, sname))
    with np.load(src, allow_pickle=False) as z:
        L = np.stack([z['logit_bank_%s' % m] for m in MEMBERS], 1).astype(np.float64)
        pq = z['pairs_val']
    store = dict(pairs_val=pq)
    store['logit_aggmean_member'] = L.mean(1).astype(np.float32)
    store['logit_aggmedian_member'] = np.median(L, 1).astype(np.float32)
    srt = np.sort(L, 1)
    store['logit_aggtrim_member'] = srt[:, 1:-1].mean(1).astype(np.float32)
    P = 1.0 / (1.0 + np.exp(-L))
    store['logit_aggprob_member'] = (np.log(P.mean(1) + 1e-12) - np.log1p(-P.mean(1) - 1e-12)).astype(np.float32)
    store['logit_aggrank_member'] = np.stack([rankify(L[:, j]) for j in range(L.shape[1])], 1).mean(1).astype(np.float32)
    save(out_dir, task, sname, store)
    print('aggregated', task, sname, flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
