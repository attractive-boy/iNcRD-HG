"""Direction C: evidence-strength-weighted training objective (no new features).

Training positives are re-weighted by their own curation support (number of distinct
PMIDs, presence of low-throughput evidence); sampled negatives keep weight 1.  Members
hgb / ExtraTrees / LogisticRegression (the accepted members that accept sample_weight)
are fitted on the accepted context76 bank and fused into the reference at weight w.
The control is the identical member set with uniform weights.  Schemes:
    uniform   : w=1                                   (control)
    pmid_log  : w = 1 + log2(n_pmid)                  (1 PMID -> 1, 2 -> 2, 4 -> 3)
    lowthr    : w = 1 + [has low-throughput record]
    combined  : w = (1 + log2(n_pmid)) * (1 + 0.5*[low-throughput])
    inverse   : w = 1 / (1 + log2(n_pmid))            (direction check)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from improve.context_anchor_verify import learner
from provenance_channel_v1 import surfaces, logit, RECORDS

HERE = Path(__file__).resolve().parent
MEMBERS = ('hgb', 'et', 'lr')
SCHEMES = ('uniform', 'pmid_log', 'lowthr', 'combined', 'inverse')


def pair_evidence(task):
    rec = pd.read_csv(RECORDS)
    rec = rec[rec.task == task]
    g = rec.groupby(['ri', 'di']).agg(n_pmid=('PMID', 'nunique'), n_low=('ht', lambda s: int((s == 0).sum())))
    return {(int(a), int(b)): (int(c), int(d)) for (a, b), (c, d) in zip(g.index, g.values)}


def weights(scheme, pairs, y, ev):
    w = np.ones(len(y))
    for i in np.flatnonzero(y == 1):
        n_pmid, n_low = ev.get((int(pairs[0, i]), int(pairs[1, i])), (1, 0))
        n_pmid = max(n_pmid, 1)
        if scheme == 'pmid_log':
            w[i] = 1 + np.log2(n_pmid)
        elif scheme == 'lowthr':
            w[i] = 1 + (n_low > 0)
        elif scheme == 'combined':
            w[i] = (1 + np.log2(n_pmid)) * (1 + 0.5 * (n_low > 0))
        elif scheme == 'inverse':
            w[i] = 1 / (1 + np.log2(n_pmid))
    return w


def fit_members(x_tr, y_tr, x_q, w):
    out = {}
    for name in MEMBERS:
        m = learner(name)
        if name == 'lr':
            m.fit(x_tr, y_tr, logisticregression__sample_weight=w)
        else:
            m.fit(x_tr, y_tr, sample_weight=w)
        out[name] = logit(m.predict_proba(x_q)[:, 1])
    return out


if __name__ == '__main__':
    task, sname = sys.argv[1], sys.argv[2]
    out_dir = HERE / 'cells_weight'
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / ('%s_%s.npz' % (task, sname))
    if dest.exists():
        sys.exit(0)
    s = next(x for x in surfaces(task) if x['name'] == sname)
    with np.load(s['bank']('context76'), allow_pickle=False) as z:
        x_tr, y_tr, x_q, ptr, pq = z['x_train'], z['y_train'], z['x_val'], z['pairs_train'], z['pairs_val']
    ev = pair_evidence(task)
    store = dict(pairs_val=pq)
    for scheme in SCHEMES:
        w = weights(scheme, ptr, y_tr, ev)
        for name, v in fit_members(x_tr, y_tr, x_q, w).items():
            store['logit_%s_%s' % (scheme, name)] = v
        store['w_%s' % scheme] = w
        print('fitted', task, sname, scheme, 'mean pos weight %.3f' % w[y_tr == 1].mean(), flush=True)
    np.savez_compressed(dest, **store)
