"""Column-matched controls for the text channel: fit the missing labels into the existing cells_text npz."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance_channel_v1 import surfaces, member_logits

HERE = Path(__file__).resolve().parent
EXTRA = {
    'bank+prov+text_all_perm': ('perm', list(range(9)), True),
    'bank+prov+text_full_perm': ('perm', [3, 4, 5], True),
    'bank+text_all': ('real', list(range(9)), False),
    'bank+text_all_perm': ('perm', list(range(9)), False),
}
if __name__ == '__main__':
    task, sname = sys.argv[1], sys.argv[2]
    dest = HERE / ('cells_text/%s_%s.npz' % (task, sname))
    store = dict(np.load(dest, allow_pickle=False))
    if all(('logit_%s_hgb' % k) in store for k in EXTRA):
        sys.exit(0)
    s = next(x for x in surfaces(task) if x['name'] == sname)
    with np.load(s['bank']('context76'), allow_pickle=False) as z:
        x_tr, y_tr, x_q, pq = z['x_train'], z['y_train'], z['x_val'], z['pairs_val']
    assert np.array_equal(pq, store['pairs_val'])
    with np.load(HERE / ('cells/%s_%s_context76.npz' % (task, sname)), allow_pickle=False) as z:
        prov_tr, prov_q = z['feat_train_real'], z['feat_val_real']
    for label, (variant, cols, with_prov) in EXTRA.items():
        if ('logit_%s_hgb' % label) in store:
            continue
        ft, fq = store['feat_train_' + variant], store['feat_val_' + variant]
        xt = np.c_[x_tr, prov_tr, ft[:, cols]] if with_prov else np.c_[x_tr, ft[:, cols]]
        xq = np.c_[x_q, prov_q, fq[:, cols]] if with_prov else np.c_[x_q, fq[:, cols]]
        for name, v in member_logits(xt, y_tr, xq).items():
            store['logit_%s_%s' % (label, name)] = v
        print('fitted', task, sname, label, flush=True)
    tmp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **store)
    tmp.replace(dest)
