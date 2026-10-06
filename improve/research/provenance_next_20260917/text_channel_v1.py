"""Direction D, step 2: language-model semantic-context matching features + member fits.

For a query (r, d) and a set of TRAINING-POSITIVE records (cross-fitted with the frozen
fold ids exactly as in provenance_channel_v1): let E_r be the embeddings of r's records and
E_d those of d's records (each entity capped at 400 records, fixed seed).  Features per view
(ctx / full / title):
    cos_mean : cosine(mean(E_r), mean(E_d))           profile similarity
    cos_max  : max_{i,j} cos(E_r[i], E_d[j])           closest study-context pair
    cos_top3 : mean of the 3 largest pairwise cosines
Missing entity -> 0 for all three.  Control: embeddings permuted across records of the task
(fixed seed), which preserves every marginal but destroys the entity<->text map.
Recipes fitted with the accepted members (hgb/et/mlp/lr) on context76:
    bank+text_ctx, bank+text_full, bank+text_ctx_perm,
    bank+prov+text_ctx, bank+prov+text_full, bank+prov+text_ctx_perm
(`prov` = the real provenance columns cached in cells/<task>_<surface>_context76.npz).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance_channel_v1 import surfaces, member_logits, RECORDS, CONTROL_SEED

HERE = Path(__file__).resolve().parent
VIEWS = ('ctx', 'full', 'title')
CAP = 400
NAMES = tuple('text_%s_%s' % (v, s) for v in VIEWS for s in ('cos_mean', 'cos_max', 'cos_top3'))


class TextIndex:
    def __init__(self, task, permute):
        rec = pd.read_csv(RECORDS)
        mask = (rec.task == task).values
        self.ri, self.di = rec.ri.values[mask].astype(int), rec.di.values[mask].astype(int)
        self.emb = {}
        rng = np.random.default_rng(CONTROL_SEED + 7)
        perm = rng.permutation(mask.sum())
        for v in VIEWS:
            e = np.load(HERE / ('text_%s.npy' % v)).astype(np.float32)[mask]
            self.emb[v] = e[perm] if permute else e
        self.pairs = np.array(list(zip(self.ri, self.di)))

    def encoder(self, support):
        keep = np.array([(int(a), int(b)) in support for a, b in zip(self.ri, self.di)])
        return TextEncoder(self, keep)


class TextEncoder:
    def __init__(self, index, keep):
        rng = np.random.default_rng(CONTROL_SEED)
        idx = np.flatnonzero(keep)
        self.by_r, self.by_d = {}, {}
        for i in idx:
            self.by_r.setdefault(int(index.ri[i]), []).append(i); self.by_d.setdefault(int(index.di[i]), []).append(i)
        for dct in (self.by_r, self.by_d):
            for k, v in dct.items():
                if len(v) > CAP:
                    dct[k] = list(rng.choice(v, CAP, replace=False))
        self.index = index
        self.mean = {v: {} for v in VIEWS}
        for v in VIEWS:
            e = index.emb[v]
            for side, dct in (('r', self.by_r), ('d', self.by_d)):
                for k, rows in dct.items():
                    m = e[rows].mean(0); self.mean[v][(side, k)] = m / max(np.linalg.norm(m), 1e-9)

    def features(self, pairs):
        out = np.zeros((pairs.shape[1], len(NAMES)), dtype=np.float32)
        for n, (r, d) in enumerate(zip(pairs[0], pairs[1])):
            rr, dd = self.by_r.get(int(r)), self.by_d.get(int(d))
            if not rr or not dd:
                continue
            col = 0
            for v in VIEWS:
                e = self.index.emb[v]
                s = e[rr] @ e[dd].T
                flat = np.sort(s.ravel())[::-1]
                out[n, col:col + 3] = [float(self.mean[v][('r', int(r))] @ self.mean[v][('d', int(d))]), float(flat[0]), float(flat[:3].mean())]
                col += 3
        return out


def channel_arrays(index, ptr, ytr, fold_ids, pq):
    pos = ptr[:, ytr == 1]
    all_support = set(zip(pos[0].tolist(), pos[1].tolist()))
    x_train = np.zeros((ptr.shape[1], len(NAMES)), dtype=np.float32)
    for k in np.unique(fold_ids):
        q = np.flatnonzero(fold_ids == k)
        fold_pos = set(zip(ptr[0, q][ytr[q] == 1].tolist(), ptr[1, q][ytr[q] == 1].tolist()))
        x_train[q] = index.encoder(all_support - fold_pos).features(ptr[:, q])
    return x_train, index.encoder(all_support).features(pq)


if __name__ == '__main__':
    task, sname = sys.argv[1], sys.argv[2]
    out_dir = HERE / 'cells_text'; out_dir.mkdir(exist_ok=True)
    dest = out_dir / ('%s_%s.npz' % (task, sname))
    if dest.exists():
        sys.exit(0)
    s = next(x for x in surfaces(task) if x['name'] == sname)
    with np.load(s['bank']('context76'), allow_pickle=False) as z:
        x_tr, y_tr, x_q, ptr, pq = z['x_train'], z['y_train'], z['x_val'], z['pairs_train'], z['pairs_val']
    with np.load(s['folds'], allow_pickle=False) as z:
        fold_ids = z['fold_ids']
    with np.load(HERE / ('cells/%s_%s_context76.npz' % (task, sname)), allow_pickle=False) as z:
        prov_tr, prov_q = z['feat_train_real'], z['feat_val_real']
        assert np.array_equal(z['pairs_val'], pq)
    store = dict(pairs_val=pq)
    feats = {}
    for variant, permute in (('real', False), ('perm', True)):
        feats[variant] = channel_arrays(TextIndex(task, permute), ptr, y_tr, fold_ids, pq)
        store['feat_train_' + variant], store['feat_val_' + variant] = feats[variant]
        print('features', task, sname, variant, flush=True)
    ctx_cols, full_cols = [0, 1, 2], [3, 4, 5]
    recipes = {
        'bank+text_ctx': ('real', ctx_cols, False), 'bank+text_full': ('real', full_cols, False), 'bank+text_ctx_perm': ('perm', ctx_cols, False),
        'bank+prov+text_ctx': ('real', ctx_cols, True), 'bank+prov+text_full': ('real', full_cols, True), 'bank+prov+text_ctx_perm': ('perm', ctx_cols, True),
        'bank+prov+text_all': ('real', list(range(9)), True),
    }
    for label, (variant, cols, with_prov) in recipes.items():
        ft, fq = feats[variant]
        xt = np.c_[x_tr, prov_tr, ft[:, cols]] if with_prov else np.c_[x_tr, ft[:, cols]]
        xq = np.c_[x_q, prov_q, fq[:, cols]] if with_prov else np.c_[x_q, fq[:, cols]]
        for name, v in member_logits(xt, y_tr, xq).items():
            store['logit_%s_%s' % (label, name)] = v
        print('fitted', task, sname, label, flush=True)
    np.savez_compressed(dest, **store)
