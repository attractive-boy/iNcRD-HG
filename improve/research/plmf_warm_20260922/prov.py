# -*- coding: utf-8 -*-
"""Curation-provenance (study hyperedge) channel for PLMF-MDA warm 5cv.
Each ncRNADrug record carries a PMID; a PMID is a hyperedge over the (miRNA, drug) pairs it reports.
For query (m,d) and SUPPORT positives S (never containing the query):
  P_m = multiset of PMIDs of support pairs (m, d'), P_d = PMIDs of support pairs (m', d)
  prov_bridge = |P_m ∩ P_d| (number of training studies that report both m and d through other pairs)
  size-normalised: sum_p 1/size(p);  Jaccard-style normalisation; per-entity #PMIDs.
Control: permute the PMID lists across pairs (keeps per-pair counts)."""
import os, sys, argparse, pickle, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset
sys.path.insert(0, HERE)
from mprior import MPrior, build_extra


class Prov:
    """groups: pmid (study hyperedge bridge), costudy (co-studied-entity label transfer),
    ctx (IDF-weighted experimental-context token overlap), rec (per-entity record statistics)."""
    def __init__(self, ds, shuffle=False, rng=None, groups=('pmid', 'costudy', 'ctx', 'rec')):
        mir, dru, pos, neg = load_dataset(ds)
        self.nr, self.nd = len(mir), len(dru); self.groups = set(groups)
        names = mir['miRNA_Name'].astype(str).str.upper().values; dbs = dru['DrugBank_ID'].astype(str).values
        recs = pickle.load(open(os.path.join(HERE, 'data', 'pair_records.pkl'), 'rb'))
        pmids = sorted(set(p for v in recs.values() for p in v['pmids'])); self.pid = {p: i for i, p in enumerate(pmids)}
        toks = sorted(set(t for v in recs.values() for t in v['toks'])); self.tid = {t: i for i, t in enumerate(toks)}
        self.npm, self.ntok = len(pmids), len(toks)
        df = np.zeros(self.ntok)
        for v in recs.values():
            for t in v['toks']: df[self.tid[t]] += 1
        self.idf = np.log((len(recs) + 1) / (df + 1)) + 1
        self.pair_rec = {}
        for i, j in pos:
            k = (names[i], dbs[j])
            if k in recs:
                v = recs[k]
                self.pair_rec[(i, j)] = dict(pm=[self.pid[p] for p in v['pmids']], tk=[self.tid[t] for t in v['toks']],
                                             years=v['years'], ht=v['ht'], nrec=v['nrec'])
        if shuffle:
            keys = list(self.pair_rec); vals = [self.pair_rec[k] for k in keys]
            perm = (rng or np.random.RandomState(0)).permutation(len(keys))
            self.pair_rec = {keys[i]: vals[perm[i]] for i in range(len(keys))}

    def _mats(self, sup):
        rr, rd, cp, tr_, td, ct = [], [], [], [], [], []
        stat_r = np.zeros((self.nr, 5)); stat_d = np.zeros((self.nd, 5))   # nrec, ht, ysum, ymax, ycount
        for i, j in sup:
            v = self.pair_rec.get((i, j))
            if v is None: continue
            for p in v['pm']: rr.append(i); rd.append(j); cp.append(p)
            for t in v['tk']: tr_.append(i); td.append(j); ct.append(t)
            for s, idx in ((stat_r, i), (stat_d, j)):
                s[idx, 0] += v['nrec']; s[idx, 1] += v['ht']
                if v['years']:
                    s[idx, 2] += sum(v['years']); s[idx, 3] = max(s[idx, 3], max(v['years'])); s[idx, 4] += len(v['years'])
        def binm(rows, cols, n, m):
            M = sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(n, m)); M.sum_duplicates(); M.data[:] = 1; return M
        return binm(rr, cp, self.nr, self.npm), binm(rd, cp, self.nd, self.npm), binm(tr_, ct, self.nr, self.ntok), binm(td, ct, self.nd, self.ntok), stat_r, stat_d

    def transform(self, sup, q):
        sup = np.asarray(sup); q = np.asarray(q); r, d = q[:, 0], q[:, 1]
        Mr, Md, Tr, Td, stat_r, stat_d = self._mats(sup)
        A = sp.csr_matrix((np.ones(len(sup)), (sup[:, 0], sup[:, 1])), shape=(self.nr, self.nd))
        cols = {}
        if 'pmid' in self.groups:
            size = np.asarray(Mr.sum(0)).ravel() + np.asarray(Md.sum(0)).ravel()
            Qr, Qd = Mr[r], Md[d]; inter = Qr.multiply(Qd)
            bridge = np.asarray(inter.sum(1)).ravel()
            nr_ = np.asarray(Qr.sum(1)).ravel(); nd_ = np.asarray(Qd.sum(1)).ravel()
            cols.update(prov_bridge=bridge, prov_bridge_w=np.asarray(inter.multiply(1.0 / np.maximum(size, 1)[None, :]).sum(1)).ravel(),
                        prov_jacc=bridge / np.maximum(nr_ + nd_ - bridge, 1), prov_npm_r=nr_, prov_npm_d=nd_, prov_has_r=(nr_ > 0).astype(float))
        if 'costudy' in self.groups:
            Sr = (Mr @ Mr.T).tolil(); Sr.setdiag(0); Sr = Sr.tocsr()      # miRNA x miRNA shared studies
            Sd = (Md @ Md.T).tolil(); Sd.setdiag(0); Sd = Sd.tocsr()
            SrA = (Sr @ A).toarray(); ASd = (A @ Sd).toarray()
            srs = np.asarray(Sr.sum(1)).ravel(); sds = np.asarray(Sd.sum(1)).ravel()
            cols.update(cost_r=SrA[r, d], cost_r_norm=SrA[r, d] / np.maximum(srs[r], 1), cost_r_deg=srs[r],
                        cost_d=ASd[r, d], cost_d_norm=ASd[r, d] / np.maximum(sds[d], 1), cost_d_deg=sds[d])
        if 'ctx' in self.groups:
            Qr, Qd = Tr[r], Td[d]; inter = Qr.multiply(Qd)
            ov = np.asarray(inter.multiply(self.idf[None, :]).sum(1)).ravel()
            wr = np.sqrt(np.asarray(Qr.multiply(self.idf[None, :] ** 2).sum(1)).ravel()); wd = np.sqrt(np.asarray(Qd.multiply(self.idf[None, :] ** 2).sum(1)).ravel())
            cols.update(ctx_overlap=ov, ctx_cos=ov / np.maximum(wr * wd, 1e-8), ctx_ntok_r=np.asarray(Qr.sum(1)).ravel(), ctx_ntok_d=np.asarray(Qd.sum(1)).ravel())
        if 'rec' in self.groups:
            for s, idx, tag in ((stat_r, r, 'r'), (stat_d, d, 'd')):
                n = s[idx, 0]; yc = np.maximum(s[idx, 4], 1)
                cols[f'rec_n_{tag}'] = n; cols[f'rec_ht_{tag}'] = s[idx, 1] / np.maximum(n, 1)
                cols[f'rec_ymean_{tag}'] = np.where(s[idx, 4] > 0, s[idx, 2] / yc - 2000, 0); cols[f'rec_ymax_{tag}'] = np.where(s[idx, 4] > 0, s[idx, 3] - 2000, 0)
        return pd.DataFrame(cols, dtype=np.float32)


def crossfit(obj, tr, ytr, te, seed, fn, K=5):
    Xtr = None
    for fit, qy in StratifiedKFold(K, shuffle=True, random_state=seed).split(tr, ytr.astype(int)):
        sup = tr[fit[ytr[fit] == 1]]
        f = fn(obj, sup, tr[qy])
        if Xtr is None: Xtr = np.zeros((len(ytr), f.shape[1]), np.float32)
        Xtr[qy] = f.values
    Xte = fn(obj, tr[ytr == 1], te).values
    both = np.concatenate([Xtr, Xte], 0); both = np.log1p(np.abs(both)) * np.sign(both)
    Xtr, Xte = both[:len(ytr)], both[len(ytr):]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--split', default='5cv')
    ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--only', action='store_true')
    ap.add_argument('--mprior', default='', help='also append miRNA prior fams, e.g. fam,seed,hmdd,xdeg')
    ap.add_argument('--C', type=float, default=0.3); ap.add_argument('--tag', default=''); ap.add_argument('--save', action='store_true')
    ap.add_argument('--groups', default='pmid,costudy,ctx,rec'); ap.add_argument('--base', default='ctx+diff')
    a = ap.parse_args()
    pv = Prov(a.dataset, a.shuffle, np.random.RandomState(a.seed), groups=a.groups.split(','))
    mp = MPrior(a.dataset) if a.mprior else None
    res, ll, rows = [], [], []
    for f in (range(5) if a.split == '5cv' else [0]):
        z = np.load(os.path.join(HERE, 'cache', f'{a.dataset}_{"single" if a.split == "single" else ""}s{a.seed}_f{f}_{a.base}_kernel10.npz'))
        Xtr, Xte, tr, ytr, te, yte = z['Xtr'], z['Xte'], z['tr'], z['ytr'], z['te'], z['yte']
        Ptr, Pte = crossfit(pv, tr, ytr, te, a.seed * 100 + f, lambda o, s, q: o.transform(s, q))
        if a.only: Xtr, Xte = Ptr, Pte
        else: Xtr = np.concatenate([Xtr, Ptr], 1); Xte = np.concatenate([Xte, Pte], 1)
        if mp is not None:
            Etr, Ete, _ = build_extra(mp, tr, ytr, te, set(a.mprior.split(',')), a.seed * 100 + f)
            Xtr = np.concatenate([Xtr, Etr], 1); Xte = np.concatenate([Xte, Ete], 1)
        m = LogisticRegression(max_iter=5000, C=a.C).fit(Xtr, ytr.astype(int)); s = m.predict_proba(Xte)[:, 1]
        res.append((roc_auc_score(yte, s), average_precision_score(yte, s)))
        if a.save: rows.append(pd.DataFrame({'fold': f, 'r': te[:, 0], 'd': te[:, 1], 'y': yte.astype(int), 'score': s}))
        sup = tr[ytr == 1]; dr = np.bincount(sup[:, 0], minlength=pv.nr)
        negs = np.sort(s[yte == 0]); above = 1 - np.searchsorted(negs, s[yte == 1], side='left') / len(negs)
        lo = dr[te[yte == 1, 0]] <= 1; ll.append((above[lo].mean(), above[~lo].mean()))
    res = np.array(res); ll = np.array(ll)
    tag = a.tag or (('only:' if a.only else a.base + '+') + 'prov[' + a.groups + ']' + (f'+{a.mprior}' if a.mprior else '') + (' shuf' if a.shuffle else ''))
    line = (f'## {a.dataset} {"single" if a.split == "single" else ""}s{a.seed} C{a.C} {tag}: AUC {res[:,0].mean():.4f}±{res[:,0].std():.4f} AUPR {res[:,1].mean():.4f}±{res[:,1].std():.4f} '
            f'| loss(deg_r<=1) {ll[:,0].mean():.4f} loss(deg_r>1) {ll[:,1].mean():.4f} ncol {Xtr.shape[1]}')
    print(line, flush=True)
    with open(os.path.join(HERE, 'results', 'prov.txt'), 'a') as fh: fh.write(line + '\n')
    if a.save:
        safe = tag.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '.').replace('+', '_')
        pd.concat(rows).to_csv(os.path.join(HERE, 'results', f'scores_{a.dataset}_{"single" if a.split == "single" else ""}s{a.seed}_{safe}.csv'), index=False)


if __name__ == '__main__':
    main()
