# -*- coding: utf-8 -*-
"""miRNA-side prior bank for PLMF-MDA warm 5cv.
All columns are label-free w.r.t. the query: they use (i) miRNA side information that is
independent of the benchmark labels (name family, seed family, HMDD disease profile,
other-phenotype matrix, PubMed counts) and (ii) the SUPPORT positives only.
Evaluated by appending to the cached ctx+diff features and refitting LR (cross-fitted,
same inner folds as run_probe.build_xy)."""
import os, sys, re, argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset, get_split
HMDD = '/data1/licheng/incrd-hg/data_external/ncrna_disease_20260917/hmdd_v4_alldata.txt'
PUB = os.path.join(HERE, 'data', 'pubmed_mirna_plmf.csv')


def stem(n):
    s = n.lower().strip()
    s = re.sub(r'-(3p|5p)$', '', s)
    return s

def family(n):
    s = stem(n).replace('hsa-', '')
    s = re.sub(r'-[12]$', '', s)            # miR-199a-1 -> miR-199a
    s = re.sub(r'([0-9])[a-z]$', r'\1', s)  # miR-200a -> miR-200
    return s

def seed7(seq):
    return str(seq).upper().replace('T', 'U')[1:8]


class MPrior:
    def __init__(self, ds, rng=None, shuffle=None):
        self.ds = ds
        mir, dru, pos, neg = load_dataset(ds)
        other = 'MDS' if ds == 'MDR' else 'MDR'
        omir, odru, opos, oneg = load_dataset(other)
        self.names = mir['miRNA_Name'].astype(str).values
        self.seqs = mir['miRNA_Sequence'].astype(str).values
        self.nr, self.nd = len(mir), len(dru)
        self.dbs = dru['DrugBank_ID'].astype(str).values
        # --- groups: name family, seed family
        self.fam = pd.factorize(pd.Series([family(n) for n in self.names]))[0]
        self.seed = pd.factorize(pd.Series([seed7(s) for s in self.seqs]))[0]
        # --- HMDD disease profile (miRNA stem x disease)
        h = pd.read_csv(HMDD, sep='\t', encoding='latin1', on_bad_lines='skip')
        h['m'] = h['miRNA'].astype(str).str.lower().str.strip(); h['dz'] = h['disease'].astype(str).str.strip()
        stems = np.array([stem(n) for n in self.names])
        dz_idx = {d: i for i, d in enumerate(sorted(h['dz'].unique()))}
        H = np.zeros((self.nr, len(dz_idx)), np.float32)
        st_idx = {}
        for i, s in enumerate(stems): st_idx.setdefault(s, []).append(i)
        for m, dz in zip(h['m'], h['dz']):
            for i in st_idx.get(m, []): H[i, dz_idx[dz]] = 1
        self.H = H
        # --- other-phenotype degree (by name) and other matrix (aligned by name / DrugBank id)
        oname = {n: i for i, n in enumerate(omir['miRNA_Name'].astype(str))}
        odeg = np.bincount(opos[:, 0], minlength=len(omir))
        self.xdeg = np.array([odeg[oname[n]] if n in oname else -1 for n in self.names], np.float32)
        # --- PubMed counts
        if os.path.exists(PUB):
            p = pd.read_csv(PUB, on_bad_lines='skip'); p = p[~p['name'].duplicated()].set_index('name')
            self.pub = np.array([[float(p.loc[n, c]) if n in p.index else -1.0 for c in ('n_all', 'n_drugresp', 'n_cancer')] for n in self.names], np.float32)
            self.pub_cov = float(np.mean(self.pub[:, 0] >= 0))
        else:
            self.pub = None
        # --- shuffle control: permute side info across miRNAs
        if shuffle:
            perm = (rng or np.random.RandomState(0)).permutation(self.nr)
            if 'fam' in shuffle: self.fam = self.fam[perm]
            if 'seed' in shuffle: self.seed = self.seed[perm]
            if 'hmdd' in shuffle: self.H = self.H[perm]
            if 'xdeg' in shuffle: self.xdeg = self.xdeg[perm]
            if 'pub' in shuffle and self.pub is not None: self.pub = self.pub[perm]
        # similarity from HMDD profile (cosine), top-k
        Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-8)
        S = Hn @ Hn.T; np.fill_diagonal(S, 0)
        self.S_h = S
        k = 20
        thr = np.sort(S, axis=1)[:, -k][:, None]
        self.S_hk = np.where(S >= thr, S, 0)

    def group_cols(self, g, A, r, d, tag):
        G = np.zeros((g.max() + 1, self.nr), np.float32); G[g, np.arange(self.nr)] = 1
        size = G.sum(1)[g] - 1                      # family mates (excluding self)
        GA = G @ A                                  # group x drug link counts
        fsum = GA[g[r], d] - A[r, d]                # exclude the query miRNA's own support link (which is 0 for queries anyway)
        fdeg = (G @ A.sum(1))[g] - A.sum(1)         # mates' total degree
        return {f'{tag}_size': size[r], f'{tag}_sum': fsum, f'{tag}_mean': fsum / np.maximum(size[r], 1),
                f'{tag}_deg': fdeg[r], f'{tag}_frac': fsum / np.maximum(fdeg[r], 1)}

    def transform(self, sup, q, fams):
        sup = np.asarray(sup); q = np.asarray(q)
        A = np.zeros((self.nr, self.nd), np.float32); A[sup[:, 0], sup[:, 1]] = 1
        r, d = q[:, 0], q[:, 1]
        dr = A.sum(1); dd = A.sum(0)
        cols = {}
        if 'fam' in fams: cols.update(self.group_cols(self.fam, A, r, d, 'fam'))
        if 'seed' in fams: cols.update(self.group_cols(self.seed, A, r, d, 'seed'))
        if 'hmdd' in fams:
            H = self.H
            ndz = H.sum(1)
            cols['hmdd_ndz'] = ndz[r]
            cols['hmdd_has'] = (ndz[r] > 0).astype(np.float32)
            # disease bridge: sum over diseases of m of (#support links of drug d from miRNAs with that disease)
            DA = H.T @ A                                    # disease x drug
            bridge = (H[r] * DA[:, d].T).sum(1)             # sum_dz H[m,dz] * DA[dz,d]
            cols['hmdd_bridge'] = bridge
            cols['hmdd_bridge_norm'] = bridge / np.maximum(ndz[r] * dd[d], 1)
            # kNN label transfer via HMDD similarity
            SA = self.S_hk @ A
            cols['hmdd_knn'] = SA[r, d]
            cols['hmdd_knn_mean'] = SA[r, d] / np.maximum(self.S_hk.sum(1)[r], 1e-8)
            cols['hmdd_sim_deg'] = (self.S_hk @ dr)[r]        # popularity of HMDD-neighbours
        if 'xdeg' in fams:
            cols['xdeg'] = self.xdeg[r]
            cols['xdeg_has'] = (self.xdeg[r] >= 0).astype(np.float32)
        if 'pub' in fams and self.pub is not None:
            for j, c in enumerate(('pub_all', 'pub_drugresp', 'pub_cancer')):
                cols[c] = np.log1p(np.maximum(self.pub[r, j], 0))
        return pd.DataFrame(cols, dtype=np.float32)


def build_extra(mp, tr, ytr, te, fams, seed, K=5):
    Xtr = None
    for fit, qy in StratifiedKFold(K, shuffle=True, random_state=seed).split(tr, ytr.astype(int)):
        sup = tr[fit[ytr[fit] == 1]]
        f = mp.transform(sup, tr[qy], fams)
        if Xtr is None: Xtr = np.zeros((len(ytr), f.shape[1]), np.float32); names = list(f.columns)
        Xtr[qy] = f.values
    Xte = mp.transform(tr[ytr == 1], te, fams).values
    both = np.concatenate([Xtr, Xte], 0)
    both = np.log1p(np.abs(both)) * np.sign(both)
    Xtr, Xte = both[:len(ytr)], both[len(ytr):]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--split', default='5cv')
    ap.add_argument('--fams', default='fam,seed,hmdd,xdeg'); ap.add_argument('--shuffle', default='')
    ap.add_argument('--base', default='ctx+diff'); ap.add_argument('--C', type=float, default=0.3)
    ap.add_argument('--only', action='store_true', help='evaluate the prior columns alone')
    ap.add_argument('--tag', default=''); ap.add_argument('--save', action='store_true')
    ap.add_argument('--gate', action='store_true', help='append prior columns gated by sparse-miRNA indicators')
    a = ap.parse_args()
    fams = set(a.fams.split(',')) if a.fams else set()
    mp = MPrior(a.dataset, rng=np.random.RandomState(a.seed), shuffle=set(a.shuffle.split(',')) if a.shuffle else None)
    res = []; loss_lo = []; rows = []
    for f in (range(5) if a.split == '5cv' else [0]):
        z = np.load(os.path.join(HERE, 'cache', f'{a.dataset}_{"single" if a.split == "single" else ""}s{a.seed}_f{f}_{a.base}_kernel10.npz'))
        Xtr, Xte, tr, ytr, te, yte = z['Xtr'], z['Xte'], z['tr'], z['ytr'], z['te'], z['yte']
        if fams:
            Etr, Ete, names = build_extra(mp, tr, ytr, te, fams, a.seed * 100 + f)
            if a.gate:
                def gate(E, pairs, sup):
                    dr = np.bincount(sup[:, 0], minlength=mp.nr)[pairs[:, 0]]
                    g0 = (dr == 0).astype(np.float32)[:, None]; g1 = (dr <= 1).astype(np.float32)[:, None]
                    return np.concatenate([E, E * g0, E * g1, g0, g1], 1)
                # training rows: their support degree is the inner-fold support; approximate with the full-train degree minus own link
                dr_full = np.bincount(tr[ytr == 1][:, 0], minlength=mp.nr)
                drt = dr_full[tr[:, 0]] - ytr.astype(int)
                g0 = (drt == 0).astype(np.float32)[:, None]; g1 = (drt <= 1).astype(np.float32)[:, None]
                Etr = np.concatenate([Etr, Etr * g0, Etr * g1, g0, g1], 1)
                Ete = gate(Ete, te, tr[ytr == 1])
            if a.only: Xtr, Xte = Etr, Ete
            else: Xtr = np.concatenate([Xtr, Etr], 1); Xte = np.concatenate([Xte, Ete], 1)
        m = LogisticRegression(max_iter=5000, C=a.C).fit(Xtr, ytr.astype(int))
        s = m.predict_proba(Xte)[:, 1]
        auc, ap_ = roc_auc_score(yte, s), average_precision_score(yte, s)
        if a.save: rows.append(pd.DataFrame({'fold': f, 'r': te[:, 0], 'd': te[:, 1], 'y': yte.astype(int), 'score': s}))
        # loss on sparse-miRNA positives (support degree <=1)
        sup = tr[ytr == 1]; dr = np.bincount(sup[:, 0], minlength=mp.nr)
        negs = np.sort(s[yte == 0]); pos = s[yte == 1]; above = 1 - np.searchsorted(negs, pos, side='left') / len(negs)
        lo = dr[te[yte == 1, 0]] <= 1
        res.append((auc, ap_)); loss_lo.append((above[lo].mean(), above[~lo].mean()))
    res = np.array(res); ll = np.array(loss_lo)
    tag = a.tag or (('only:' if a.only else a.base + '+') + a.fams + (' gate' if a.gate else '') + (f' shuf[{a.shuffle}]' if a.shuffle else ''))
    line = (f'## {a.dataset} {"single" if a.split == "single" else ""}s{a.seed} C{a.C} {tag}: AUC {res[:,0].mean():.4f}±{res[:,0].std():.4f} AUPR {res[:,1].mean():.4f}±{res[:,1].std():.4f} '
            f'| loss(deg_r<=1) {ll[:,0].mean():.4f} loss(deg_r>1) {ll[:,1].mean():.4f} ncol {Xtr.shape[1]}')
    print(line, flush=True)
    with open(os.path.join(HERE, 'results', 'mprior.txt'), 'a') as fh: fh.write(line + '\n')
    if a.save:
        safe = tag.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '.').replace('+', '_')
        pd.concat(rows).to_csv(os.path.join(HERE, 'results', f'scores_{a.dataset}_{"single" if a.split == "single" else ""}s{a.seed}_{safe}.csv'), index=False)


if __name__ == '__main__':
    main()
