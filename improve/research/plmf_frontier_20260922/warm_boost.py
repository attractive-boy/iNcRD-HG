# -*- coding: utf-8 -*-
"""Warm single-dataset AUC/AUPR boost on PLMF-MDA (MDR/MDS), 5cv.

Adds feature families the plmf_innov bank did not include and tests model ensembles:
  knn : dense content label-transfer (Morgan-Tanimoto, ChemBERTa-cosine, k-mer-cosine),
        multiple k and normalisations -- the family that won the cold-drug probe
  aa  : Adamic-Adar / Resource-Allocation weighted length-3 paths
  mf  : higher-rank SVD products of the support matrix
Learners: lr / hgb / xgb / mlp, and `ens` = rank-average of them.

Reuses run_probe.build_xy / fit_model / load_nodes for an identical protocol.
Run with OMP_NUM_THREADS=8.
"""
import argparse, os, sys, time, csv
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
sys.path.insert(0, os.path.join(BASE, 'dlstmda_strongest_20260920'))
sys.path.insert(0, os.path.join(BASE, 'plmf_innov_20260921'))
from plmf_splits import get_split  # noqa
from bank import Bank  # noqa
import run_probe as RP  # noqa

CB = os.path.join(BASE, 'mphgnn_plmf_20260921', 'cache')


def _std(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)


def _tanimoto(x):
    xs = (x > 0).astype(np.float32); inter = xs @ xs.T; cnt = xs.sum(1, keepdims=True)
    s = inter / (cnt + cnt.T - inter + 1e-8); np.fill_diagonal(s, 0); return s


def _cosine(x):
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8); s = xn @ xn.T
    np.fill_diagonal(s, 0); return np.clip(s, 0, None)


class BoostBank(Bank):
    def __init__(self, xr, xd, sr, sd, xcb, graph=None):
        super().__init__(xr, xd, sr, sd)
        self.S_d = {'tan': _tanimoto(xd), 'cb': _cosine(xcb)}
        self.S_r = {'km': _cosine(xr)}
        self.xcb = xcb
        self.cache2 = {}
        self.mech = None
        if graph is not None:
            nr, nd = xr.shape[0], xd.shape[0]
            MG = np.zeros((nr, graph['genes'].shape[0]), np.float32); MG[graph['mirna_gene'][:, 0], graph['mirna_gene'][:, 1]] = 1
            GD = np.zeros((graph['genes'].shape[0], nd), np.float32); GD[graph['gene_drug'][:, 0], graph['gene_drug'][:, 1]] = 1
            GG = np.zeros((graph['genes'].shape[0],) * 2, np.float32); GG[graph['gene_gene'][:, 0], graph['gene_gene'][:, 1]] = 1
            GG = np.maximum(GG, GG.T)
            self.mech = dict(B0=MG @ GD, B1=MG @ GG @ GD, mg=MG.sum(1), gd=GD.sum(0))

    def _mech_cols(self, q):
        r, d = q[:, 0], q[:, 1]; B0, B1, mg, gd = (self.mech['B0'], self.mech['B1'], self.mech['mg'], self.mech['gd'])
        den = np.maximum(np.sqrt(mg[r] * gd[d]), 1)
        return {'mech_b0': B0[r, d], 'mech_b1': B1[r, d], 'mech_b0_norm': B0[r, d] / den,
                'mech_b1_norm': B1[r, d] / den, 'mech_cnt_r': mg[r], 'mech_cnt_d': gd[d],
                'mech_has_r': (mg[r] > 0).astype(np.float32), 'mech_has_d': (gd[d] > 0).astype(np.float32)}

    def _cb_cols(self, A, q):
        r, d = q[:, 0], q[:, 1]
        u = self.xcb / np.maximum(np.linalg.norm(self.xcb, axis=1, keepdims=True), 1e-8)
        S = u @ u.T; np.fill_diagonal(S, 0)
        Au = A @ u
        Au_n = Au / np.maximum(np.linalg.norm(Au, axis=1, keepdims=True), 1e-8)
        return {'cb_attr_support': (A @ S)[r, d],
                'cb_attr_profile': (Au_n[r] * u[d]).sum(1),
                'cb_attr_profile_sum': (Au[r] * u[d]).sum(1)}

    def _A(self, sup):
        A = np.zeros((self.nr, self.nd), np.float32); A[sup[:, 0], sup[:, 1]] = 1.0; return A

    def _knn_cols(self, A, q):
        r, d = q[:, 0], q[:, 1]; cols = {}
        for sn, S in (('dtan', self.S_d['tan']), ('dcb', self.S_d['cb']), ('rkm', self.S_r['km'])):
            drug_side = sn.startswith('d')
            for k in (0, 5, 20):
                Sk = S.copy()
                if k:
                    thr = np.sort(Sk, axis=1)[:, -k][:, None]; Sk = np.where(Sk >= thr, Sk, 0)
                if drug_side:            # drug d vs its similar drugs' positives
                    num = A @ Sk.T                     # [nr,nd]
                    den = np.maximum(Sk.sum(1), 1e-8)  # per drug
                    v = num[r, d]; vmean = (num / den[None, :])[r, d]
                    cols[f'knn_{sn}{k}_sum'] = v
                    cols[f'knn_{sn}{k}_mean'] = vmean
                    cols[f'knn_{sn}{k}_rfrac'] = num[r, d] / np.maximum(A.sum(1)[r], 1)
                    cols[f'knn_{sn}{k}_rpc'] = (num[r] < v[:, None]).mean(1)
                else:                    # miRNA r vs its similar miRNAs' positives
                    num = Sk @ A                       # [nr,nd]
                    den = np.maximum(Sk.sum(1), 1e-8)  # per miRNA
                    v = num[r, d]
                    cols[f'knn_{sn}{k}_sum'] = v
                    cols[f'knn_{sn}{k}_mean'] = (num / den[:, None])[r, d]
                    cols[f'knn_{sn}{k}_dfrac'] = num[r, d] / np.maximum(A.sum(0)[d], 1)
                    cols[f'knn_{sn}{k}_rpc'] = (num[:, d].T < v[:, None]).mean(1)
        return cols

    def _aa_cols(self, A, q):
        r, d = q[:, 0], q[:, 1]
        dr = np.maximum(A.sum(1), 1); dd = np.maximum(A.sum(0), 1)
        wd = 1.0 / np.log(1.0 + dd); wr = 1.0 / np.log(1.0 + dr)
        a3 = A @ (A.T @ A)                                  # length-3 paths
        aa_d = (A * wd[None, :]) @ (A.T @ A)                # weight intermediate drug d'
        aa_r = A @ (A.T @ (wr[:, None] * A))                # weight intermediate miRNA r'
        return {'aa_path3': a3[r, d], 'aa_d': aa_d[r, d], 'aa_r': aa_r[r, d],
                'aa_d_norm': aa_d[r, d] / np.maximum(dr[r] * dd[d], 1),
                'aa_r_norm': aa_r[r, d] / np.maximum(dr[r] * dd[d], 1)}

    def _mf_cols(self, A, q):
        r, d = q[:, 0], q[:, 1]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        out = {}
        for K in (32, 64):
            K = min(K, len(s))
            dot = ((U[:, :K] * s[:K]) * Vt[:K].T).sum(1)
            out[f'mf{K}_dot'] = dot
            out[f'mf{K}_rnorm'] = np.linalg.norm(U[r, :K], axis=1)
            out[f'mf{K}_dnorm'] = np.linalg.norm(Vt[:K].T[d], axis=1)
        return out

    def transform(self, sup, q, families):
        base = families & {'ctx', 'diff', 'cdiff', 'scf', 'align'}
        out = super().transform(sup, q, base) if base else pd.DataFrame(index=range(len(q)))
        sup = np.asarray(sup, np.int64); q = np.asarray(q, np.int64)
        if families & {'knn', 'aa', 'mf', 'mech', 'cb'}:
            A = self._A(sup); add = {}
            if 'knn' in families: add.update(self._knn_cols(A, q))
            if 'aa' in families: add.update(self._aa_cols(A, q))
            if 'mf' in families: add.update(self._mf_cols(A, q))
            if 'cb' in families: add.update(self._cb_cols(A, q))
            if 'mech' in families and self.mech is not None: add.update(self._mech_cols(q))
            out = pd.concat([out, pd.DataFrame(add)], axis=1)
        return out.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--split', default='5cv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--families', default='ctx,diff,knn,aa')
    ap.add_argument('--models', default='lr:0.3,ens')
    ap.add_argument('--graph', default='kernel'); ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    xr, xd = RP.load_nodes(a.dataset)
    cb = np.load(os.path.join(CB, 'drug_chemberta_zinc.npz'), allow_pickle=True)
    g = np.load(os.path.join(CB, f'graph_{a.dataset}.npz'), allow_pickle=True)
    m = {k: i for i, k in enumerate(cb['drugbank_id'])}
    xcb = np.stack([cb['x'][m[k]] for k in g['drug_ids']]).astype(np.float32)
    sr = RP.build_adj(xr, a.knn, 'cosine', a.graph); sd = RP.build_adj(xd, a.knn, 'tanimoto', a.graph)
    graph = np.load(os.path.join(CB, f'graph_{a.dataset}.npz'), allow_pickle=True)
    bank = BoostBank(xr, xd, sr, sd, xcb, graph)
    fams = set(a.families.split(','))
    models = a.models.split(',')
    rows = []
    for f in range(5):
        tr, ytr, te, yte = get_split(a.dataset, a.split, f, a.seed)
        Xtr, Xte = RP.build_xy(bank, tr, ytr, te, fams, a.seed * 100 + f)
        preds = {}
        for mdl in models:
            if mdl == 'ens':
                parts = ['lr:0.3', 'hgb', 'xgb', 'mlp']
                ps = [fit(m, Xtr, ytr, Xte, a.seed * 100 + f) for m in parts]
                preds[mdl] = np.mean([rankdata(p) for p in ps], axis=0)
            else:
                preds[mdl] = fit(mdl, Xtr, ytr, Xte, a.seed * 100 + f)
        for mdl, s in preds.items():
            rows.append(dict(fold=f, model=mdl, auc=roc_auc_score(yte, s), aupr=average_precision_score(yte, s)))
    df = pd.DataFrame(rows)
    for mdl in models:
        g_ = df[df.model == mdl]
        print(f'## {a.tag or a.families} {a.dataset} {a.split} fams={a.families} {mdl} ncol={Xtr.shape[1]}: '
              f'AUC {g_.auc.mean():.4f}±{g_.auc.std():.4f} AUPR {g_.aupr.mean():.4f}±{g_.aupr.std():.4f}', flush=True)
    out = os.path.join(BASE, 'plmf_innov_20260921', 'results', 'warm_boost.csv'); new = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['tag', 'dataset', 'split', 'families', 'model', 'ncol', 'auc', 'auc_sd', 'aupr', 'aupr_sd'])
        for mdl in models:
            g_ = df[df.model == mdl]
            w.writerow([a.tag, a.dataset, a.split, a.families, mdl, Xtr.shape[1],
                        f'{g_.auc.mean():.6f}', f'{g_.auc.std():.6f}', f'{g_.aupr.mean():.6f}', f'{g_.aupr.std():.6f}'])


def fit(name, Xtr, ytr, Xte, seed):
    return RP.fit_model(name, Xtr, ytr, Xte, seed)


if __name__ == '__main__':
    main()
