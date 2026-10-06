# -*- coding: utf-8 -*-
"""Cross-phenotype transfer probe on PLMF-MDA: use the *other* dataset's full label
matrix (MDS when predicting MDR, and vice versa) as side information.

Legitimacy: the other dataset's labels are a different phenotype (sensitivity vs
resistance) and are never the target label; every current-task feature is still
support-only / cross-fitted exactly as in plmf_innov (run_probe.build_xy).

Families added on top of Bank (ctx/diff/...):
  xds  : other-dataset columns for pair (r,d) mapped through shared miRNA name / DrugBank id:
         x_pair (0/1 pair positive in other task), x_deg_r, x_deg_d (other-task degrees, 0 if
         entity absent), x_ppr/x_katz (other-task diffusion score), x_srA/x_ASd (content-smoothed
         other-task adjacency), x_shared_r / x_shared_d (entity present in other task).
  xdsp : only x_pair (ablation)
  xdsd : only degrees + presence (ablation: popularity-in-other-task, no pair identity)
Run: OMP_NUM_THREADS=8 python xds_probe.py --dataset MDR --families ctx,diff,xds --model lr:0.3
"""
import argparse, os, sys, time, csv
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
for p in ('mphgnn_plmf_20260921', 'dlstmda_strongest_20260920', 'plmf_innov_20260921'):
    sys.path.insert(0, os.path.join(BASE, p))
from plmf_splits import get_split, load_dataset  # noqa
from bank import Bank, diffusion  # noqa
import run_probe as RP  # noqa

OTHER = {'MDR': 'MDS', 'MDS': 'MDR'}


def other_matrix(ds):
    mir, dru, pos, _ = load_dataset(ds)
    mo, do, po, _ = load_dataset(OTHER[ds])
    mname = {n.lower(): i for i, n in enumerate(mo['miRNA_Name'].astype(str))}
    dname = {n.lower(): i for i, n in enumerate(do['DrugBank_ID'].astype(str))}
    Ao = np.zeros((len(mo), len(do)), np.float64); Ao[po[:, 0], po[:, 1]] = 1
    rmap = np.array([mname.get(n.lower(), -1) for n in mir['miRNA_Name'].astype(str)])
    dmap = np.array([dname.get(n.lower(), -1) for n in dru['DrugBank_ID'].astype(str)])
    # aligned matrix in current index space; rows/cols absent in other task -> 0
    A = np.zeros((len(mir), len(dru)), np.float64)
    rr = np.where(rmap >= 0)[0]; dd = np.where(dmap >= 0)[0]
    A[np.ix_(rr, dd)] = Ao[np.ix_(rmap[rr], dmap[dd])]
    return A, (rmap >= 0).astype(np.float64), (dmap >= 0).astype(np.float64)


class XBank(Bank):
    def __init__(self, xr, xd, sr, sd, Ax, pr, pd_):
        super().__init__(xr, xd, sr, sd)
        self.Ax, self.pr, self.pd_ = Ax, pr, pd_
        D = diffusion(Ax, 0.3, 0.01)
        self.xg = dict(x_ppr=D['ppr_rd'], x_katz=D['katz_rd'], x_pprd=D['ppr_dr'], x_path7=D['path7'],
                       x_srA=self.sr @ Ax, x_ASd=Ax @ self.sd, x_srASd=self.sr @ Ax @ self.sd)

    def transform(self, sup, q, families):
        df = super().transform(sup, q, families - {'xds', 'xdsp', 'xdsd', 'xdsx'})
        q = np.asarray(q, dtype=np.int64); r, d = q[:, 0], q[:, 1]
        cols = {}
        if families & {'xds', 'xdsp'}:
            cols['x_pair'] = self.Ax[r, d]
        if families & {'xds', 'xdsd'}:
            cols['x_deg_r'] = self.Ax.sum(1)[r]; cols['x_deg_d'] = self.Ax.sum(0)[d]
            cols['x_shared_r'] = self.pr[r]; cols['x_shared_d'] = self.pd_[d]
        if families & {'xds', 'xdsx'}:
            for k, M in self.xg.items():
                v = M[r, d]; cols[k] = v
                cols[k + '_rz'] = (v - M[r].mean(1)) / (M[r].std(1) + 1e-8)
                cols[k + '_dz'] = (v - M[:, d].mean(0)) / (M[:, d].std(0) + 1e-8)
        for k, v in cols.items():
            df[k] = np.asarray(v, np.float64)
        return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--split', default='5cv')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--families', default='ctx,diff,xds')
    ap.add_argument('--model', default='lr:0.3'); ap.add_argument('--graph', default='kernel'); ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--shuffle_other', action='store_true', help='control: permute other-task rows+cols')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    xr, xd = RP.load_nodes(a.dataset)
    sr = RP.build_adj(xr, a.knn, 'cosine', a.graph); sd = RP.build_adj(xd, a.knn, 'tanimoto', a.graph)
    Ax, pr, pd_ = other_matrix(a.dataset)
    if a.shuffle_other:
        rng = np.random.RandomState(a.seed); Ax = Ax[rng.permutation(Ax.shape[0])][:, rng.permutation(Ax.shape[1])]
    bank = XBank(xr, xd, sr, sd, Ax, pr, pd_)
    fams = set(a.families.split(','))
    aucs, auprs = [], []
    for f in range(5):
        t0 = time.time()
        tr, ytr, te, yte = get_split(a.dataset, a.split, f, a.seed)
        Xtr, Xte = RP.build_xy(bank, tr, ytr, te, fams, a.seed * 100 + f)
        s = RP.fit_model(a.model, Xtr, ytr, Xte, a.seed * 100 + f)
        aucs.append(roc_auc_score(yte, s)); auprs.append(average_precision_score(yte, s))
        print(f'  fold {f}: AUC {aucs[-1]:.4f} AUPR {auprs[-1]:.4f} ncol {Xtr.shape[1]} ({time.time()-t0:.0f}s)', flush=True)
    tag = a.tag or a.families + ('_shuf' if a.shuffle_other else '')
    print(f'## {tag} {a.dataset} {a.split} {a.model} ncol={Xtr.shape[1]}: AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f} '
          f'AUPR {np.mean(auprs):.4f}±{np.std(auprs):.4f}', flush=True)
    out = os.path.join(HERE, 'results', 'xds_probe.csv'); new = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new: w.writerow(['tag', 'dataset', 'split', 'model', 'families', 'shuffle', 'ncol', 'auc', 'auc_sd', 'aupr', 'aupr_sd', 'fold_auc', 'fold_aupr'])
        w.writerow([tag, a.dataset, a.split, a.model, a.families, int(a.shuffle_other), Xtr.shape[1],
                    f'{np.mean(aucs):.6f}', f'{np.std(aucs):.6f}', f'{np.mean(auprs):.6f}', f'{np.std(auprs):.6f}',
                    ';'.join(f'{x:.5f}' for x in aucs), ';'.join(f'{x:.5f}' for x in auprs)])


if __name__ == '__main__':
    main()
