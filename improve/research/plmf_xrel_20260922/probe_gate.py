# -*- coding: utf-8 -*-
"""Learning-side probes on the PHL feature bank (seed 42, 5cv), no new information:
  gate  : degree-conditioned linear readout  X ⊗ [1, g_d, g_r, ...] where g_d = 1[support degree of drug <= tau_d], g_r = 1[support degree of miRNA <= tau_r]
  slope : smooth version X ⊗ [1, z(log1p deg_d), z(log1p deg_r)]
  w     : inverse-degree sample weights (IPS-style) on training rows
  diag  : feature-group contributions of the top-scored negatives vs all negatives (plain LR)."""
import os, sys, argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); WARM = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_warm_20260922'
GROUPS = [('ctx', 0, 33), ('diff', 33, 85), ('prov', 85, 109), ('mprior', 109, 128)]

def load(ds, seed, f, K=10, split='5cv'):
    prefix = f'{ds}_{"single" if split == "single" else ""}s{seed}'
    z = np.load(os.path.join(WARM, 'cache', f'{prefix}_f{f}_ctx+diffK{K}_kernel10.npz'))
    zz = np.load(os.path.join(HERE, 'cache', f'phl_{prefix}_f{f}_K{K}.npz'))
    Xtr = np.concatenate([z['Xtr'], zz['Ptr'], zz['Etr']], 1); Xte = np.concatenate([z['Xte'], zz['Pte'], zz['Ete']], 1)
    tr, ytr, te, yte = z['tr'], z['ytr'], z['te'], z['yte']
    nr, nd = int(max(tr[:, 0].max(), te[:, 0].max())) + 1, int(max(tr[:, 1].max(), te[:, 1].max())) + 1
    # support degrees seen by each row (train rows: their inner-fold support; test rows: all training positives)
    dr_tr = np.zeros(len(tr)); dd_tr = np.zeros(len(tr))
    for fit, qy in StratifiedKFold(K, shuffle=True, random_state=seed * 100 + f).split(tr, ytr.astype(int)):
        sup = tr[fit[ytr[fit] == 1]]; dr_tr[qy] = np.bincount(sup[:, 0], minlength=nr)[tr[qy, 0]]; dd_tr[qy] = np.bincount(sup[:, 1], minlength=nd)[tr[qy, 1]]
    sup = tr[ytr == 1]; dr_te = np.bincount(sup[:, 0], minlength=nr)[te[:, 0]]; dd_te = np.bincount(sup[:, 1], minlength=nd)[te[:, 1]]
    return Xtr, Xte, ytr, yte, dr_tr, dd_tr, dr_te, dd_te

def expand(X, gates):
    return np.concatenate([X] + [X * g[:, None] for g in gates] + [g[:, None] for g in gates], 1)

def run(ds, seed, variant, C=0.3, split='5cv'):
    res = []
    for f in (range(5) if split == '5cv' else [0]):
        Xtr, Xte, ytr, yte, dr_tr, dd_tr, dr_te, dd_te = load(ds, seed, f, split=split)
        w = None
        kind, _, arg = variant.partition(':')
        if kind == 'plain': A, B = Xtr, Xte
        elif kind == 'gate':
            gtr, gte = [], []
            for tok in arg.split(','):
                side, tau = tok[0], float(tok[1:])
                if side == 'd': gtr.append((dd_tr <= tau).astype(np.float32)); gte.append((dd_te <= tau).astype(np.float32))
                else: gtr.append((dr_tr <= tau).astype(np.float32)); gte.append((dr_te <= tau).astype(np.float32))
            A, B = expand(Xtr, gtr), expand(Xte, gte)
        elif kind == 'slope':
            ltr = [np.log1p(dd_tr), np.log1p(dr_tr)]; lte = [np.log1p(dd_te), np.log1p(dr_te)]
            gtr, gte = [], []
            for a_, b_ in zip(ltr, lte):
                mu, sd = a_.mean(), a_.std() + 1e-6; gtr.append((a_ - mu) / sd); gte.append((b_ - mu) / sd)
            A, B = expand(Xtr, gtr), expand(Xte, gte)
        elif kind == 'w':
            alpha = float(arg); w = (1 + dr_tr) ** (-alpha)
            for c in (0, 1): w[ytr == c] /= w[ytr == c].mean()
            A, B = Xtr, Xte
        m = LogisticRegression(max_iter=5000, C=C).fit(A, ytr.astype(int), sample_weight=w); s = m.predict_proba(B)[:, 1]
        res.append((roc_auc_score(yte, s), average_precision_score(yte, s)))
    res = np.array(res); return res[:, 0].mean(), res[:, 1].mean(), res[:, 0].std()

def diag(ds, seed, f=0, top=100):
    Xtr, Xte, ytr, yte, dr_tr, dd_tr, dr_te, dd_te = load(ds, seed, f)
    m = LogisticRegression(max_iter=5000, C=0.3).fit(Xtr, ytr.astype(int)); s = m.decision_function(Xte)
    contrib = Xte * m.coef_[0][None, :]
    neg = np.where(yte == 0)[0]; order = neg[np.argsort(-s[neg])]; topn = order[:top]
    print(f'--- {ds} fold {f}: top-{top} negatives: median deg_d {np.median(dd_te[topn]):.0f} deg_r {np.median(dr_te[topn]):.0f}; all negatives: deg_d {np.median(dd_te[neg]):.0f} deg_r {np.median(dr_te[neg]):.0f}')
    pos = np.where(yte == 1)[0]
    for name, a, b in GROUPS:
        print(f'  {name:7s} contribution: top-neg {contrib[topn, a:b].sum(1).mean():+.2f}  all-neg {contrib[neg, a:b].sum(1).mean():+.2f}  all-pos {contrib[pos, a:b].sum(1).mean():+.2f}')
    # single columns with the largest excess contribution on the top negatives
    ex = contrib[topn].mean(0) - contrib[neg].mean(0); idx = np.argsort(-ex)[:12]
    print('  columns with largest excess contribution on top negatives (index: excess):', ', '.join(f'{i}:{ex[i]:+.2f}' for i in idx))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='MDR'); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--split', default='5cv')
    ap.add_argument('--variants', default='plain,gate:d5,gate:d10,gate:d20,gate:r1,gate:r3,gate:d10-r1,slope,w:0.5,w:1'); ap.add_argument('--diag', action='store_true'); a = ap.parse_args()
    if a.diag: diag(a.dataset, a.seed)
    base = None
    for v in a.variants.split(','):
        v = v.replace('-', ',')
        auc, ap_, sd = run(a.dataset, a.seed, v, split=a.split)
        if base is None: base = (auc, ap_)
        line = f'## {a.dataset} s{a.seed} {a.split} {v:14s}: AUC {auc:.4f}±{sd:.4f} AUPR {ap_:.4f}  Δ {auc - base[0]:+.4f} / {ap_ - base[1]:+.4f}'
        print(line, flush=True); open(os.path.join(HERE, 'results', 'gate.txt'), 'a').write(line + '\n')
