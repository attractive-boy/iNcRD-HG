# -*- coding: utf-8 -*-
"""PHL — Provenance-Hypergraph Heuristic Learner (single model, linear readout).

score(m,d) = w · z( h(m,d | support graph, provenance hypergraph, miRNA side-information) ) + b
  h = deterministic heuristic bank: graph-context (33) + diffusion (52) [ctx+diff]
      + study/context hyperedge heuristics (prov: pmid, costudy, ctx, rec)
      + sparse-entity priors (mprior: name family, seed family, HMDD disease profile, other-phenotype degree, PubMed)
  z = log1p-signed + standardisation; w,b = logistic regression (C=0.3).  Training rows are cross-fitted
  with K inner folds so the query pair is never in its own support.

  python final.py --dataset MDR --split 5cv --seed 42 --variant final
"""
import os, sys, argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset
from mprior import MPrior, build_extra
from prov import Prov, crossfit

VARIANTS = {
    'base':        dict(prov='', mprior=''),
    'prov':        dict(prov='pmid,costudy,ctx,rec', mprior=''),
    'mprior':      dict(prov='', mprior='fam,seed,hmdd,xdeg'),
    'final':       dict(prov='pmid,costudy,ctx,rec', mprior='fam,seed,hmdd,xdeg'),
    'final_pub':   dict(prov='pmid,costudy,ctx,rec', mprior='fam,seed,hmdd,xdeg,pub'),
    'final_shuf':  dict(prov='pmid,costudy,ctx,rec', mprior='fam,seed,hmdd,xdeg', shuffle=True),
    'pmid_only':   dict(prov='pmid', mprior=''),
    'final_noxdeg': dict(prov='pmid,costudy,ctx,rec', mprior='fam,seed,hmdd'),
    'final_lean':  dict(prov='pmid,costudy', mprior='fam,hmdd,xdeg'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--split', default='5cv'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--inner', type=int, default=10); ap.add_argument('--variant', default='final'); ap.add_argument('--C', type=float, default=0.3); ap.add_argument('--model', default='lr')
    a = ap.parse_args()
    v = VARIANTS[a.variant]; shuf = v.get('shuffle', False)
    pv = Prov(a.dataset, shuf, np.random.RandomState(a.seed), groups=v['prov'].split(',')) if v['prov'] else None
    mp = MPrior(a.dataset, rng=np.random.RandomState(a.seed), shuffle=set(v['mprior'].split(',')) if shuf else None) if v['mprior'] else None
    if mp is not None and 'pub' in v['mprior'] and (mp.pub is None or mp.pub_cov < 0.95):
        raise SystemExit(f'PubMed coverage {getattr(mp, "pub_cov", 0):.2f} < 0.95; wait for the fetch to finish')
    prefix = f'{a.dataset}_{"single" if a.split == "single" else ""}s{a.seed}'
    res, rows = [], []
    for f in (range(5) if a.split == '5cv' else [0]):
        z = np.load(os.path.join(HERE, 'cache', f'{prefix}_f{f}_ctx+diff{"" if a.inner == 5 else "K%d" % a.inner}_kernel10.npz'))
        Xtr, Xte, tr, ytr, te, yte = z['Xtr'], z['Xte'], z['tr'], z['ytr'], z['te'], z['yte']
        if pv is not None:
            Ptr, Pte = crossfit(pv, tr, ytr, te, a.seed * 100 + f, lambda o, s, q: o.transform(s, q), K=a.inner)
            Xtr = np.concatenate([Xtr, Ptr], 1); Xte = np.concatenate([Xte, Pte], 1)
        if mp is not None:
            Etr, Ete, _ = build_extra(mp, tr, ytr, te, set(v['mprior'].split(',')), a.seed * 100 + f, K=a.inner)
            Xtr = np.concatenate([Xtr, Etr], 1); Xte = np.concatenate([Xte, Ete], 1)
        if a.model == 'lr':
            m = LogisticRegression(max_iter=5000, C=a.C).fit(Xtr, ytr.astype(int))
        elif a.model == 'hgb':
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=a.seed).fit(Xtr, ytr.astype(int))
        elif a.model == 'mlp':
            from sklearn.neural_network import MLPClassifier
            m = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=400, random_state=a.seed, early_stopping=True, n_iter_no_change=20).fit(Xtr, ytr.astype(int))
        s = m.predict_proba(Xte)[:, 1]
        res.append((roc_auc_score(yte, s), average_precision_score(yte, s)))
        sup = tr[ytr == 1]; dr = np.bincount(sup[:, 0], minlength=pv.nr if pv else mp.nr if mp else 100000)
        rows.append(pd.DataFrame({'fold': f, 'r': te[:, 0], 'd': te[:, 1], 'y': yte.astype(int), 'score': s, 'deg_r': dr[te[:, 0]]}))
    res = np.array(res)
    out = os.path.join(HERE, 'results', 'final', f'scores_{prefix}_K{a.inner}_{a.variant}{"" if a.model == "lr" else "_" + a.model}.csv')
    pd.concat(rows).to_csv(out, index=False)
    line = f'## {prefix} K{a.inner} {a.variant}{"" if a.model == "lr" else "/" + a.model} ncol={Xtr.shape[1]}: AUC {res[:,0].mean():.4f}±{res[:,0].std():.4f} AUPR {res[:,1].mean():.4f}±{res[:,1].std():.4f}'
    print(line, flush=True); open(os.path.join(HERE, 'results', 'final', 'summary.txt'), 'a').write(line + '\n')


if __name__ == '__main__':
    main()
