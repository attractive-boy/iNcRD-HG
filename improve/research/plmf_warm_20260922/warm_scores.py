# -*- coding: utf-8 -*-
"""Reproduce the strongest warm single model (ctx+diff x LR) on PLMF-MDA 5cv and
persist per-pair test scores + cached feature matrices for downstream probes.

  python warm_scores.py --dataset MDR --seed 42 --families ctx,diff --model lr:0.3
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'plmf_innov_20260921'))
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
sys.path.insert(0, os.path.join(BASE, 'dlstmda_strongest_20260920'))
from run_probe import load_nodes, make_bank, build_xy, fit_model  # noqa
from plmf_splits import get_split  # noqa


def build_xy_k(bank, tr, ytr, te, families, seed, K):
    from sklearn.model_selection import StratifiedKFold
    Xtr = None
    for fit, qy in StratifiedKFold(K, shuffle=True, random_state=seed).split(tr, ytr.astype(int)):
        sup = tr[fit[ytr[fit] == 1]]
        f = bank.transform(sup, tr[qy], families)
        if Xtr is None:
            Xtr = np.zeros((len(ytr), f.shape[1]), np.float32)
        Xtr[qy] = f.values
    Xte = bank.transform(tr[ytr == 1], te, families).values
    both = np.concatenate([Xtr, Xte], 0)
    both = np.log1p(np.abs(both)) * np.sign(both)
    Xtr, Xte = both[:len(ytr)], both[len(ytr):]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def cached_xy(bank, ds, seed, fold, fams, graph, knn, inner=5, split='5cv'):
    key = f'{ds}_{"single" if split == "single" else ""}s{seed}_f{fold}_{"+".join(sorted(fams))}{"" if inner == 5 else "K%d" % inner}_{graph}{knn}.npz'
    p = os.path.join(HERE, 'cache', key)
    tr, ytr, te, yte = get_split(ds, split, fold, seed)
    if os.path.exists(p):
        z = np.load(p)
        return z['Xtr'], z['Xte'], tr, ytr, te, yte
    Xtr, Xte = build_xy(bank, tr, ytr, te, fams, seed * 100 + fold) if inner == 5 else build_xy_k(bank, tr, ytr, te, fams, seed * 100 + fold, inner)
    np.savez_compressed(p, Xtr=Xtr.astype(np.float32), Xte=Xte.astype(np.float32),
                        tr=tr, ytr=ytr, te=te, yte=yte)
    return Xtr, Xte, tr, ytr, te, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--families', default='ctx,diff')
    ap.add_argument('--model', default='lr:0.3')
    ap.add_argument('--graph', default='kernel')
    ap.add_argument('--knn', type=int, default=10)
    ap.add_argument('--tag', default='')
    ap.add_argument('--inner', type=int, default=5)
    ap.add_argument('--split', default='5cv')
    a = ap.parse_args()
    fams = set(a.families.split(','))
    xr, xd = load_nodes(a.dataset)
    bank = make_bank(xr, xd, a.graph, a.knn)
    rows, aucs, auprs = [], [], []
    for f in (range(5) if a.split == '5cv' else [0]):
        t0 = time.time()
        Xtr, Xte, tr, ytr, te, yte = cached_xy(bank, a.dataset, a.seed, f, fams, a.graph, a.knn, a.inner, a.split)
        s = fit_model(a.model, Xtr, ytr, Xte, a.seed * 100 + f)
        aucs.append(roc_auc_score(yte, s)); auprs.append(average_precision_score(yte, s))
        rows.append(pd.DataFrame({'fold': f, 'r': te[:, 0], 'd': te[:, 1], 'y': yte.astype(int), 'score': s}))
        print(f'  fold {f}: AUC {aucs[-1]:.4f} AUPR {auprs[-1]:.4f} ncol {Xtr.shape[1]} ({time.time()-t0:.0f}s)', flush=True)
    tag = a.tag or f'{a.families.replace(",", "+")}{"" if a.inner == 5 else "K%d" % a.inner}_{a.model.replace(":", "")}'
    out = os.path.join(HERE, 'results', f'scores_{a.dataset}_{"single" if a.split == "single" else ""}s{a.seed}_{tag}.csv')
    pd.concat(rows).to_csv(out, index=False)
    print(f'## {a.dataset} seed{a.seed} {tag}: AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f} '
          f'AUPR {np.mean(auprs):.4f}±{np.std(auprs):.4f} -> {out}', flush=True)


if __name__ == '__main__':
    main()
