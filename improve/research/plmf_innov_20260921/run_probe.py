# -*- coding: utf-8 -*-
"""Evaluate feature families / learners on PLMF-MDA MDR/MDS with cross-fitting.

Usage:
  python run_probe.py --dataset MDR --split 5cv --families ctx,diff --model hgb
"""
import argparse, os, sys, time, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'mphgnn_plmf_20260921'))
sys.path.insert(0, os.path.join(BASE, 'dlstmda_strongest_20260920'))
from plmf_splits import get_split, load_dataset  # noqa
from bank import Bank  # noqa
import features as F33  # noqa


def load_nodes(ds):
    g = np.load(os.path.join(BASE, 'mphgnn_plmf_20260921', 'cache', f'graph_{ds}.npz'), allow_pickle=True)
    xr = g['mirna_kmer'].astype(np.float32)
    star = np.load(os.path.join(BASE, 'plmf_strongest_20260920', f'graph_cache_{ds}.npz'), allow_pickle=True)
    xd = star['xd'].astype(np.float32)
    return xr, xd


def _sim(x, metric):
    if metric == 'tanimoto':
        xs = (x > 0).astype(np.float32)
        inter = xs @ xs.T; cnt = xs.sum(1, keepdims=True)
        return inter / (cnt + cnt.T - inter + 1e-8)
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return (xn @ xn.T).astype(np.float32)


def build_adj(x, k, metric, mode):
    s = _sim(x.astype(np.float32), metric)
    np.fill_diagonal(s, -1e9 if mode != 'kernel' else 0.0)
    if mode == 'kernel':
        d = np.clip(1.0 - s, 0, 2)
        w = np.exp(-d / 0.1)
        np.fill_diagonal(w, 0.0)
        thr = np.sort(w, axis=1)[:, -k][:, None]
        return np.where(w >= thr, w, 0.0).astype(np.float32)
    idx = np.argpartition(-s, min(k, s.shape[0] - 1), axis=1)[:, :k]
    w = np.zeros_like(s)
    rows = np.arange(s.shape[0])[:, None]
    w[rows, idx] = 1.0 if mode == 'binary' else np.clip(s[rows, idx], 0, None)
    a = np.maximum(w, w.T)
    np.fill_diagonal(a, 0.0)
    return a.astype(np.float32)


def make_bank(xr, xd, mode='binary', k=10):
    sr = build_adj(xr, k, 'cosine', mode)
    sd = build_adj(xd, k, 'tanimoto', mode)
    return Bank(xr, xd, sr, sd)


def build_xy(bank, tr, ytr, te, families, seed):
    Xtr = None
    for fit, qy in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, ytr.astype(int)):
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


def fit_model(name, Xtr, ytr, Xte, seed):
    if name == 'hgb':
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=seed)
    elif name.startswith('lr'):
        C = float(name.split(':')[1]) if ':' in name else 1.0
        m = LogisticRegression(max_iter=3000, C=C)
    elif name == 'mlp':
        m = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=400, random_state=seed,
                          early_stopping=True, n_iter_no_change=20)
    elif name == 'xgb':
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              eval_metric='logloss', tree_method='hist', n_jobs=8,
                              random_state=seed)
    else:
        raise ValueError(name)
    m.fit(Xtr, ytr.astype(int))
    return m.predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR')
    ap.add_argument('--split', default='5cv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--families', default='ctx')
    ap.add_argument('--model', default='hgb')
    ap.add_argument('--folds', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--graph', default='binary', choices=['binary', 'weighted', 'kernel'])
    ap.add_argument('--knn', type=int, default=10)
    a = ap.parse_args()
    fams = set(a.families.split(','))
    xr, xd = load_nodes(a.dataset)
    bank = make_bank(xr, xd, a.graph, a.knn)
    folds = list(range(5)) if a.split == '5cv' else [0]
    if a.folds:
        folds = [int(x) for x in a.folds.split(',')]
    aucs, auprs, ncol = [], [], 0
    for f in folds:
        t0 = time.time()
        tr, ytr, te, yte = get_split(a.dataset, a.split, f, a.seed)
        Xtr, Xte = build_xy(bank, tr, ytr, te, fams, a.seed * 100 + f)
        ncol = Xtr.shape[1]
        s = fit_model(a.model, Xtr, ytr, Xte, a.seed * 100 + f)
        aucs.append(roc_auc_score(yte, s)); auprs.append(average_precision_score(yte, s))
        print(f'  fold {f}: AUC {aucs[-1]:.4f} AUPR {auprs[-1]:.4f} ncol {ncol} ({time.time()-t0:.0f}s)', flush=True)
    print(f'## {a.tag or a.families} {a.dataset} {a.split} {a.model} ncol={ncol}: '
          f'AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f} AUPR {np.mean(auprs):.4f}±{np.std(auprs):.4f}', flush=True)
    import csv as _csv
    out = os.path.join(HERE, 'results', 'probe.csv')
    new = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        wr = _csv.writer(fh)
        if new:
            wr.writerow(['tag', 'dataset', 'split', 'model', 'families', 'ncol',
                         'nfold', 'auc', 'aupr'])
        wr.writerow([a.tag or a.families, a.dataset, a.split, a.model, a.families,
                     ncol, len(folds), f'{np.mean(aucs):.6f}', f'{np.mean(auprs):.6f}'])


if __name__ == '__main__':
    main()
