# -*- coding: utf-8 -*-
"""iNcRD-HG strongest portable reference on MSFFMDI's ncDR / RNAInter benchmark.

Run in the TabICL venv:
  /data1/licheng/incrd-hg/.venv-tabicl/bin/python run_tabicl_msffmdi.py --dataset ncDR
"""
import argparse
import csv
import os
import pickle
import statistics as st
import sys
import time

import numpy as np
from scipy.special import expit, logit

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'dlstmda_strongest_20260920'))
sys.path.insert(0, HERE)
import features as F  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix

CHECKPOINT = '/data1/licheng/incrd-hg/.external/models/tabicl/tabicl-classifier-v2-20260212.ckpt'
CKPT_VERSION = 'tabicl-classifier-v2-20260212.ckpt'
SEED = 20260914
W = 0.30
SUBSET = ('hgb', 'et', 'mlp', 'lr')
RESULTS = os.path.join(HERE, 'results')


def learner(name, seed=SEED):
    return {
        'hgb': lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=seed),
        'et': lambda: ExtraTreesClassifier(n_estimators=600, max_features='sqrt', n_jobs=8, random_state=seed),
        'mlp': lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=600,
                                                                    random_state=seed, early_stopping=True)),
        'lr': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
    }[name]()


def metrics(y, scores, pairs):
    y = np.asarray(y); scores = np.asarray(scores)
    pred = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auc=float(roc_auc_score(y, scores)),
                aupr=float(average_precision_score(y, scores)),
                acc=float(accuracy_score(y, pred)),
                sen=float(tp / (tp + fn) if (tp + fn) else 0.0),
                spec=float(tn / (tn + fp) if (tn + fp) else 0.0))


def tabicl_scores(x_train, y_train, x_test, device, n_estimators, seed=SEED):
    from tabicl import TabICLClassifier
    model = TabICLClassifier(n_estimators=n_estimators, random_state=seed,
                             model_path=CHECKPOINT, checkpoint_version=CKPT_VERSION,
                             allow_auto_download=False, device=device,
                             offload_mode='auto', n_jobs=4, softmax_temperature=0.9,
                             average_logits=True)
    model.fit(x_train, np.asarray(y_train).astype(np.int64))
    p = np.asarray(model.predict_proba(x_test))
    classes = np.asarray(model.classes_)
    return p[:, int(np.flatnonzero(classes == 1)[0])].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['ncDR', 'RNAInter'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n-estimators', type=int, default=32)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    d = np.load(os.path.join(HERE, f'graph_cache_{args.dataset}.npz'), allow_pickle=True)
    xr, xd = d['xr'].astype(np.float32), d['xd'].astype(np.float32)
    with open(os.path.join(HERE, f'folds_{args.dataset}.pkl'), 'rb') as fh:
        mirnas, drugs, folds = pickle.load(fh)
    encoder = F.encoder_from_arrays(xr, xd)

    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, f'strongest_{args.dataset}.csv')
    rows = []
    for f in range(5):
        t0 = time.time()
        a = F.fold_arrays(mirnas, drugs, folds, f, encoder=encoder)
        ref = tabicl_scores(a['x_train'], a['y_train'], a['x_test'], args.device, args.n_estimators)
        mem_logits = []
        for name in SUBSET:
            m = learner(name); m.fit(a['x_train'], a['y_train'])
            mem_logits.append(logit(np.clip(m.predict_proba(a['x_test'])[:, 1], 1e-6, 1 - 1e-6)))
        mem = np.mean(mem_logits, axis=0)
        fused = expit((1 - W) * logit(np.clip(ref, 1e-6, 1 - 1e-6)) + W * mem)
        mr = metrics(a['y_test'], ref, a['pairs_test'])
        mf = metrics(a['y_test'], fused, a['pairs_test'])
        rows.append(dict(dataset=args.dataset, method='iNcRD-HG-strongest', fold=f,
                         ref_auc=mr['auc'], ref_aupr=mr['aupr'],
                         fused_auc=mf['auc'], fused_aupr=mf['aupr'],
                         acc=mf['acc'], sen=mf['sen'], spec=mf['spec'],
                         n_cols=len(a['names']), time_s=round(time.time() - t0, 1)))
        print(f'== {args.dataset} fold {f}: ref {mr["auc"]:.4f}/{mr["aupr"]:.4f} | '
              f'+anchor {mf["auc"]:.4f}/{mf["aupr"]:.4f} ACC {mf["acc"]:.4f} ({rows[-1]["time_s"]}s)',
              flush=True)

    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'\n=== iNcRD-HG-strongest {args.dataset} ===')
    for k in ('ref_auc', 'ref_aupr', 'fused_auc', 'fused_aupr', 'acc', 'sen', 'spec'):
        v = [r[k] for r in rows]
        print(f'  {k}: {st.mean(v):.4f} +/- {st.stdev(v):.4f}')
    print('saved', out)


if __name__ == '__main__':
    main()
