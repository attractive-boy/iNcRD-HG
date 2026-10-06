# -*- coding: utf-8 -*-
"""Our strongest portable model (frozen TabICLv2 graph-context reference + anchor
shrinkage) on HGNNLDA's dataset (RNAactDrug lncRNA-drug sensitivity).

Same protocol as HGNNLDA: official 5 folds, balanced test (positives + equal random
unknown negatives), AUC / AUPR.  Features are the leak-free graph-derived pair features
(SVD + degree of the training adjacency), i.e. the graph-context tier of iNcRD-HG.

Run in the TabICL venv:
  /data1/licheng/incrd-hg/.venv-tabicl/bin/python run_strongest_rnaactdrug.py --fold 0
"""
import argparse
import csv
import os
import statistics as st
import sys
import time

import numpy as np
from scipy.special import expit, logit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hgnnlda_data as D  # noqa

from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

CHECKPOINT = '/data1/licheng/incrd-hg/.external/models/tabicl/tabicl-classifier-v2-20260212.ckpt'
CKPT_VERSION = 'tabicl-classifier-v2-20260212.ckpt'
SEED = 20260914
W = 0.30
SUBSET = ('hgb', 'et', 'mlp', 'lr')
RESULTS = os.path.join(HERE, 'results')


def feats(A, k=64):
    A = np.asarray(A, dtype=np.float32)
    deg_r, deg_c = A.sum(1), A.sum(0)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(S))
    Er = np.hstack([U[:, :k] * S[:k], (deg_r / max(deg_r.max(), 1))[:, None]])
    Ec = np.hstack([Vt[:k].T * S[:k], (deg_c / max(deg_c.max(), 1))[:, None]])
    return Er.astype(np.float32), Ec.astype(np.float32)


def pairfeat(Er, Ec, dm, dd, idx):
    i, j = idx[:, 0], idx[:, 1]
    return np.hstack([Er[i], Ec[j], Er[i] * Ec[j],
                      (dm[i] / max(dm.max(), 1))[:, None],
                      (dd[j] / max(dd.max(), 1))[:, None]]).astype(np.float32)


def learner(name, seed=SEED):
    return {
        'hgb': lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=seed),
        'et': lambda: ExtraTreesClassifier(n_estimators=600, max_features='sqrt', n_jobs=8, random_state=seed),
        'mlp': lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=600,
                                                                    random_state=seed, early_stopping=True)),
        'lr': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
    }[name]()


def tabicl_scores(Xtr, ytr, Xte, device, n_estimators):
    from tabicl import TabICLClassifier
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=SEED, model_path=CHECKPOINT,
                           checkpoint_version=CKPT_VERSION, allow_auto_download=False,
                           device=device, offload_mode='auto', n_jobs=4,
                           softmax_temperature=0.9, average_logits=True)
    clf.fit(Xtr, ytr.astype(np.int64))
    p = np.asarray(clf.predict_proba(Xte))
    return p[:, int(np.flatnonzero(np.asarray(clf.classes_) == 1)[0])].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=-1)
    ap.add_argument('--neg-ratio', type=float, default=1.0, help='train negatives per positive')
    ap.add_argument('--n-estimators', type=int, default=32)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--skip-tabicl', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    n_l, n_d = D.full_sizes()
    ass = D.load_all()
    os.makedirs(RESULTS, exist_ok=True)
    out = a.out or os.path.join(RESULTS, 'strongest_rnaactdrug.csv')
    folds = range(5) if a.fold < 0 else [a.fold]
    rows = []
    for f in folds:
        t0 = time.time()
        tr, te = D.load_fold(f)
        A = D.train_adjacency(tr, n_l, n_d)
        Er, Ec = feats(A)
        dm, dd = A.sum(1), A.sum(0)

        rng = np.random.default_rng(666 + f)
        n_neg = int(a.neg_ratio * len(tr))
        neg_tr = D.sample_negatives(ass, n_l, n_d, n_neg, rng)
        idx_tr = np.vstack([tr[:, :2], neg_tr])
        y_tr = np.concatenate([np.ones(len(tr)), np.zeros(len(neg_tr))])
        Xtr = pairfeat(Er, Ec, dm, dd, idx_tr)

        neg_te = D.sample_negatives(ass, n_l, n_d, len(te), rng)
        idx_te = np.vstack([te[:, :2], neg_te])
        y_te = np.concatenate([np.ones(len(te)), np.zeros(len(neg_te))])
        Xte = pairfeat(Er, Ec, dm, dd, idx_te)

        ref = (np.full(len(y_te), np.nan) if a.skip_tabicl
               else tabicl_scores(Xtr, y_tr, Xte, a.device, a.n_estimators))
        mem = []
        for name in SUBSET:
            m = learner(name); m.fit(Xtr, y_tr)
            mem.append(logit(np.clip(m.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)))
        fused = (np.full(len(y_te), np.nan) if a.skip_tabicl else
                 expit((1 - W) * logit(np.clip(ref, 1e-6, 1 - 1e-6)) + W * np.mean(mem, axis=0)))

        hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=SEED)
        hgb.fit(Xtr, y_tr)
        s_hgb = hgb.predict_proba(Xte)[:, 1]

        row = dict(dataset='RNAactDrug', fold=f, n_train=len(y_tr), n_test=len(y_te),
                   neg_ratio=a.neg_ratio, n_estimators=a.n_estimators,
                   ref_auc=(np.nan if a.skip_tabicl else roc_auc_score(y_te, ref)),
                   ref_aupr=(np.nan if a.skip_tabicl else average_precision_score(y_te, ref)),
                   fused_auc=(np.nan if a.skip_tabicl else roc_auc_score(y_te, fused)),
                   fused_aupr=(np.nan if a.skip_tabicl else average_precision_score(y_te, fused)),
                   hgb_auc=roc_auc_score(y_te, s_hgb), hgb_aupr=average_precision_score(y_te, s_hgb),
                   time_s=round(time.time() - t0, 1))
        rows.append(row)
        print(f'== fold {f}: ref {row["ref_auc"]:.4f}/{row["ref_aupr"]:.4f} | '
              f'+anchor {row["fused_auc"]:.4f}/{row["fused_aupr"]:.4f} | '
              f'HistGB {row["hgb_auc"]:.4f}/{row["hgb_aupr"]:.4f} ({row["time_s"]}s)', flush=True) if not a.skip_tabicl else \
             print(f'== fold {f}: HistGB {row["hgb_auc"]:.4f}/{row["hgb_aupr"]:.4f} ({row["time_s"]}s)', flush=True)

    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('\n=== iNcRD-HG-strongest on RNAactDrug ===')
    for k in ('ref_auc', 'ref_aupr', 'fused_auc', 'fused_aupr', 'hgb_auc', 'hgb_aupr'):
        v = [r[k] for r in rows]
        print(f'  {k:10s} {st.mean(v):.4f} +/- {st.stdev(v) if len(v) > 1 else 0:.4f}')
    print('saved', out)


if __name__ == '__main__':
    main()
