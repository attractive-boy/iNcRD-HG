# -*- coding: utf-8 -*-
"""Run iNcRD-HG's strongest *portable* reference on the DLST-MDA benchmark.

Portable part = the frozen TabICLv2 graph-context reference (miRNA view = raw, 32 members)
+ the task-shared anchor-shrinkage fusion (members hgb+et+mlp+lr, w=0.30), exactly as in
improve/context_anchor_verify.py.  The evidence-tiered and DepMap-context tiers of the
full iNcRD-HG reference are not portable (DLST-MDA has no expression / gene-pathway /
target-evidence data).

Run in the TabICL venv:
  /data1/licheng/incrd-hg/.venv-tabicl/bin/python run_tabicl.py --device cuda:0
"""
import argparse
import csv
import json
import os
import random
import statistics as st
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa

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


def learner(name, seed=SEED):
    table = {
        'hgb': lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=seed),
        'et': lambda: ExtraTreesClassifier(n_estimators=600, max_features='sqrt', n_jobs=8, random_state=seed),
        'mlp': lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=600,
                                                                    random_state=seed, early_stopping=True)),
        'lr': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
    }
    return table[name]()


def metrics(y, scores, pairs):
    y = np.asarray(y); scores = np.asarray(scores)
    auc = float(roc_auc_score(y, scores))
    ap = float(average_precision_score(y, scores))
    per_drug = []
    for drug in np.unique(pairs[1]):
        m = pairs[1] == drug
        if len(np.unique(y[m])) == 2:
            per_drug.append(average_precision_score(y[m], scores[m]))
    return dict(auc=auc, ap=ap, macro_drug_ap=float(np.mean(per_drug)) if per_drug else None,
                acc=float(((scores >= 0.5).astype(int) == y).mean()))


def tabicl_scores(x_train, y_train, x_test, device, n_estimators, view, seed=SEED):
    from tabicl import TabICLClassifier
    model = TabICLClassifier(n_estimators=n_estimators, random_state=seed,
                             model_path=CHECKPOINT, checkpoint_version=CKPT_VERSION,
                             allow_auto_download=False, device=device,
                             offload_mode='auto', n_jobs=4, softmax_temperature=0.9,
                             average_logits=True)
    if view == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=min(256, len(x_train)), output_distribution='normal',
                                 subsample=100000, random_state=20260910)
        x_train = qt.fit_transform(x_train).astype(np.float32)
        x_test = qt.transform(x_test).astype(np.float32)
    model.fit(x_train, np.asarray(y_train).astype(np.int64))
    p = np.asarray(model.predict_proba(x_test))
    classes = np.asarray(model.classes_)
    return p[:, int(np.flatnonzero(classes == 1)[0])].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--n-estimators', type=int, default=32)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--view', default='raw', choices=['raw', 'quantile'])
    ap.add_argument('--out', default=os.path.join(RESULTS, 'strongest_per_fold.csv'))
    ap.add_argument('--proc', default=None)
    ap.add_argument('--keep-all', action='store_true', help='do not drop entity-level duplicate test pairs (author-row protocol)')
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(',')]
    os.makedirs(RESULTS, exist_ok=True)

    xr, xd, mirnas, drugs = F.load_cache()
    encoder = F.encoder_from_arrays(xr, xd)
    _, _, fold_data, _ = F.load_pairs(args.proc) if args.proc else F.load_pairs()

    rows = []
    for f in folds:
        t0 = time.time()
        a = F.fold_arrays(mirnas, drugs, fold_data, f, encoder=encoder, drop_dup=not args.keep_all)
        # TabICL reference
        ref = tabicl_scores(a['x_train'], a['y_train'], a['x_test'], args.device,
                            args.n_estimators, args.view, args.seed)
        # anchor members
        mem_logits = []
        for name in SUBSET:
            m = learner(name, args.seed)
            m.fit(a['x_train'], a['y_train'])
            mem_logits.append(logit(np.clip(m.predict_proba(a['x_test'])[:, 1], 1e-6, 1 - 1e-6)))
        mem = np.mean(mem_logits, axis=0)
        fused = expit((1 - W) * logit(np.clip(ref, 1e-6, 1 - 1e-6)) + W * mem)
        mr = metrics(a['y_test'], ref, a['pairs_test'])
        mf = metrics(a['y_test'], fused, a['pairs_test'])
        row = dict(dataset='DLST-MDA-1:1' + ('-keepall' if args.keep_all else ''), method='iNcRD-HG-strongest(graphctx+anchor)',
                   fold=f, seed=args.seed, ref_auc=mr['auc'], ref_aupr=mr['ap'],
                   fused_auc=mf['auc'], fused_aupr=mf['ap'], acc=mf['acc'],
                   macro_drug_ap=mf['macro_drug_ap'], time_s=round(time.time() - t0, 1),
                   n_cols=len(a['names']))
        rows.append(row)
        print(f'== fold {f}: graphctx-ref AUC {mr["auc"]:.4f}/AP {mr["ap"]:.4f} | '
              f'+anchor w{W} AUC {mf["auc"]:.4f}/AP {mf["ap"]:.4f} '
              f'(cols {len(a["names"])}, {row["time_s"]}s)', flush=True)

    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('\n=== iNcRD-HG strongest (portable graph-context + anchor) ===')
    for k in ('ref_auc', 'ref_aupr', 'fused_auc', 'fused_aupr', 'acc'):
        v = [r[k] for r in rows]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f'  {k}: {st.mean(v):.4f} +/- {sd:.4f}')
    print('saved', args.out)


if __name__ == '__main__':
    main()
