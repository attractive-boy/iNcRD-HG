# -*- coding: utf-8 -*-
"""Frozen TabICLv2 in-context link predictor on the cold-drug splits (same splits/features as cold_probe.py).
Run in /data1/licheng/incrd-hg/.venv-tabicl/bin/python."""
import argparse, os, sys, time, csv, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cold_probe as CP  # noqa
CHECKPOINT = '/data1/licheng/incrd-hg/.external/models/tabicl/tabicl-classifier-v2-20260212.ckpt'


def tabicl_scores(Xtr, ytr, Xte, device, n_est=32, seed=20260914):
    from tabicl import TabICLClassifier
    m = TabICLClassifier(n_estimators=n_est, random_state=seed, model_path=CHECKPOINT,
                         checkpoint_version='tabicl-classifier-v2-20260212.ckpt', allow_auto_download=False,
                         device=device, offload_mode='auto', n_jobs=4, softmax_temperature=0.9, average_logits=True)
    m.fit(Xtr.astype(np.float32), ytr.astype(np.int64)); p = np.asarray(m.predict_proba(Xte.astype(np.float32)))
    return p[:, int(np.flatnonzero(np.asarray(m.classes_) == 1)[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MDR'); ap.add_argument('--protocol', default='plmf', choices=['plmf', 'ldo', 'plmf_mirna'])
    ap.add_argument('--families', default='ctx'); ap.add_argument('--fit', default='warm'); ap.add_argument('--kshot', type=int, default=0)
    ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--device', default='cuda:0'); ap.add_argument('--tag', default='')
    a = ap.parse_args(); fams = set(a.families.split(','))
    D = pickle.load(open(os.path.join(HERE, 'cache', f'D_{a.dataset}.pkl'), 'rb')); cb = CP.ColdBank(D, 'kernel', 10)
    A = np.zeros((D['nr'], D['nd']), np.int8); A[D['pos'][:, 0], D['pos'][:, 1]] = 1; ddeg = A.sum(0)
    rows = []
    for seed in [int(s) for s in a.seeds.split(',')]:
        side = 'mirna' if a.protocol == 'plmf_mirna' else 'drug'
        if a.protocol == 'plmf':
            folds = [np.where((ddeg >= 11) & (ddeg <= 20))[0]]
        elif a.protocol == 'plmf_mirna':
            mdeg = A.sum(1); folds = [np.where((mdeg >= 5) & (mdeg <= 6))[0]]
        else:
            bins = np.digitize(ddeg, [2, 6, 21, 101])
            folds = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(D['nd']), bins)]
        for fi, test_ent in enumerate(folds):
            t0 = time.time()
            tr, ytr, te, yte = CP.cold_split(D, test_ent, seed * 10 + fi, side, a.kshot)
            Xtr, names = CP.train_features(cb, tr, ytr, fams, a.fit, seed)
            Xte = cb.transform(tr[ytr == 1], te, fams).values
            Xtr, Xte = CP.prep(Xtr, Xte)
            s = tabicl_scores(Xtr, ytr, Xte, a.device)
            auc, ap_ = roc_auc_score(yte, s), average_precision_score(yte, s); mauc, nd_ = CP.macro_auc(te, yte, s, 1 if side == 'drug' else 0)
            rows.append(dict(seed=seed, fold=fi, auc=auc, aupr=ap_, macro_auc=mauc, n_test=len(yte)))
            print(f'  seed {seed} fold {fi}: pairs {len(yte)} AUC {auc:.4f} AUPR {ap_:.4f} macroAUC {mauc:.4f} ncol {Xtr.shape[1]} {time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    print(f"## tabicl {a.tag or a.families} {a.dataset} {a.protocol} fit={a.fit} k={a.kshot}: AUC {df.auc.mean():.4f}±{df.auc.std():.4f} AUPR {df.aupr.mean():.4f}±{df.aupr.std():.4f} macroAUC {df.macro_auc.mean():.4f}", flush=True)
    out = os.path.join(HERE, 'results', 'tabicl_cold.csv'); new = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['tag', 'dataset', 'protocol', 'fit', 'families', 'kshot', 'auc', 'auc_sd', 'aupr', 'aupr_sd', 'macro_auc', 'n_runs'])
        w.writerow([a.tag or a.families, a.dataset, a.protocol, a.fit, a.families, a.kshot, f'{df.auc.mean():.6f}', f'{df.auc.std():.6f}', f'{df.aupr.mean():.6f}', f'{df.aupr.std():.6f}', f'{df.macro_auc.mean():.6f}', len(df)])


if __name__ == '__main__':
    main()
