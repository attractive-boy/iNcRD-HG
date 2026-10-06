"""Compute the full metric set (AUC/AUPR/ACC/Precision/Recall/Specificity/F1/MCC)
for our TabICLv2 transfer by re-scoring the saved pair-feature npz dumps.

Threshold-based metrics use 0.5 and the best-F1 threshold (Youden/threshold sweep).
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

CKPT = "/data1/licheng/incrd-hg/.external/models/tabicl/tabicl-classifier-v2-20260212.ckpt"
HERE = os.path.dirname(os.path.abspath(__file__))


def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(ACC=(tp + tn) / len(y), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0), Specificity=spec,
                F1=f1_score(y, pred, zero_division=0), MCC=matthews_corrcoef(y, pred))


def score(files, tag):
    from tabicl import TabICLClassifier
    rows = []
    for f in files:
        z = np.load(f)
        Xtr = z["X_train"].astype(np.float64); ytr = z["y_train"].astype(np.int64)
        Xte = z["X_test"].astype(np.float64); yte = z["y_test"].astype(int)
        clf = TabICLClassifier(n_estimators=8, random_state=666, model_path=CKPT,
                               checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
                               allow_auto_download=False, device="cuda:0", batch_size=4)
        clf.fit(Xtr, ytr)
        s = np.asarray(clf.predict_proba(Xte))[:, 1]
        m05 = metrics_at(yte, s, 0.5)
        # best-F1 threshold over the actual PR-curve thresholds (handles compressed scores)
        prec, rec, ths = precision_recall_curve(yte, s)
        f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
        bt = float(ths[int(np.argmax(f1s[:-1]))]) if len(ths) else 0.5
        mbf = metrics_at(yte, s, bt)
        rows.append(dict(file=os.path.basename(f), AUC=roc_auc_score(yte, s),
                         AUPR=average_precision_score(yte, s),
                         **{f"{k}@0.5": v for k, v in m05.items()},
                         best_f1_thr=float(bt), **{f"{k}@bestF1": v for k, v in mbf.items()}))
    agg = {}
    for k in rows[0]:
        if k == "file":
            continue
        vals = np.array([r[k] for r in rows], dtype=float)
        agg[k] = (float(vals.mean()), float(vals.std()))
    print(f"\n===== {tag} (n={len(rows)}) =====")
    for k, (m, sd) in agg.items():
        print(f"  {k:16s} {m:.4f} +- {sd:.4f}")
    json.dump({"tag": tag, "per_file": rows, "agg": agg},
              open(os.path.join(HERE, f"metrics_ours_{tag}.json"), "w"), indent=2)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ncdr"
    if which == "ncdr":
        score(sorted(glob.glob(os.path.join(HERE, "tabicl_ncdr", "ncdr_seed*.npz"))), "ncdr")
    elif which == "gcfmcl":
        score(sorted(glob.glob(os.path.join(HERE, "tabicl_gcfmcl", "*.npz"))), "gcfmcl")
    elif which == "pan2693":
        score(sorted(glob.glob("/tmp/opencode/feat_lrgcpnd_pan_fold*.npz")), "pan2693")
    elif which == "lnc41":
        score(sorted(glob.glob("/tmp/opencode/feat_lrgcpnd_lnc_fold*.npz")), "lnc41")
    elif which == "rnaactdrug":
        score(sorted(glob.glob("/tmp/opencode/feat_rnaactdrug.npz")), "rnaactdrug")
