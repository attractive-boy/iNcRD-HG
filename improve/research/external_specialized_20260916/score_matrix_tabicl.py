"""Score iNcRD-HG graph pair features on external datasets with the frozen
TabICLv2 checkpoint (the flagship tabular-foundation stage of our pipeline).

Consumes the npz dumps produced by transfer_matrix_benchmark.py --dump-prefix
(or transfer_gcfmcl_features.py) and reports AUC / AUPR.
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

CKPT = "/data1/licheng/incrd-hg/.external/models/tabicl/tabicl-classifier-v2-20260212.ckpt"


def score(files, seeds, device, n_estimators):
    from tabicl import TabICLClassifier
    per = []
    for f in files:
        z = np.load(f)
        Xtr, ytr = z["X_train"].astype(np.float64), z["y_train"].astype(np.int64)
        Xte, yte = z["X_test"].astype(np.float64), z["y_test"]
        aucs, auprs = [], []
        for sd in seeds:
            clf = TabICLClassifier(n_estimators=n_estimators, random_state=sd, model_path=CKPT,
                                   checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
                                   allow_auto_download=False, device=device, batch_size=4)
            clf.fit(Xtr, ytr)
            s = np.asarray(clf.predict_proba(Xte))[:, 1]
            aucs.append(roc_auc_score(yte, s)); auprs.append(average_precision_score(yte, s))
        per.append((os.path.basename(f), float(np.mean(aucs)), float(np.mean(auprs))))
        print(f"{os.path.basename(f)}: AUC {np.mean(aucs):.4f} AUPR {np.mean(auprs):.4f}", flush=True)
    a = np.array([[p[1], p[2]] for p in per])
    return per, a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[666])
    ap.add_argument("--n-estimators", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = []
    for g in a.glob:
        files += sorted(glob.glob(g))
    print(f"scoring {len(files)} files")
    per, arr = score(files, a.seeds, a.device, a.n_estimators)
    res = {"files": per, "mean_auc": float(arr[:, 0].mean()), "std_auc": float(arr[:, 0].std()),
           "mean_aupr": float(arr[:, 1].mean()), "std_aupr": float(arr[:, 1].std())}
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nTabICLv2 over {len(files)} files: AUC {arr[:,0].mean():.4f}±{arr[:,0].std():.4f} "
          f"AUPR {arr[:,1].mean():.4f}±{arr[:,1].std():.4f}")
    print("wrote", a.out)
