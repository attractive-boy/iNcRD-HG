"""Second stage of the DeepLDA transfer: score iNcRD-HG pair embeddings with a
tabular learner (HistGB) and the frozen TabICLv2 checkpoint, mirroring the
accepted 'graph-derived pair features + tabular foundation model' pipeline.

Run in the TabICL venv for TabICLv2; HistGB works in either env.
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier

CKPT = "/data1/licheng/incrd-hg/.external/models/tabicl/tabicl-classifier-v2-20260212.ckpt"
OUT = "/tmp/opencode/deeplncrna"


def run_histgb(files):
    per = []
    for f in files:
        z = np.load(f)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                             max_leaf_nodes=31, random_state=666)
        clf.fit(z["X_train"], z["y_train"])
        s = clf.predict_proba(z["X_test"])[:, 1]
        per.append((os.path.basename(f), roc_auc_score(z["y_test"], s),
                    average_precision_score(z["y_test"], s)))
    a = np.array([[p[1], p[2]] for p in per])
    return per, a


def run_tabicl(files, seeds=(666, 0, 1, 2, 3)):
    from tabicl import TabICLClassifier
    per = []
    for f in files:
        z = np.load(f)
        Xtr, ytr = z["X_train"], z["y_train"].astype(np.int64)
        Xte, yte = z["X_test"], z["y_test"]
        aucs, auprs = [], []
        for sd in seeds:
            clf = TabICLClassifier(n_estimators=8, random_state=sd, model_path=CKPT,
                                   checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
                                   allow_auto_download=False, device="cuda:0", batch_size=4)
            clf.fit(Xtr, ytr)
            s = np.asarray(clf.predict_proba(Xte))[:, 1]
            aucs.append(roc_auc_score(yte, s)); auprs.append(average_precision_score(yte, s))
        per.append((os.path.basename(f), float(np.mean(aucs)), float(np.mean(auprs))))
    a = np.array([[p[1], p[2]] for p in per])
    return per, a


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    report = {}
    for ds in ("1", "2"):
        files = sorted(glob.glob(os.path.join(OUT, f"deeplncrna{ds}_seed*.npz")))
        if not files:
            continue
        if which in ("both", "histgb"):
            per, a = run_histgb(files)
            report[f"dataset{ds}_histgb"] = {"per_seed": per,
                                             "mean_auc": float(a[:, 0].mean()),
                                             "std_auc": float(a[:, 0].std()),
                                             "mean_aupr": float(a[:, 1].mean()),
                                             "std_aupr": float(a[:, 1].std())}
            print(f"Dataset{ds} HistGB   AUC {a[:,0].mean():.4f}±{a[:,0].std():.4f} "
                  f"AUPR {a[:,1].mean():.4f}±{a[:,1].std():.4f}", flush=True)
        if which in ("both", "tabicl"):
            per, a = run_tabicl(files)
            report[f"dataset{ds}_tabicl"] = {"per_seed": per,
                                             "mean_auc": float(a[:, 0].mean()),
                                             "std_auc": float(a[:, 0].std()),
                                             "mean_aupr": float(a[:, 1].mean()),
                                             "std_aupr": float(a[:, 1].std())}
            print(f"Dataset{ds} TabICLv2 AUC {a[:,0].mean():.4f}±{a[:,0].std():.4f} "
                  f"AUPR {a[:,1].mean():.4f}±{a[:,1].std():.4f}", flush=True)
    json.dump(report, open(os.path.join(OUT, "deeplncrna_tabular.json"), "w"), indent=2)
    print("wrote", os.path.join(OUT, "deeplncrna_tabular.json"))
