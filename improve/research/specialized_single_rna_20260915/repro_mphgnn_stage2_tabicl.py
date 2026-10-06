import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tabicl import TabICLClassifier

CKPT = "/data1/licheng/incrd-hg/.external/models/tabicl/tabicl-classifier-v2-20260212.ckpt"
z = np.load("/tmp/opencode/mphgnn_pair_emb.npz")
Xtr, ytr = z["X_train"], z["y_train"].astype(np.int64)
Xte, yte = z["X_test"], z["y_test"]
print("data", Xtr.shape, Xte.shape, "pos", int(ytr.sum()))

for seed in (666, 0, 1, 2, 3):
    clf = TabICLClassifier(n_estimators=8, random_state=seed, model_path=CKPT,
                           checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
                           allow_auto_download=False, device="cuda:0", batch_size=4)
    clf.fit(Xtr, ytr)
    s = np.asarray(clf.predict_proba(Xte))[:, 1]
    print("TabICL seed %d: AUC %.4f  AUPR %.4f" % (seed, roc_auc_score(yte, s), average_precision_score(yte, s)), flush=True)
