"""Shared same-protocol harness for external baseline reproduction.

Protocol (identical to improve/research/external_specialized_20260916 ours-ncDR run,
see /tmp/opencode/exp_final.py):
  - ncDR miRNA-drug matrix: 701 miRNAs x 101 drugs, 3163 associations
  - per seed: RandomLinkSplit(test=0.2) over the bipartite edges
  - global AUC / AUPR on held-out positives + the SAME number of uniformly
    sampled negatives (np.random.default_rng(seed+7)).
"""
import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

DATA = "/tmp/opencode/comp/GAMMDR"
NM, ND = 701, 101
SEEDS = [42, 0, 1, 2, 3]


def load_ncdr():
    a = pd.read_csv(f"{DATA}/data_3000.csv", header=None)
    mir = sorted(set(a[0]))
    drug = sorted(set(a[1]))
    mi2i = {m: i for i, m in enumerate(mir)}
    dr2i = {d: i for i, d in enumerate(drug)}
    pairs = [(mi2i[m], dr2i[d]) for m, d in a[[0, 1]].values]
    return pairs, mir, drug


def _build_adj(pairs):
    adj = torch.LongTensor([[m, d + NM] for m, d in pairs]).T
    return adj


def build_split(pairs, seed):
    """Return (train_cross, test_pos_cross) as (2, n) int arrays in separate
    miRNA / drug index spaces."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    adj = _build_adj(pairs)
    feat = torch.zeros(NM + ND, 1)
    tr, _, te = RandomLinkSplit(
        num_val=0, num_test=0.2, is_undirected=True, split_labels=True,
        add_negative_train_samples=False)(Data(x=feat, edge_index=adj))

    def only3(ei):
        e = ei.cpu().numpy()
        e = e[:, (e[0] < NM) & (e[1] >= NM)]
        return np.stack([e[0], e[1] - NM])

    msg = only3(tr.edge_index)
    tpos = only3(te.pos_edge_label_index)
    return msg, tpos


def sample_neg(pos_set, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        i = int(rng.integers(0, NM))
        j = int(rng.integers(0, ND))
        if (i, j) not in pos_set:
            out.append((i, j))
    return np.array(out).T


def ncdr_pos_set(pairs):
    return set((int(m), int(d)) for m, d in pairs)


def evaluate(pos_scores, neg_scores):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    s = np.r_[pos_scores, neg_scores]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def full_metrics(pos_scores, neg_scores):
    """AUC/AUPR + threshold metrics at the best-F1 threshold."""
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 matthews_corrcoef, precision_recall_curve,
                                 roc_auc_score)
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    s = np.r_[pos_scores, neg_scores]
    prec, rec, ths = precision_recall_curve(y, s)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    thr = float(ths[int(np.argmax(f1[:-1]))]) if len(ths) else 0.5
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(AUC=float(roc_auc_score(y, s)), AUPR=float(average_precision_score(y, s)),
                ACC=(tp + tn) / len(y), Precision=tp / (tp + fp + 1e-12),
                Recall=tp / (tp + fn + 1e-12), Specificity=tn / (tn + fp + 1e-12),
                F1=2 * tp / (2 * tp + fp + fn + 1e-12), MCC=float(matthews_corrcoef(y, pred)),
                thr=thr)


def summarize(name, aucs, auprs, out_csv=None):
    aucs = np.asarray(aucs, dtype=float)
    auprs = np.asarray(auprs, dtype=float)
    row = dict(method=name, n_seeds=len(aucs),
               auc_mean=aucs.mean(), auc_std=aucs.std(),
               aupr_mean=auprs.mean(), aupr_std=auprs.std())
    print(f"[{name}] AUC {aucs.mean():.4f}+-{aucs.std():.4f}  "
          f"AUPR {auprs.mean():.4f}+-{auprs.std():.4f}")
    if out_csv:
        pd.DataFrame([row]).to_csv(out_csv, index=False)
    return row
