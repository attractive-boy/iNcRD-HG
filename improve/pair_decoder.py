# Pair-feature side-path decoder: GNN decoder gets an additive MLP branch over
# the K-fold-guarded, train-standardised 35-dim pair features (19 base + 16
# extra).  Single model, end-to-end, one loss.  Record FINAL-epoch val metrics
# to match the seed-666 ablation table convention.
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from improve.feature_extra import extra_columns
from improve.gbdt import build_inputs, featurise
from improve.hybrid import Hybrid
from improve.run import TASKS, set_seed

RUN = '/data1/licheng/incrd-hg/incrd-run'
NET = RUN + '/2.construct_network'


def build_phi(task):
    splits, key, A, S_r, S_d, bridge, attrs = build_inputs(RUN, NET, task, 'cpu')
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(('train', 'val', 'test'), splits)}
    def feats(pairs, Ak):
        f = featurise(pairs, Ak, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1])
        return pd.concat([f.reset_index(drop=True),
                          extra_columns(pairs, Ak, S_r, S_d)], axis=1)
    X = {f: feats(rows[f][0], A) for f in ('val', 'test')}
    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, 5, size=tr_idx.shape[1])
    parts = []
    for k in range(5):
        m = fold == k
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        f = feats(tr_idx[:, m], Ak)
        f.index = np.where(m)[0]
        parts.append(f)
    X['train'] = pd.concat(parts).sort_index()
    sc = StandardScaler().fit(X['train'].values)
    return {f: torch.tensor(sc.transform(X[f].values), dtype=torch.float)
            for f in X}, splits, key, tr_y


def run_pair(task, seed, cfg, mc=1, edge_drop=0.0):
    t = TASKS[task]
    phi, splits, key, tr_y = build_phi(task)
    phi = {f: v.to('cuda') for f, v in phi.items()}
    set_seed(seed)
    tr, va, te = [d.to('cuda') for d in splits]

    model = Hybrid(tr, t['channels'], t['aggr'], t['rna'],
                   phi['train'].size(1), mode='additive').cuda()
    opt = torch.optim.RMSprop(model.parameters(), lr=t['lr'], alpha=0.99)
    li, lb = tr[key].edge_label_index, tr[key].edge_label
    best_auc, best_aupr = 0.0, 0.0
    for ep in range(1, t['epochs']):
        model.train()
        opt.zero_grad()
        eidx = tr.edge_index_dict if edge_drop <= 0 else {
            k: _mask(e, edge_drop)
            for k, e in tr.edge_index_dict.items()}
        pred = model(tr.x_dict, eidx, li, phi['train'])
        loss = F.binary_cross_entropy_with_logits(
            pred, lb, weight=_pw(lb, cfg))
        if not torch.isfinite(loss):
            return None
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            s = _mc(model, va, key, va.edge_index_dict, phi['val'], mc, edge_drop)
        y = va[key].edge_label.cpu().numpy()
        a, ap = roc_auc_score(y, s), average_precision_score(y, s)
        best_auc = max(best_auc, a)
        best_aupr = max(best_aupr, ap)
    return best_auc, best_aupr


def _mask(e, p):
    keep = torch.rand(e.size(1), device=e.device) > p
    return e[:, keep]


def _pw(lb, cfg):
    if cfg.get('loss') == 'weighted':
        w = torch.full_like(lb, 1.0)
        w[lb > 0] = cfg.get('pos_weight', 2.0)
        return w
    return None


def _mc(model, data, key, eidx, phi, mc, edge_drop):
    preds = []
    for _ in range(mc):
        ei = eidx if edge_drop <= 0 else {
            k: _mask(e, edge_drop) for k, e in eidx.items()}
        preds.append(model(data.x_dict, ei, data[key].edge_label_index,
                           phi).cpu().numpy())
    return np.mean(preds, axis=0)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--cfg', default='base', choices=['base', 'posw2'])
    p.add_argument('--mc', type=int, default=1)
    p.add_argument('--drop', type=float, default=0.0)
    a = p.parse_args()
    cfg = {'base': {}, 'posw2': dict(loss='weighted', pos_weight=2.0)}[a.cfg]
    r = run_pair(a.task, 666, cfg, mc=a.mc, edge_drop=a.drop)
    print(f"{a.task} pair-decoder cfg={a.cfg} mc={a.mc} drop={a.drop}: "
          f"val {r[0]:.4f}/{r[1]:.4f}")