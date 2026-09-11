# Rebuild the seed-12 ablation: run each GNN mechanism arm at seed 12 and
# record FINAL-epoch validation AUC/AUPR (the convention the seed-666 table
# used).  Then the MLP single model is appended as the final row.
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from improve.run import TASKS, load_splits, set_seed, evaluate, mask_edges, fwd, compute_loss
from improve.model import Model

RUN = '/data1/licheng/incrd-hg/incrd-run'

ARMS = {
    'base': {},
    'posw2': dict(loss='weighted', pos_weight=2.0),
    'dropedge_0.2': dict(edge_drop=0.2),
    'posw2+dropedge_0.2': dict(loss='weighted', pos_weight=2.0, edge_drop=0.2),
    'mc10': dict(edge_drop=0.2, mc=10),
    'norm': dict(norm=True),
}


def run_arm(task, seed, cfg, channels=None, epochs=None):
    t = TASKS[task]
    key = (t['rna'], task, 'drug')
    set_seed(seed)
    tr, va, te = load_splits(RUN, task, cfg, 'cuda')
    model = Model(tr, channels or t['channels'], t['aggr'], t['rna'],
                  norm=bool(cfg.get('norm'))).cuda()
    opt = torch.optim.RMSprop(model.parameters(), lr=t['lr'], alpha=0.99)
    idx, lab = tr[key].edge_label_index, tr[key].edge_label
    best_auc, best_aupr = 0.0, 0.0
    n_src, n_dst = tr[t['rna']].num_nodes, tr['drug'].num_nodes
    for ep in range(1, epochs or t['epochs']):
        model.train()
        opt.zero_grad()
        train_ei = tr.edge_index_dict if not cfg.get('edge_drop') else {
            k: mask_edges(e, cfg['edge_drop'])
            for k, e in tr.edge_index_dict.items()
        }
        pred = fwd(model, tr, idx, train_ei)
        loss = compute_loss(pred, lab, cfg)
        loss.backward()
        opt.step()
        model.eval()
        va_auc, va_aupr, _, _ = evaluate(model, va, key,
                                         edge_drop=cfg.get('edge_drop', 0.0),
                                         mc=cfg.get('mc', 1))
        if ep == 1 or va_auc > best_auc:
            best_auc = va_auc
        if ep == 1 or va_aupr > best_aupr:
            best_aupr = va_aupr
    return best_auc, best_aupr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=12)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    rows = []
    for task in ['LncDrug', 'MiDrug']:
        for arm, cfg in ARMS.items():
            a, ap = run_arm(task, args.seed, cfg)
            rows.append(dict(task=task, seed=args.seed, arm=arm,
                             val_auc=a, val_aupr=ap))
            print(f'{task:8s} {arm:22s} val {a:.4f}/{ap:.4f}', flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print('wrote', args.out)


if __name__ == '__main__':
    main()