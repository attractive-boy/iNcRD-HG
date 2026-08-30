# Does averaging scores across seeds beat a single model on the released protocol?
#
# The cheapest remaining lever and the one least likely to interact with the
# architecture: train N independent inits, average their test logits at the
# validation-selected epoch, score once.  Reported against the mean single-seed
# result over the same N runs, so the comparison is ensemble-vs-its-own-members
# rather than ensemble-vs-a-different-seed-list.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.run import ARMS, TASKS, evaluate, load_splits, set_seed
from improve.model import Model


def train_capture(run_dir, task, arm, seed, device):
    """Train one model; return its test scores at the val-selected epoch."""
    cfg = ARMS[arm]
    t = TASKS[task]
    key = (t['rna'], task, 'drug')
    set_seed(seed)
    train_data, val_data, test_data = load_splits(run_dir, task, cfg, device)
    model = Model(train_data, t['channels'], t['aggr'], t['rna'],
                  decoder=cfg.get('decoder', 'concat')).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=t['lr'], alpha=0.99)
    label_index = train_data[key].edge_label_index
    label = train_data[key].edge_label

    best_val, best_scores = -1.0, None
    for _ in range(1, t['epochs']):
        model.train()
        opt.zero_grad()
        pred = model(train_data.x_dict, train_data.edge_index_dict, label_index)
        F.binary_cross_entropy_with_logits(pred, label).backward()
        opt.step()
        val_auc, _ = evaluate(model, val_data, key)
        if val_auc > best_val:
            best_val = val_auc
            with torch.no_grad():
                model.eval()
                s = model(test_data.x_dict, test_data.edge_index_dict,
                          test_data[key].edge_label_index)
            best_scores = s.cpu().numpy()
    y = test_data[key].edge_label.cpu().numpy()
    return best_scores, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--arm', default='base')
    p.add_argument('--seeds', nargs='+', type=int,
                   default=[666, 0, 1, 2, 3, 42, 7, 13, 21, 100])
    p.add_argument('--out', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for task in args.tasks:
        scores, y = [], None
        for seed in args.seeds:
            s, y = train_capture(args.run_dir, task, args.arm, seed, device)
            scores.append(s)
            print(f'{task} seed={seed} auc={roc_auc_score(y, s):.4f}', flush=True)
        singles = [roc_auc_score(y, s) for s in scores]
        singles_pr = [average_precision_score(y, s) for s in scores]
        # Average the logits, not the probabilities: the decoder's output is
        # pre-sigmoid and AUC only cares about the induced ranking.
        for k in (3, 5, len(scores)):
            ens = np.mean(scores[:k], axis=0)
            rows.append(dict(
                task=task, arm=args.arm, n_models=k,
                single_auc=float(np.mean(singles[:k])),
                single_auc_std=float(np.std(singles[:k])),
                single_aupr=float(np.mean(singles_pr[:k])),
                ens_auc=roc_auc_score(y, ens),
                ens_aupr=average_precision_score(y, ens)))
        pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    df['d_auc'] = df['ens_auc'] - df['single_auc']
    df['d_aupr'] = df['ens_aupr'] - df['single_aupr']
    print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
