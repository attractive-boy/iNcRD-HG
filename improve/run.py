# Train iNcRD-HG variants on the released protocol and report paired deltas.
#
# Protocol: the frozen RandomLinkSplit(num_val=0.2, num_test=0.2,
# neg_sampling_ratio=2.0) folds written by heter_data_prepare.py, scored with
# global AUC / AUPR -- exactly what the paper reports.  Nothing here debiases
# the split; that is the point of this experiment.
#
# Three numbers are recorded per run because the paper's own selection rule is
# not conservative:
#   final     -- the last epoch, no selection at all
#   best_test -- the best epoch measured on test (what predict_association.py
#                prints and what a headline number is usually read off)
#   sel_val   -- the epoch chosen by validation AUC, scored on test
# Quote sel_val when the claim has to survive a reviewer; best_test is there so
# the comparison with the published 0.9857 is like-for-like.
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.features import add_degree_encoding, normalise_features
from improve.model import Model

# The author's tuned settings, from predict_association.py.
TASKS = {
    'LncDrug': dict(rna='lncRNA', lr=0.0005, epochs=101, channels=64, aggr='min'),
    'MiDrug': dict(rna='miRNA', lr=0.0001, epochs=301, channels=256, aggr='mean'),
}

# Each arm is the base architecture plus the named switches.  The cumulative
# arms stack in the order the switches were argued for.
ARMS = {
    'base': {},
    'neg': dict(neg_resample=True),
    'dec': dict(decoder='rich'),
    'deg': dict(degree_enc=True),
    'featnorm': dict(feat_norm=True),
    'neg+dec': dict(neg_resample=True, decoder='rich'),
    'neg+dec+deg': dict(neg_resample=True, decoder='rich', degree_enc=True),
    'all': dict(neg_resample=True, decoder='rich', degree_enc=True,
                feat_norm=True),
    # Relation fusion.  The released model fuses per-relation messages with a
    # fixed reduction (min on LncDrug, mean on MiDrug); `relattn` learns it.
    # The aggr_* arms are the controls it has to beat -- otherwise a win is
    # just "min was a bad choice", which is a hyperparameter, not a mechanism.
    'relattn': dict(encoder='relattn'),
    'aggr_sum': dict(aggr='sum'),
    'aggr_mean': dict(aggr='mean'),
    'aggr_max': dict(aggr='max'),
    # Joint supervision from the other ncRNA relation.  LncDrug has 1267
    # training positives against MiDrug's 5394, and the two share every drug
    # node, so the other task's edges can sharpen the drug embeddings that
    # LncDrug's decoder reads.  The cross-task DEGREE failed
    # (improve/cross_task_prior.py); cross-task supervision is a different
    # mechanism and is tested separately here.
    'multitask': dict(multitask=True),
    'multitask+neg': dict(multitask=True, neg_resample=True),
    # Methods from the 2022-2025 ncRNA-drug literature, reimplemented.
    # LightGCN (GSLRDA's backbone) is the strongest bet on this protocol: no
    # feature transform, no nonlinearity, symmetric-normalised propagation --
    # the closest thing to a smoothed degree model, and 99.7% of the metric is
    # a degree comparison.  Reliable negatives are DMGAT / LSNSCDA.
    'lightgcn': dict(encoder='lightgcn'),
    'lightgcn_mlp': dict(encoder='lightgcn', lgn_decoder='mlp'),
    'lightgcn+neg': dict(encoder='lightgcn', neg_resample=True),
    'relneg': dict(reliable_neg=True),
    'lightgcn+relneg': dict(encoder='lightgcn', reliable_neg=True),
}

OTHER_TASK = {'LncDrug': ('MiDrug', 'miRNA'), 'MiDrug': ('LncDrug', 'lncRNA')}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_splits(run_dir, task, cfg, device):
    splits = [
        torch.load(f'{run_dir}/3.heter_data/{task}_{s}_data.pth',
                   map_location='cpu')
        for s in ('train', 'val', 'test')
    ]
    if cfg.get('feat_norm'):
        splits = normalise_features(splits)
    if cfg.get('degree_enc'):
        splits = add_degree_encoding(splits)
    return [d.to(device) for d in splits]


def fwd(model, data, idx):
    # LightGCN needs the graph itself to build its normalised adjacency.
    if model.__class__.__name__ == 'LightGCN':
        return model(data.x_dict, data.edge_index_dict, idx, data=data)
    return model(data.x_dict, data.edge_index_dict, idx)


@torch.no_grad()
def evaluate(model, data, key):
    model.eval()
    pred = fwd(model, data, data[key].edge_label_index)
    y = data[key].edge_label.cpu().numpy()
    s = pred.cpu().numpy()
    return roc_auc_score(y, s), average_precision_score(y, s)


def run_one(run_dir, task, arm, seed, device):
    cfg = ARMS[arm]
    t = TASKS[task]
    key = (t['rna'], task, 'drug')
    set_seed(seed)
    train_data, val_data, test_data = load_splits(run_dir, task, cfg, device)

    model = Model(train_data, t['channels'], cfg.get('aggr', t['aggr']),
                  t['rna'], decoder=cfg.get('decoder', 'concat'),
                  encoder=cfg.get('encoder', 'hetero')).to(device)
    aux = None
    if cfg.get('multitask'):
        # A second decoder head on the other ncRNA relation, sharing the
        # encoder.  Its positives are that relation's edges as they sit in this
        # task's graph -- a different node type entirely, so no target-task
        # test edge can appear among them.
        o_task, o_rna = OTHER_TASK[task]
        from improve.model import EdgeDecoder
        aux_dec = EdgeDecoder(t['channels'], o_rna,
                              mode=cfg.get('decoder', 'concat')).to(device)
        model.aux_decoder = aux_dec
        aux = dict(task=o_task, rna=o_rna,
                   pos=train_data[o_rna, o_task, 'drug'].edge_index,
                   n_src=train_data[o_rna].num_nodes,
                   n_dst=train_data['drug'].num_nodes)
    opt = torch.optim.RMSprop(model.parameters(), lr=t['lr'], alpha=0.99)

    # Fixed training labels as shipped by RandomLinkSplit.
    label_index = train_data[key].edge_label_index
    label = train_data[key].edge_label
    pos_index = label_index[:, label == 1]
    n_pos = pos_index.size(1)
    n_src = train_data[t['rna']].num_nodes
    n_dst = train_data['drug'].num_nodes

    history = []
    for epoch in range(1, t['epochs']):
        model.train()
        if cfg.get('neg_resample'):
            # Fresh negatives every epoch at the protocol's 2:1 ratio.
            # Excluded set is the training positives only -- excluding val/test
            # positives too would leak the held-out edges into training.
            neg_index = negative_sampling(
                pos_index, num_nodes=(n_src, n_dst),
                num_neg_samples=2 * n_pos, method='sparse')
            cur_index = torch.cat([pos_index, neg_index], dim=1)
            cur_label = torch.cat([
                torch.ones(n_pos, device=device),
                torch.zeros(neg_index.size(1), device=device)])
        elif cfg.get('reliable_neg') and epoch > 20:
            # DMGAT / LSNSCDA: keep the candidates the current model scores
            # most confidently negative.  Warm up on the shipped negatives
            # first, otherwise the selector is untrained noise.
            from improve.sota import reliable_negatives
            model.eval()
            neg_index = reliable_negatives(
                lambda c: fwd(model, train_data, c), pos_index,
                n_src, n_dst, 2 * n_pos, device)
            model.train()
            cur_index = torch.cat([pos_index, neg_index], dim=1)
            cur_label = torch.cat([
                torch.ones(n_pos, device=device),
                torch.zeros(neg_index.size(1), device=device)])
        else:
            cur_index, cur_label = label_index, label

        opt.zero_grad()
        pred = fwd(model, train_data, cur_index)
        loss = F.binary_cross_entropy_with_logits(pred, cur_label)
        if aux is not None:
            z_dict = model.encoder(train_data.x_dict, train_data.edge_index_dict)
            a_neg = negative_sampling(
                aux['pos'], num_nodes=(aux['n_src'], aux['n_dst']),
                num_neg_samples=2 * aux['pos'].size(1), method='sparse')
            a_idx = torch.cat([aux['pos'], a_neg], dim=1)
            a_lbl = torch.cat([torch.ones(aux['pos'].size(1), device=device),
                               torch.zeros(a_neg.size(1), device=device)])
            loss = loss + F.binary_cross_entropy_with_logits(
                model.aux_decoder(z_dict, a_idx), a_lbl)
        loss.backward()
        opt.step()

        val_auc, val_aupr = evaluate(model, val_data, key)
        test_auc, test_aupr = evaluate(model, test_data, key)
        history.append((epoch, float(loss), val_auc, val_aupr,
                        test_auc, test_aupr))

    h = pd.DataFrame(history, columns=['epoch', 'loss', 'val_auc', 'val_aupr',
                                       'test_auc', 'test_aupr'])
    best_val = h.loc[h['val_auc'].idxmax()]
    best_test = h.loc[h['test_auc'].idxmax()]
    last = h.iloc[-1]
    return dict(
        task=task, arm=arm, seed=seed,
        final_auc=last['test_auc'], final_aupr=last['test_aupr'],
        best_test_auc=best_test['test_auc'], best_test_aupr=best_test['test_aupr'],
        sel_val_epoch=int(best_val['epoch']),
        sel_val_auc=best_val['test_auc'], sel_val_aupr=best_val['test_aupr'],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--arms', nargs='+', default=['base'])
    p.add_argument('--seeds', nargs='+', type=int,
                   default=[666, 0, 1, 2, 3, 42, 7, 13, 21, 100])
    p.add_argument('--out', required=True)
    p.add_argument('--lr', type=float, default=None,
                   help='override the task lr (arms other than base were '
                        'otherwise judged on hyperparameters tuned for base)')
    p.add_argument('--epochs', type=int, default=None)
    args = p.parse_args()

    for t in TASKS.values():
        if args.lr is not None:
            t['lr'] = args.lr
        if args.epochs is not None:
            t['epochs'] = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for task in args.tasks:
        for arm in args.arms:
            for seed in args.seeds:
                r = run_one(args.run_dir, task, arm, seed, device)
                r['lr'] = TASKS[task]['lr']
                r['epochs'] = TASKS[task]['epochs']
                rows.append(r)
                print(f"{task:8s} {arm:12s} seed={seed:<4d} "
                      f"final={r['final_auc']:.4f} "
                      f"best_test={r['best_test_auc']:.4f} "
                      f"sel_val={r['sel_val_auc']:.4f}", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
