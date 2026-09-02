# Score a chosen configuration on the paper's ablation table (Table 2).
#
# `improve/hpsearch.py` selects a configuration on the validation fold, which is
# the author's own selection rule (`find_best_params.py` maximises
# (val AUC + val AUPR)/2).  This scores that configuration properly:
#
#   * every seed, so the number is a distribution and not a draw;
#   * both folds and both metrics, so a validation-fold gain cannot hide a
#     test-fold loss;
#   * as a 10-seed logit ensemble as well as a single model -- the ensemble is
#     a fixed, reproducible predictor, unlike "the best seed";
#   * on the ablation variants too, so the whole table moves under one
#     protocol and the margins are re-derived rather than mixed across configs.
#
# Variants are `ablation_predict_association.py`'s, verbatim:
#   full          -- the released heterogeneous graph
#   wo_lncmi      -- drop the other ncRNA node type and every edge touching it
#                    (`extract_subgraph`), which removes the LncMi edges
#   wo_attribute  -- replace expression / ChemBERTa features with one-hot
#                    (`node_onehot`)
#   wo_both       -- both, which the code supports and the paper does not report
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.model import Model
from improve.run import ARMS, TASKS, compute_loss, fwd, set_seed

# Paper Table 2 (S_val) and Table 3 (S_test), for the full model.
PAPER = {'LncDrug': dict(val=(0.9834, 0.9545), test=(0.9857, 0.9585)),
         'MiDrug': dict(val=(0.9555, 0.9183), test=(0.9564, 0.9165))}
# Paper Table 2, the ablation rows.
PAPER_ABL = {
    ('LncDrug', 'wo_lncmi'): (0.9729, 0.9242),
    ('LncDrug', 'wo_attribute'): (0.9699, 0.9282),
    ('MiDrug', 'wo_lncmi'): (0.9518, 0.9098),
    ('MiDrug', 'wo_attribute'): (0.9492, 0.9062),
}


def extract_subgraph(splits, rna_type):
    """ablation_predict_association.py:88 -- keep only {rna_type, drug}."""
    keep = {rna_type, 'drug'}
    out = []
    for d in splits:
        mask = {nt: (torch.ones if nt in keep else torch.zeros)(
            d[nt].x.size(0), dtype=torch.bool) for nt in d.node_types}
        out.append(d.subgraph(mask))
    return out


def node_onehot(splits):
    """ablation_predict_association.py:106 -- identity features per node type."""
    eye = {}
    for nt in splits[0].node_types:
        n = splits[0][nt].x.size(0)
        eye[nt] = F.one_hot(torch.arange(n), num_classes=n).float()
    for d in splits:
        for nt in d.node_types:
            d[nt].x = eye[nt].clone()
    return splits


def load_variant(run_dir, task, rna_type, variant, device):
    splits = [torch.load(f'{run_dir}/3.heter_data/{task}_{s}_data.pth',
                         map_location='cpu')
              for s in ('train', 'val', 'test')]
    # Order matters and follows the released script: one-hot first, then the
    # subgraph, so `wo_both` gets one-hot features on the reduced graph.
    if variant in ('wo_attribute', 'wo_both'):
        splits = node_onehot(splits)
    if variant in ('wo_lncmi', 'wo_both'):
        splits = extract_subgraph(splits, rna_type)
    return [d.to(device) for d in splits]


@torch.no_grad()
def scores(model, data, key):
    model.eval()
    return fwd(model, data, data[key].edge_label_index).cpu().numpy()


def capture(splits, task, arm, lr, channels, aggr, epochs, seed, device,
            rule='best_val'):
    """Train one model; return val/test score vectors at the selected epoch.

    `best_val` reads each metric at its own best-validation epoch -- the rule
    improve/run.py and improve/grid.py use, and the one the released printout
    tracks.  `last` reads the final epoch, which is what
    find_best_params.py and ablation_predict_association.py actually do; the
    paper's ablation rows are only comparable under it.
    """
    cfg = ARMS[arm]
    t = TASKS[task]
    key = (t['rna'], task, 'drug')
    set_seed(seed)
    train_data, val_data, test_data = splits

    model = Model(train_data, channels, aggr, t['rna'],
                  decoder=cfg.get('decoder', 'concat'),
                  encoder=cfg.get('encoder', 'hetero')).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)

    li = train_data[key].edge_label_index
    lb = train_data[key].edge_label
    pos_index = li[:, lb == 1]
    n_pos = pos_index.size(1)
    n_src, n_dst = train_data[t['rna']].num_nodes, train_data['drug'].num_nodes

    y = {f: d[key].edge_label.cpu().numpy()
         for f, d in (('val', val_data), ('test', test_data))}
    best = {'auc': -1.0, 'aupr': -1.0}
    cap = {'auc': None, 'aupr': None}
    for _ in range(1, epochs):
        model.train()
        if cfg.get('neg_resample'):
            neg = negative_sampling(
                pos_index, num_nodes=(n_src, n_dst),
                num_neg_samples=cfg.get('neg_ratio', 2) * n_pos,
                method='sparse')
            cur_i = torch.cat([pos_index, neg], dim=1)
            cur_l = torch.cat([torch.ones(n_pos, device=device),
                               torch.zeros(neg.size(1), device=device)])
        else:
            cur_i, cur_l = li, lb
        opt.zero_grad()
        loss = compute_loss(fwd(model, train_data, cur_i), cur_l, cfg)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()

        s_val = scores(model, val_data, key)
        if rule == 'last':
            cap['auc'] = cap['aupr'] = {'val': s_val,
                                        'test': scores(model, test_data, key)}
            continue
        for m, v in (('auc', roc_auc_score(y['val'], s_val)),
                     ('aupr', average_precision_score(y['val'], s_val))):
            if v > best[m]:
                best[m] = v
                cap[m] = {'val': s_val, 'test': scores(model, test_data, key)}
    return cap, y


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--arm', default='posw2')
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--channels', type=int, required=True)
    p.add_argument('--aggr', required=True)
    p.add_argument('--epochs', type=int, default=301)
    p.add_argument('--variants', nargs='+', default=['full'],
                   choices=['full', 'wo_lncmi', 'wo_attribute', 'wo_both'])
    p.add_argument('--seeds', nargs='+', type=int,
                   default=[666, 0, 1, 2, 3, 42, 7, 13, 21, 100])
    p.add_argument('--rule', default='best_val', choices=['best_val', 'last'],
                   help="'last' is find_best_params.py / "
                        'ablation_predict_association.py, and is the only rule '
                        "under which the paper's ablation rows are comparable")
    p.add_argument('--out', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rna = TASKS[args.task]['rna']
    rows, per_seed = [], []

    for variant in args.variants:
        splits = load_variant(args.run_dir, args.task, rna, variant, device)
        acc = {f: {'auc': [], 'aupr': [], 'z_auc': [], 'z_aupr': []}
               for f in ('val', 'test')}
        for seed in args.seeds:
            cap, y = capture(splits, args.task, args.arm, args.lr,
                             args.channels, args.aggr, args.epochs, seed,
                             device, rule=args.rule)
            rec = dict(task=args.task, variant=variant, arm=args.arm,
                       lr=args.lr, channels=args.channels, aggr=args.aggr,
                       rule=args.rule, seed=seed)
            for f in ('val', 'test'):
                a = roc_auc_score(y[f], cap['auc'][f])
                u = average_precision_score(y[f], cap['aupr'][f])
                acc[f]['auc'].append(a)
                acc[f]['aupr'].append(u)
                acc[f]['z_auc'].append(zs(cap['auc'][f]))
                acc[f]['z_aupr'].append(zs(cap['aupr'][f]))
                rec[f'{f}_auc'], rec[f'{f}_aupr'] = a, u
            per_seed.append(rec)
            print(f"{args.task} {variant:12s} seed={seed:<4d} "
                  f"val {rec['val_auc']:.4f}/{rec['val_aupr']:.4f}  "
                  f"test {rec['test_auc']:.4f}/{rec['test_aupr']:.4f}",
                  flush=True)
            pd.DataFrame(per_seed).to_csv(
                args.out.replace('.csv', '_seeds.csv'), index=False)

        for f in ('val', 'test'):
            a, u = np.array(acc[f]['auc']), np.array(acc[f]['aupr'])
            if variant == 'full':
                pa, pp = PAPER[args.task][f]
            elif f == 'val':
                pa, pp = PAPER_ABL.get((args.task, variant), (np.nan, np.nan))
            else:
                pa = pp = np.nan
            rows.append(dict(
                task=args.task, variant=variant, arm=args.arm, fold=f,
                n=len(a), lr=args.lr, channels=args.channels, aggr=args.aggr,
                rule=args.rule,
                mean_auc=a.mean(), std_auc=a.std(), best_auc=a.max(),
                ens_auc=roc_auc_score(y[f], np.mean(acc[f]['z_auc'], axis=0)),
                mean_aupr=u.mean(), std_aupr=u.std(), best_aupr=u.max(),
                ens_aupr=average_precision_score(
                    y[f], np.mean(acc[f]['z_aupr'], axis=0)),
                paper_auc=pa, paper_aupr=pp))
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print('\n=== ' + args.task + ' ===')
    print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
