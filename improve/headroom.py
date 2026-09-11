# Where does the remaining error live, and is the drug prior orthogonal to the model?
#
# Two questions the arm sweep could not answer:
#
# 1. The degree-only scorer reaches 0.9775 and the model 0.9866, but their
#    scores correlate at rho~0.84 -- they do not fail on the same edges.  A
#    blend `model + alpha * log1p(drug_degree)`, with alpha chosen on the
#    validation fold, can in principle beat both.  This is the one lever that
#    works *with* the protocol instead of against it.
# 2. If the residual errors sit on low-degree drugs there is ncRNA-side signal
#    left to capture; if they are spread evenly across the degree range, the
#    remaining gap is label noise and the protocol really is at its ceiling.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import negative_sampling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.run import (ARMS, TASKS, compute_loss, evaluate, fwd, mask_edges,
                        load_splits, set_seed)
from improve.model import Model


def train_capture(run_dir, task, seed, device, arm='base'):
    """Train `arm`; return val/test scores at the val-selected epoch."""
    cfg = ARMS[arm]
    t = TASKS[task]
    key = (t['rna'], task, 'drug')
    set_seed(seed)
    train_data, val_data, test_data = load_splits(run_dir, task, cfg, device)
    model = Model(train_data, t['channels'], cfg.get('aggr', t['aggr']),
                  t['rna'], decoder=cfg.get('decoder', 'concat'),
                  encoder=cfg.get('encoder', 'hetero'),
                  norm=bool(cfg.get('norm')),
                  residual=bool(cfg.get('res'))).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=t['lr'], alpha=0.99,
                              weight_decay=cfg.get('wd', 0))
    if cfg.get('optimizer') == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=t['lr'],
                                weight_decay=cfg.get('wd', 0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=t['epochs']) if cfg.get('cosine') else None
    li = train_data[key].edge_label_index
    lb = train_data[key].edge_label
    pos_index = li[:, lb == 1]
    n_pos = pos_index.size(1)
    n_src, n_dst = train_data[t['rna']].num_nodes, train_data['drug'].num_nodes

    best_val, cap = -1.0, None
    best_pr, cap_pr = -1.0, None
    for epoch in range(1, t['epochs']):
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
        drop = cfg.get('edge_drop')
        if drop and cfg.get('drop_start'):
            frac = min(1.0, epoch / max(1, cfg.get('anneal_epochs', 50)))
            drop = drop + (cfg['drop_start'] - drop) * (1 - frac)
        train_edge_index = train_data.edge_index_dict if not drop else {
            k: mask_edges(edge_index, drop)
            for k, edge_index in train_data.edge_index_dict.items()
        }
        tr_x = None
        if cfg.get('feat_drop'):
            tr_x = {k: x * (torch.rand(x.shape, device=x.device)
                            > cfg['feat_drop']).float()
                    for k, x in train_data.x_dict.items()}
        # Must go through compute_loss, not a hard-coded BCE: the loss-shaping
        # arms (posw*, focal, bpr) are defined entirely by this call.
        compute_loss(fwd(model, train_data, cur_i, train_edge_index, tr_x),
                     cur_l, cfg).backward()
        opt.step()
        if sched is not None:
            sched.step()
        mc = cfg.get('mc', 1)
        drop_r = cfg.get('edge_drop', 0.0)
        va, vp, _, _ = evaluate(
            model, val_data, key, drop_r if mc > 1 else 0.0, mc,
            cfg.get('mc_rate'), bool(cfg.get('mc_full')))
        if vp > best_pr:
            best_pr = vp
            model.eval()
            with torch.no_grad():
                cap_pr = {n: model(d.x_dict, d.edge_index_dict,
                                   d[key].edge_label_index).cpu().numpy()
                          for n, d in (('val', val_data), ('test', test_data))}
        if va > best_val:
            best_val = va
            model.eval()
            with torch.no_grad():
                cap = {}
                for name, d in (('val', val_data), ('test', test_data)):
                    cap[name] = model(d.x_dict, d.edge_index_dict,
                                      d[key].edge_label_index).cpu().numpy()
    # Drug training degree, from the training message-passing edges only.
    deg = torch.zeros(train_data['drug'].num_nodes)
    di = train_data[key].edge_index[1].cpu()
    deg.scatter_add_(0, di, torch.ones_like(di, dtype=torch.float))
    out = {}
    for name, d in (('val', val_data), ('test', test_data)):
        out[name] = dict(
            score=cap[name],
            # Same model, epoch chosen by validation AUPR instead of AUC.  A
            # best-epoch table selects each metric on its own curve; keeping
            # both makes that explicit rather than implicit.
            score_pr=cap_pr[name],
            y=d[key].edge_label.cpu().numpy(),
            drug=d[key].edge_label_index[1].cpu().numpy())
    return out, deg.numpy()


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alphas = np.concatenate([[0.0], np.linspace(0.05, 3.0, 60)])
    rows, diag = [], []

    for task in args.tasks:
        for seed in args.seeds:
            cap, deg = train_capture(args.run_dir, task, seed, device)
            prior = {k: np.log1p(deg[cap[k]['drug']]) for k in cap}

            # alpha selected on val, applied once to test
            zs = {k: zscore(cap[k]['score']) for k in cap}
            zp = {k: zscore(prior[k]) for k in cap}
            val_auc = [roc_auc_score(cap['val']['y'], zs['val'] + a * zp['val'])
                       for a in alphas]
            a_star = float(alphas[int(np.argmax(val_auc))])
            blend = zs['test'] + a_star * zp['test']
            rows.append(dict(
                task=task, seed=seed, alpha=a_star,
                model_auc=roc_auc_score(cap['test']['y'], zs['test']),
                model_aupr=average_precision_score(cap['test']['y'], zs['test']),
                prior_auc=roc_auc_score(cap['test']['y'], zp['test']),
                blend_auc=roc_auc_score(cap['test']['y'], blend),
                blend_aupr=average_precision_score(cap['test']['y'], blend),
                spearman=float(pd.Series(zs['test']).corr(
                    pd.Series(zp['test']), method='spearman'))))
            print(f"{task} seed={seed} alpha={a_star:.2f} "
                  f"model={rows[-1]['model_auc']:.4f} "
                  f"prior={rows[-1]['prior_auc']:.4f} "
                  f"blend={rows[-1]['blend_auc']:.4f}", flush=True)

            # Where the model's residual error lives, by drug training degree.
            t = cap['test']
            d = pd.DataFrame(dict(y=t['y'], s=zs['test'],
                                  deg=deg[t['drug']], drug=t['drug']))
            d['bin'] = pd.qcut(d['deg'], 4, labels=False, duplicates='drop')
            for b, g in d.groupby('bin', observed=True):
                if g['y'].nunique() < 2:
                    continue
                diag.append(dict(task=task, seed=seed, bin=str(b), n=len(g),
                                 pos=int(g['y'].sum()),
                                 auc=roc_auc_score(g['y'], g['s'])))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.DataFrame(diag).to_csv(args.out.replace('.csv', '_bydeg.csv'), index=False)
    print('\n=== blend vs model (paired) ===')
    df['delta'] = df['blend_auc'] - df['model_auc']
    print(df.groupby('task')[['model_auc', 'prior_auc', 'blend_auc', 'delta',
                              'alpha', 'spearman']].mean()
          .to_string(float_format=lambda v: f'{v:.4f}'))
    print('\n=== residual error by drug training degree ===')
    print(pd.DataFrame(diag).groupby(['task', 'bin'], observed=True)[['auc', 'n', 'pos']]
          .mean().to_string(float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
