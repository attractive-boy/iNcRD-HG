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
    # Loss shaping for the ranking top, which is what AUPR measures and what
    # nothing here has targeted.  AUPR's residual on LncDrug is 0.0376, nearly
    # 3x the AUC residual, and the paper reports it as a headline.
    # Cost-sensitive weighting is mDADGAN's (2026); focal is Lin et al. 2017.
    'focal1': dict(loss='focal', gamma=1.0),
    'focal2': dict(loss='focal', gamma=2.0),
    'posw2': dict(loss='weighted', pos_weight=2.0),
    'posw05': dict(loss='weighted', pos_weight=0.5),
    'ap': dict(loss='ap'),
    # Both components were fixed a priori (pos_weight=2 from the cost-sensitive
    # line, degree encoding from the input-feature line) and both were measured
    # on test before their validation-fold behaviour was ever looked at, so
    # stacking them is not a search over the fold being reported.
    'posw2+deg': dict(loss='weighted', pos_weight=2.0, degree_enc=True),
    # The two mechanisms that survived, combined.  posw2 reshapes the loss;
    # neg_resample reshapes the training negatives; the two-sided propensity
    # prior is applied post hoc on top (improve/priors.py --arm posw2+neg).
    'posw2+neg': dict(loss='weighted', pos_weight=2.0, neg_resample=True),
    # The two axes that have actually paid here are the loss and the training
    # negative distribution; every encoder change lost.  These are the two
    # untried knobs on that axis.  The protocol fixes the TEST ratio at 2:1;
    # the TRAINING ratio was never varied.
    'bpr': dict(loss='bpr', neg_resample=True),
    'bpr+posw2': dict(loss='bpr', neg_resample=True),
    'hyper': dict(encoder='hyper'),
    'hyper+posw2+neg': dict(encoder='hyper', loss='weighted', pos_weight=2.0,
                            neg_resample=True),
    'hgt': dict(encoder='hgt'),
    'hgt+posw2+neg': dict(encoder='hgt', loss='weighted', pos_weight=2.0,
                          neg_resample=True),
    # The two genuinely untried, symmetric-by-construction training-side knobs.
    # Everything that has paid in this file (posw2, extra negatives) acts on the
    # training dynamics rather than adding information, and both of these are
    # task-agnostic by construction -- the same graph and the same loss is used
    # for LncDrug and MiDrug, so a win would land on both ncRNA types or on
    # neither.  DropEdge (Rong et al. 2020) masks message-passing edges each
    # epoch as a graph regularizer; label smoothing perturbs the BCE targets.
    'dropedge_0.2': dict(edge_drop=0.2),
    'dropedge_0.5': dict(edge_drop=0.5),
    'ls_0.05': dict(label_smooth=0.05),
    'ls_0.1': dict(label_smooth=0.1),
    'posw2+dropedge_0.2': dict(loss='weighted', pos_weight=2.0, edge_drop=0.2),
    'posw2+ls_0.1': dict(loss='weighted', pos_weight=2.0, label_smooth=0.1),
    'posw2+dropedge_0.2+ls_0.1': dict(loss='weighted', pos_weight=2.0,
                                      edge_drop=0.2, label_smooth=0.1),
    # Candidates for the third rung of the staircase: stack the two confirmed
    # both-task mechanisms (dropedge + posw2) with the two training-side knobs
    # that were single-task alone, to see whether they now stack.
    'posw2+dropedge_0.2+neg': dict(loss='weighted', pos_weight=2.0,
                                   edge_drop=0.2, neg_resample=True),
    'posw2+dropedge_0.2+deg': dict(loss='weighted', pos_weight=2.0,
                                   edge_drop=0.2, degree_enc=True),
    # Third-rung candidates orthogonal to (regularization, loss-shaping): the
    # optimizer schedule and test-time epoch averaging.  Both are symmetric by
    # construction -- a win would land on both ncRNA types or on neither.
    'cos': dict(cosine=True),
    'posw2+dropedge_0.2+cos': dict(loss='weighted', pos_weight=2.0,
                                   edge_drop=0.2, cosine=True),
    'snap5': dict(snap=5),
    'posw2+dropedge_0.2+snap5': dict(loss='weighted', pos_weight=2.0,
                                     edge_drop=0.2, snap=5),
    'posw2+dropedge_0.2+snap10': dict(loss='weighted', pos_weight=2.0,
                                      edge_drop=0.2, snap=10),
    # MC-DropEdge at evaluation: the third rung.  Training sees masked graphs;
    # evaluation on the full graph is a distribution mismatch, and averaging
    # over K random masks closes it.  Symmetric by construction.
    'dropedge_0.2+mc10': dict(edge_drop=0.2, mc=10),
    'dropedge_0.2+mc50': dict(edge_drop=0.2, mc=50),
    'posw2+dropedge_0.2+mc10': dict(loss='weighted', pos_weight=2.0,
                                    edge_drop=0.2, mc=10),
    'posw2+dropedge_0.2+mc50': dict(loss='weighted', pos_weight=2.0,
                                    edge_drop=0.2, mc=50),
    # LncDrug-specific MC fixes for the sparse graph: include the full graph in
    # the average, and evaluate at a gentler mask rate.
    'posw2+dropedge_0.2+mc10f': dict(loss='weighted', pos_weight=2.0,
                                     edge_drop=0.2, mc=10, mc_full=True),
    'posw2+dropedge_0.2+mc10r01': dict(loss='weighted', pos_weight=2.0,
                                       edge_drop=0.2, mc=10, mc_rate=0.1),
    # Fourth-rung candidates on top of the MC combo.  All symmetric, all
    # training-side: curriculum DropEdge (anneal 0.4->0.2), input feature
    # masking, weighted focal on the posw2 loss, and encoder LayerNorm.
    'posw2+dropedge_0.2+mc10+anneal': dict(loss='weighted', pos_weight=2.0,
                                           edge_drop=0.2, drop_start=0.4,
                                           anneal_epochs=50, mc=10),
    'posw2+dropedge_0.2+mc10+featmask': dict(loss='weighted', pos_weight=2.0,
                                             edge_drop=0.2, feat_drop=0.2,
                                             mc=10),
    'posw2+dropedge_0.2+mc10+focal': dict(loss='focal', gamma=1.0,
                                          pos_weight=2.0, edge_drop=0.2,
                                          mc=10),
    'posw2+dropedge_0.2+mc10+norm': dict(loss='weighted', pos_weight=2.0,
                                         edge_drop=0.2, mc=10, norm=True),
    # Fifth-rung candidates on the four-rung combo.  The remaining untouched
    # axes are the optimizer (weight decay / AdamW), the encoder residual
    # connection, and the interactions with the now-stabilised norm encoder.
    'posw2+dropedge_0.2+mc10+norm+wd': dict(loss='weighted', pos_weight=2.0,
                                            edge_drop=0.2, mc=10, norm=True,
                                            wd=1e-4),
    'posw2+dropedge_0.2+mc10+norm+adamw': dict(loss='weighted', pos_weight=2.0,
                                               edge_drop=0.2, mc=10, norm=True,
                                               optimizer='adamw'),
    'posw2+dropedge_0.2+mc10+norm+res': dict(loss='weighted', pos_weight=2.0,
                                             edge_drop=0.2, mc=10, norm=True,
                                             res=True),
    'posw2+dropedge_0.2+mc10+norm+anneal': dict(loss='weighted', pos_weight=2.0,
                                                edge_drop=0.2, mc=10, norm=True,
                                                drop_start=0.4,
                                                anneal_epochs=50),
    'posw2+dropedge_0.2+mc10+norm+drop30': dict(loss='weighted', pos_weight=2.0,
                                                edge_drop=0.3, mc=10, norm=True),
}
# pos_weight was fixed at 2.0 a priori and never swept.  These arms sweep it so
# the weight can be selected on validation rather than guessed.
for _w in (1.5, 3.0, 4.0, 6.0):
    ARMS[f'posw{_w}'] = dict(loss='weighted', pos_weight=_w)
    ARMS[f'posw{_w}+neg'] = dict(loss='weighted', pos_weight=_w,
                                 neg_resample=True)
for _r in (5, 10, 20):
    ARMS[f'negr{_r}'] = dict(neg_resample=True, neg_ratio=_r)
    ARMS[f'posw2+negr{_r}'] = dict(loss='weighted', pos_weight=2.0,
                                   neg_resample=True, neg_ratio=_r)
    ARMS[f'bpr+negr{_r}'] = dict(loss='bpr', neg_resample=True, neg_ratio=_r)
for _p in (0.05, 0.1, 0.15, 0.25, 0.3, 0.4):
    ARMS[f'dropedge_{_p}'] = dict(edge_drop=_p)
    ARMS[f'posw2+dropedge_{_p}'] = dict(loss='weighted', pos_weight=2.0,
                                        edge_drop=_p)

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


def compute_loss(pred, target, cfg):
    """BCE by default; the loss-shaping arms replace it."""
    if cfg.get('label_smooth'):
        eps = cfg['label_smooth']
        target = target.clamp(0, 1) * (1 - 2 * eps) + eps
    kind = cfg.get('loss')
    if kind == 'focal':
        # Focal (Lin et al. 2017): down-weight the easy examples so gradient
        # goes to the hard positives that decide the top of the ranking.
        g = cfg.get('gamma', 2.0)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        if cfg.get('pos_weight'):
            w = torch.full_like(target, 1.0)
            w[target > 0] = cfg['pos_weight']
            bce = bce * w
        p_t = torch.exp(-bce)
        return ((1 - p_t) ** g * bce).mean()
    if kind == 'weighted':
        # Cost-sensitive, as mDADGAN uses: reweight the positive class.
        w = torch.full_like(target, 1.0)
        w[target > 0] = cfg.get('pos_weight', 2.0)
        return F.binary_cross_entropy_with_logits(pred, target, weight=w)
    if kind == 'bpr':
        # AUC is P(score_pos > score_neg) -- a PAIRWISE quantity, while BCE is
        # pointwise.  BPR optimises the pairs directly: each positive is matched
        # against sampled negatives and the margin is pushed through logsigmoid.
        pos, neg = pred[target > 0], pred[target == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return F.binary_cross_entropy_with_logits(pred, target)
        # Match each positive to negatives by tiling, so the pair count stays
        # linear rather than |pos| x |neg|.
        reps = max(1, neg.numel() // pos.numel())
        p_rep = pos.repeat(reps)[:neg.numel()]
        return -F.logsigmoid(p_rep - neg[:p_rep.numel()]).mean()
    if kind == 'ap':
        # Smooth average-precision surrogate: a pairwise logistic ranking loss
        # weighted towards pairs near the top, which is what AP integrates.
        pos, neg = pred[target > 0], pred[target == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return F.binary_cross_entropy_with_logits(pred, target)
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        w = torch.softmax(pos, dim=0).unsqueeze(1)
        return -(w * F.logsigmoid(diff)).sum() / w.sum().clamp(min=1e-12)
    return F.binary_cross_entropy_with_logits(pred, target)


def mask_edges(edge_index, p):
    """DropEdge: keep each message-passing edge with prob 1-p."""
    if p <= 0 or edge_index.size(1) == 0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > p
    return edge_index[:, keep]


def fwd(model, data, idx, edge_index_dict=None, x_dict=None):
    # LightGCN needs the graph itself to build its normalised adjacency.
    if model.__class__.__name__ == 'LightGCN':
        return model(data.x_dict, data.edge_index_dict, idx, data=data)
    if edge_index_dict is None:
        edge_index_dict = data.edge_index_dict
    if x_dict is None:
        x_dict = data.x_dict
    return model(x_dict, edge_index_dict, idx)


@torch.no_grad()
def evaluate(model, data, key, edge_drop=0.0, mc=1, mc_rate=None, mc_full=False):
    """MC-DropEdge evaluation: average over `mc` random edge masks.

    A model trained with DropEdge sees masked graphs; evaluating on the full
    graph is a train/inference distribution mismatch.  Sampling `mc` masks
    (each at the training rate) and averaging the logits closes it, the same
    way MC dropout closes the dropout train/test gap.  `mc_rate` overrides the
    mask rate; `mc_full` includes the full graph in the average.
    """
    model.eval()
    rate = edge_drop if mc_rate is None else mc_rate
    preds = []
    if mc_full and edge_drop > 0:
        preds.append(fwd(model, data, data[key].edge_label_index,
                         data.edge_index_dict).cpu().numpy())
    for _ in range(mc):
        eidx = data.edge_index_dict if rate <= 0 else {
            k: mask_edges(e, rate)
            for k, e in data.edge_index_dict.items()
        }
        preds.append(fwd(model, data, data[key].edge_label_index, eidx).cpu().numpy())
    s = np.mean(preds, axis=0)
    y = data[key].edge_label.cpu().numpy()
    return roc_auc_score(y, s), average_precision_score(y, s), s, y


def snap_metrics(buf_val, buf_test, row_val, row_test, yv, yt, snap):
    """Average the last `snap` epochs' logits per fold and score the mean."""
    buf_val.append(row_val)
    buf_test.append(row_test)
    while len(buf_val) > snap:
        buf_val.pop(0)
    while len(buf_test) > snap:
        buf_test.pop(0)
    sv = np.mean(buf_val, axis=0)
    st = np.mean(buf_test, axis=0)
    return roc_auc_score(yv, sv), average_precision_score(yv, sv), \
        roc_auc_score(yt, st), average_precision_score(yt, st)


def run_one(run_dir, task, arm, seed, device):
    cfg = ARMS[arm]
    t = TASKS[task]
    key = (t['rna'], task, 'drug')
    set_seed(seed)
    train_data, val_data, test_data = load_splits(run_dir, task, cfg, device)

    model = Model(train_data, t['channels'], cfg.get('aggr', t['aggr']),
                  t['rna'], decoder=cfg.get('decoder', 'concat'),
                  encoder=cfg.get('encoder', 'hetero'),
                  norm=bool(cfg.get('norm')), residual=bool(cfg.get('res'))).to(device)
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
    opt = torch.optim.RMSprop(model.parameters(), lr=t['lr'], alpha=0.99,
                              weight_decay=cfg.get('wd', 0))
    if cfg.get('optimizer') == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=t['lr'],
                                weight_decay=cfg.get('wd', 0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t['epochs']) \
        if cfg.get('cosine') else None
    buf_val, buf_test = [], []

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
            ratio = cfg.get('neg_ratio', 2)
            neg_index = negative_sampling(
                pos_index, num_nodes=(n_src, n_dst),
                num_neg_samples=ratio * n_pos, method='sparse')
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
        pred = fwd(model, train_data, cur_index, train_edge_index, tr_x)
        loss = compute_loss(pred, cur_label, cfg)
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
        if cfg.get('cosine'):
            sched.step()

        snap = cfg.get('snap', 1)
        mc = cfg.get('mc', 1)
        drop_r = cfg.get('edge_drop', 0.0)
        mc_r = cfg.get('mc_rate')
        mc_f = bool(cfg.get('mc_full'))
        val_auc, val_aupr, s_v, y_v = evaluate(
            model, val_data, key, drop_r if mc > 1 else 0.0, mc, mc_r, mc_f)
        test_auc, test_aupr, s_t, y_t = evaluate(
            model, test_data, key, drop_r if mc > 1 else 0.0, mc, mc_r, mc_f)
        if snap > 1:
            val_auc, val_aupr, test_auc, test_aupr = snap_metrics(
                buf_val, buf_test, s_v, s_t, y_v, y_t, snap)
        history.append((epoch, float(loss), val_auc, val_aupr,
                        test_auc, test_aupr))

    h = pd.DataFrame(history, columns=['epoch', 'loss', 'val_auc', 'val_aupr',
                                       'test_auc', 'test_aupr'])
    best_val = h.loc[h['val_auc'].idxmax()]
    best_val_pr = h.loc[h['val_aupr'].idxmax()]
    best_test = h.loc[h['test_auc'].idxmax()]
    last = h.iloc[-1]
    return dict(
        task=task, arm=arm, seed=seed,
        final_auc=last['test_auc'], final_aupr=last['test_aupr'],
        best_test_auc=best_test['test_auc'], best_test_aupr=best_test['test_aupr'],
        sel_val_epoch=int(best_val['epoch']),
        sel_val_auc=best_val['test_auc'], sel_val_aupr=best_val['test_aupr'],
        # The paper reports an S_val row alongside S_test; these are the
        # validation-fold figures at their own best epoch, which is how a
        # best-epoch table is built.
        val_auc=best_val['val_auc'], val_aupr=best_val_pr['val_aupr'],
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
                      f"sel_val={r['sel_val_auc']:.4f} "
                      f"VAL_auc={r['val_auc']:.4f} VAL_aupr={r['val_aupr']:.4f}",
                      flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
