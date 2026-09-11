# Cold-start drugs: the author's leave-one-drug-out protocol, where the
# popularity shortcut is neutralised by construction.
#
# predict_simulated_drug.py removes every edge of a target drug, trains on the
# rest, and ranks all ncRNAs against that one drug with Precision@30 / NDCG@30.
# The test set is a single drug, so the drug prior is CONSTANT across it -- the
# degree shortcut that saturates the random split cannot operate here at all.
# That is why this surface still has headroom: iNcRD-HG scores 0.627 per-drug
# AUC on this repo's cold_drug/LncDrug split against a plain GCN's 0.846.
#
# The mechanism tested here follows directly from that:
#
#   A cold drug has no associations, so its degree carries nothing.  What it
#   does have is chemical structure.  Transfer the association profile of
#   chemically similar drugs onto it:
#
#       transfer(r, d) = sum_{d' != d} S(d,d') * A_train[r, d'] / sum_{d'} S(d,d')
#
#   i.e. "how strongly does ncRNA r associate with drugs that look like d".
#
# Chemical similarity was measured USELESS on the random split
# (improve/rare_drug.py: smoothed prior 0.9703 vs raw 0.9775) because drug
# degree already explained everything there.  Here degree is gone by
# construction, so the same signal should be the only one left.  Same mechanism,
# opposite verdict on the two surfaces -- that contrast is the finding.
#
# Selection discipline (the author's script updates its best result on the test
# drug's own Precision@30, predict_simulated_drug.py:288):
#   epoch -- on a 10% held-out slice of the TRAINING edges
#   beta  -- leave-one-drug-out over the OTHER target drugs, never the test drug
import argparse
import heapq
import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.model import Model
from improve.run import set_seed

# The nine drugs the author's script names as having >30 associations.  They are
# selected on the COMPLETE matrix and apply to both ncRNA types: on the full
# relations these nine are exactly the drugs with LncDrug degree >30, and all
# nine clear 30 on MiDrug as well.  predict_simulated_drug.py:264 loops the same
# target drug over ['lncRNA', 'LncDrug'] and ['miRNA', 'MiDrug'], so holding the
# set fixed is what makes the two tasks a paired comparison.
TARGET_DRUGS = [0, 1, 2, 3, 4, 5, 6, 10, 13]
CFG = {
    'LncDrug': dict(rna='lncRNA', lr=0.0005, epochs=301, channels=64, aggr='min'),
    'MiDrug': dict(rna='miRNA', lr=0.0001, epochs=301, channels=256, aggr='mean'),
}
RELS = {'LncDrug': ('lncRNA', 'LncDrug', 'drug'),
        'MiDrug': ('miRNA', 'MiDrug', 'drug')}
_TIE = itertools.count()          # heap tiebreaker: state dicts are unorderable


def load_graph(run_dir, device):
    """The COMPLETE graph, before any target-drug removal.

    `{task}_train_data.pth` holds only the random split's 60% of its own target
    relation, so reading the target drug's positives from it silently labels the
    40% sitting in the val/test folds as negatives.  The author's protocol does
    not do that: `conduct_case_all_data` builds from `lncRNA_info['lncRNADrug_
    edge']`, the whole matrix, and `conduct_case_train_test_data` removes one
    drug from it (predict_simulated_drug.py:189).

    Each ncRNA-drug relation is untouched in the *other* task's file -- LncDrug
    is complete inside `MiDrug_train_data.pth` and vice versa -- so taking each
    from the file where it is not the target recovers the full graph without
    re-running data prep.  Everything else (node features, LncLnc / MiMi /
    DrugDrug / LncMi, self-loops) is the released build.
    """
    d = torch.load(f'{run_dir}/3.heter_data/LncDrug_train_data.pth',
                   map_location='cpu')            # MiDrug already complete here
    other = torch.load(f'{run_dir}/3.heter_data/MiDrug_train_data.pth',
                       map_location='cpu')        # LncDrug complete here
    full = other[RELS['LncDrug']].edge_index
    d[RELS['LncDrug']].edge_index = full
    d['drug', 'rev_LncDrug', 'lncRNA'].edge_index = full[[1, 0]]
    # The random split's supervision labels describe the 60% relation that has
    # just been replaced; leaving them attached would be a trap for any reader.
    for attr in ('edge_label', 'edge_label_index'):
        for key in RELS.values():
            if attr in d[key]:
                del d[key][attr]
    return d.to(device)


def full_relation(data, key):
    return data[key].edge_index.unique(dim=1)


def make_split(data, key, rev_key, drug, device, val_frac=0.1, seed=0):
    """Remove the target drug's edges; hold out 10% of the rest for selection."""
    full = full_relation(data, key)
    is_target = full[1] == drug
    test_pos = full[:, is_target]
    rest = full[:, ~is_target]

    g = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(rest.size(1), generator=g).to(device)
    n_val = max(1, int(val_frac * rest.size(1)))
    val_pos, train_pos = rest[:, perm[:n_val]], rest[:, perm[n_val:]]

    train_data = data.clone()
    train_data[key].edge_index = train_pos
    train_data[rev_key].edge_index = train_pos[[1, 0]]
    return train_data, train_pos, val_pos, test_pos


class NoGraphEncoder(torch.nn.Module):
    """Node features only -- the graph is never touched.

    The control that matters on a cold drug: the target drug has no edges in
    the target relation, so anything the GNN adds must come from the OTHER
    relations.  If this matches or beats the GNN, that message passing is not
    helping the cold-start case.
    """

    def __init__(self, node_types, hidden):
        super().__init__()
        self.lin = torch.nn.ModuleDict(
            {nt: torch.nn.LazyLinear(hidden) for nt in node_types})

    def forward(self, x_dict, edge_index_dict):
        return {nt: F.leaky_relu(self.lin[nt](x), 0.1)
                for nt, x in x_dict.items()}


class NoGraphModel(torch.nn.Module):
    def __init__(self, data, hidden, rna_type):
        super().__init__()
        from improve.model import EdgeDecoder
        self.encoder = NoGraphEncoder(list(data.node_types), hidden)
        self.decoder = EdgeDecoder(hidden, rna_type, mode='concat')

    def forward(self, x_dict, edge_index_dict, idx):
        return self.decoder(self.encoder(x_dict, edge_index_dict), idx)


def parse_arm(arm, cfg):
    """`+`-composed arm string -> a training spec.

    The aggregation arms answer "which fixed reduction", which is the question
    the random split already settled.  The rest are the mechanism classes that
    are ORTHOGONAL TO GRAPH DENSITY, which is the only property that can make a
    mechanism two-task here: every single-task result in RESULTS.md splits on
    MiDrug's 8.9 training drug edges per ncRNA against LncDrug's 0.96, and the
    one mechanism positive on both (`posw2`) is the one that reshapes the label
    geometry instead of adding structure.
    """
    spec = dict(aggr=cfg['aggr'], nograph=False, pos_weight=None,
                smooth=0.0, swa=0, edge_drop=0.0, tta=0, neg_ratio=2)
    for p in arm.split('+'):
        if p == 'base':
            pass
        elif p == 'nograph':
            spec['nograph'] = True
        elif p in ('aggr_mean', 'aggr_sum', 'aggr_max'):
            spec['aggr'] = p[5:]
        elif p == 'posw2':                 # the random split's two-task winner
            spec['pos_weight'] = 2.0
        elif p == 'lblsm':                 # same family as posw2, untried
            spec['smooth'] = 0.1
        elif p == 'swa':                   # weight-space top-K average
            spec['swa'] = 10
        elif p.startswith('dropedge'):
            # `dropedge` == `dropedge20`.  A regulariser has a dose-response
            # curve; noise that happened to land on one rate does not, which is
            # the cheap control this repo has learned to run first.
            spec['edge_drop'] = int(p[8:]) / 100 if p[8:] else 0.2
        elif p.startswith('tta'):
            # Inference-time averaging over K stochastic edge-dropout masks.
            # Only meaningful on top of a dropedge arm -- it makes the test-time
            # graph distribution match the one the model was trained under,
            # instead of evaluating a dropout-trained model on the dense graph.
            spec['tta'] = int(p[3:]) if p[3:] else 10
        elif p.startswith('negr'):
            spec['neg_ratio'] = int(p[4:])
        else:
            raise ValueError(f'unknown arm component {p!r} in {arm!r}')
    return spec


def drop_edges(edge_index_dict, rate, device):
    """Independent Bernoulli edge dropout on every relation."""
    out = {}
    for k, e in edge_index_dict.items():
        if e.size(1) == 0:
            out[k] = e
            continue
        keep = torch.rand(e.size(1), device=device) >= rate
        out[k] = e[:, keep] if keep.any() else e
    return out


def train_one(train_data, train_pos, val_pos, key, cfg, device, seed,
              arm='base'):
    set_seed(seed)
    rna_type = cfg['rna']
    spec = parse_arm(arm, cfg)
    if spec['nograph']:
        model = NoGraphModel(train_data, cfg['channels'], rna_type).to(device)
    else:
        model = Model(train_data, cfg['channels'], spec['aggr'],
                      rna_type).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=cfg['lr'], alpha=0.99)
    n_src = train_data[rna_type].num_nodes
    n_dst = train_data['drug'].num_nodes
    n_pos = train_pos.size(1)

    val_neg = torch.stack([
        torch.randint(0, n_src, (2 * val_pos.size(1),), device=device),
        torch.randint(0, n_dst, (2 * val_pos.size(1),), device=device)])
    val_idx = torch.cat([val_pos, val_neg], dim=1)
    val_y = np.r_[np.ones(val_pos.size(1)), np.zeros(val_neg.size(1))]

    pw = (torch.tensor(spec['pos_weight'], device=device)
          if spec['pos_weight'] else None)
    # `swa` keeps the K best-validation states and averages them in WEIGHT
    # space.  multiens.py's `--snapshots` already does the score-space version;
    # this is the untried analogue, and it selects on the same signal as base
    # (validation AUC) so the two differ only in where the averaging happens.
    top = []
    best, best_state = -1.0, None
    for _ in range(1, cfg['epochs']):
        model.train()
        n_neg = spec['neg_ratio'] * n_pos
        neg = torch.stack([
            torch.randint(0, n_src, (n_neg,), device=device),
            torch.randint(0, n_dst, (n_neg,), device=device)])
        idx = torch.cat([train_pos, neg], dim=1)
        lbl = torch.cat([torch.ones(n_pos, device=device),
                         torch.zeros(neg.size(1), device=device)])
        if spec['smooth']:
            lbl = lbl * (1 - spec['smooth']) + spec['smooth'] / 2
        eid = (drop_edges(train_data.edge_index_dict, spec['edge_drop'], device)
               if spec['edge_drop'] else train_data.edge_index_dict)
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(
            model(train_data.x_dict, eid, idx), lbl, pos_weight=pw).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vs = model(train_data.x_dict, train_data.edge_index_dict,
                       val_idx).cpu().numpy()
        va = roc_auc_score(val_y, vs)
        if spec['swa']:
            if len(top) < spec['swa'] or va > top[0][0]:
                state = {k: v.detach().clone()
                         for k, v in model.state_dict().items()}
                heapq.heappush(top, (va, next(_TIE), state))
                if len(top) > spec['swa']:
                    heapq.heappop(top)
        if va > best:
            best = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if spec['swa']:
        avg = {k: torch.stack([s[k].float() for _, _, s in top]).mean(0)
               for k in top[0][2]}
        model.load_state_dict(avg)
    else:
        model.load_state_dict(best_state)
    return model, spec


def score_all(model, train_data, rna_type, drug, device, spec=None):
    n_src = train_data[rna_type].num_nodes
    idx = torch.stack([torch.arange(n_src, device=device),
                       torch.full((n_src,), drug, device=device)])
    model.eval()
    with torch.no_grad():
        if spec and spec['tta']:
            rate = spec['edge_drop'] or 0.2
            acc = 0.0
            for _ in range(spec['tta']):
                eid = drop_edges(train_data.edge_index_dict, rate, device)
                acc = acc + model(train_data.x_dict, eid, idx)
            return (acc / spec['tta']).cpu().numpy()
        return model(train_data.x_dict, train_data.edge_index_dict,
                     idx).cpu().numpy()


def chem_transfer(run_dir, train_pos, drug, n_src, n_drug, thresh):
    """Association profile of chemically similar drugs, projected onto `drug`."""
    S = pd.read_csv(f'{run_dir}/2.construct_network/drugSimMat.csv',
                    header=None).values.astype(float)[:n_drug, :n_drug]
    S = np.abs(S)
    np.fill_diagonal(S, 0.0)
    w = np.where(S[drug] > thresh, S[drug], 0.0)
    A = np.zeros((n_src, n_drug))
    tp = train_pos.cpu().numpy()
    A[tp[0], tp[1]] = 1.0
    denom = w.sum()
    return (A @ w) / denom if denom > 0 else np.zeros(n_src)


def metrics(y, s, k=30):
    top = np.argsort(s)[-k:]
    return dict(precision_at_k=float(y[top].sum() / k),
                recall_at_k=float(y[top].sum() / max(y.sum(), 1)),
                ndcg_at_k=float(ndcg_score([y], [s], k=k)),
                ap=float(average_precision_score(y, s)),
                auc=float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan)


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--task', default='LncDrug')
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--drugs', nargs='+', type=int, default=TARGET_DRUGS)
    p.add_argument('--thresh', type=float, default=0.15)
    p.add_argument('--arms', nargs='+',
                   default=['base', 'aggr_mean', 'aggr_sum', 'aggr_max',
                            'nograph'])
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = CFG[args.task]
    rna = cfg['rna']
    key = (rna, args.task, 'drug')
    rev_key = ('drug', 'rev_' + args.task, rna)
    data = load_graph(args.run_dir, device)
    n_src, n_drug = data[rna].num_nodes, data['drug'].num_nodes

    rows = []
    for seed in args.seeds:
        cache = {}
        for drug in args.drugs:
            train_data, train_pos, val_pos, test_pos = make_split(
                data, key, rev_key, drug, device, seed=seed)
            if test_pos.size(1) == 0:
                continue
            y = np.zeros(n_src)
            y[test_pos[0].cpu().numpy()] = 1.0
            s_chem = chem_transfer(args.run_dir, train_pos, drug, n_src,
                                   n_drug, args.thresh)
            per_arm = {}
            for arm in args.arms:
                model = train_one(train_data, train_pos, val_pos, key, cfg,
                                  device, seed, arm=arm)
                per_arm[arm] = zs(score_all(model, train_data, rna, drug, device))
            cache[drug] = (y, per_arm, zs(s_chem))

        # beta by leave-one-drug-out over the OTHER target drugs
        betas = np.concatenate([[0.0], np.linspace(0.1, 3.0, 30)])
        for drug, (y, per_arm, sc) in cache.items():
            out = []
            for arm, s in per_arm.items():
                m = metrics(y, s)
                m.update(task=args.task, seed=seed, drug=drug, arm=arm,
                         n_pos=int(y.sum()))
                rows.append(m); out.append(f"{arm}={m['precision_at_k']:.3f}")
            m = metrics(y, sc)
            m.update(task=args.task, seed=seed, drug=drug, arm='chem_only',
                     n_pos=int(y.sum()))
            rows.append(m)
            print(f"seed={seed} drug={drug:3d} n_pos={int(y.sum()):3d} "
                  f"P@30 " + ' '.join(out), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print('\n=== mean over drugs and seeds ===')
    print(df.groupby('arm')[['precision_at_k', 'ndcg_at_k', 'recall_at_k',
                             'ap', 'auc']].mean()
          .to_string(float_format=lambda v: f'{v:.4f}'))
    print('\n=== paired vs base, per (drug, seed) ===')
    from scipy import stats
    b = df[df.arm == 'base'].set_index(['drug', 'seed'])
    for arm in df.arm.unique():
        if arm == 'base':
            continue
        a = df[df.arm == arm].set_index(['drug', 'seed'])
        i = b.index.intersection(a.index)
        for met in ('precision_at_k', 'ndcg_at_k', 'auc'):
            d = (a.loc[i, met] - b.loc[i, met]).dropna()
            if len(d) < 3:
                continue
            print(f'{arm:10s} {met:15s} delta={d.mean():+.4f} '
                  f'wins={(d>0).sum()}/{len(d)} p={stats.ttest_1samp(d,0).pvalue:.4f}')


if __name__ == '__main__':
    main()
