# Weighted similarity edges: the graph construction axis the ledger never varied.
#
# Every graph variant in this repo changed WHICH edges exist (thresholds, kNN)
# or WHICH matrix built them (multi-view similarity).  The released model reads
# the graph as a 100% binary adjacency -- `GeneralConv` over `edge_index`, no
# `edge_attr` anywhere -- so a similarity edge that is 0.31 above threshold and
# one that is 0.99 are message-passed identically.  The tree features in
# `gbdt.py` have never had this problem (they read S_r / S_d continuously), but
# the GNN, the paper's own architecture, has only ever seen the thresholded
# binary graph.
#
# This file is the first weighted-graph arm: keep the exact released edge set
# (same thresholds, so the split and the topology are identical), but attach the
# continuous |similarity| as `edge_attr` on the three similarity relations, and
# run the message passing with a weighted conv that scales each neighbor
# message by its edge weight.  With all weights = 1 it reduces to the released
# architecture exactly, so the control is the released model and the only
# difference is the weights.
#
# Two further axes are here:
#   --edge-set knn   -- replace the global |sim|>0.3 threshold with per-node
#                      top-k partners (same rule as `graph_sweep`'s knn spec).
#                      The released kNN arms were binary; this combines the
#                      top-k topology WITH continuous weights, which the
#                      threshold-only weighted arm could not do.
#   --mode signed    -- keep the sign of sim instead of abs().  The released
#                      pipeline np.abs()es every similarity matrix
#                      (heter_data_prepare.py line 27), so an anti-correlated
#                      pair (sim < -0.3) is currently message-passed as a
#                      strongly co-expressed pair.  `signed` attaches the raw
#                      sim as the weight (negative weights repel the message);
#                      `split` instead separates the released edge set into
#                      `_pos` (sim > 0.3) and `_neg` (sim < -0.3) relations so
#                      the two semantics get their own transformation and are
#                      fused by the released per-relation aggregation.  The
#                      combined split edge set is exactly the released one.
#
# Why this is not covered by anything already measured: `graph_sweep` moved the
# threshold (edge EXISTENCE); `simviews` swapped the matrix (which edge set);
# neither touched edge STRENGTH.  A low-similarity edge that survives the 0.3
# cutoff currently carries the same weight as a near-1 similarity edge, and
# `feature_groups.py` measured that continuous similarity is exactly the
# information the LncDrug degree features are missing.  The weighted graph
# hands the GNN that information directly.  `graph_sweep`'s knn arms were
# binary-only and measured NEGATIVE on LncDrug test; whether top-k becomes
# useful once it carries continuous weights, and whether the released abs() is
# discarding signal or noise, are the open questions this file answers.
#
# Leakage: edge weights are a function of node attributes (expression / Morgan
# fingerprints), not of A, so they sit outside the split exactly like the
# topology does -- RandomLinkSplit partitions only the (ncRNA, task, drug)
# relation and never touches the similarity relations.
#
# Implementation note: `to_hetero` traces the module with torch.fx, and a lazy
# in_channels=-1 Linear inside a custom MessagePassing resolves to a wrong size
# during tracing (the released GeneralConv is unaffected because its lazy Linear
# is handled by PyG's own tracing path).  Rather than fight that, the encoder
# here runs the message passing manually over `edge_index_dict`, which is
# exactly the graph `to_hetero(GeneralConv, aggr)` executes: per relation a
# GeneralConv-style layer `lin_self(x_t) + sum_j w_ij * lin_msg(x_j)`, then the
# per-relation outputs are combined per target node type by the released
# aggregation (min on LncDrug, mean on MiDrug).
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.graph_prep import build, seed_all, split_all
from improve.run import TASKS, set_seed

REL_OUT = {'lncRNA': 'lncRNA', 'miRNA': 'miRNA', 'drug': 'drug'}
EPS = 1e-9


class ManualEncoder(torch.nn.Module):
    """Per-relation GeneralConv-style message passing, aggregated per node type.

    `to_hetero(GeneralConv, aggr)` clones the encoder once PER RELATION and
    combines a target node's per-relation outputs with `aggr` (min on LncDrug,
    mean on MiDrug).  This reproduces that structure directly: one lin_msg /
    lin_self per relation, `sum` aggregation inside the relation (GeneralConv's
    default), then the released reduction over incoming relations.  Weights
    scale the neighbor messages; with all weights 1.0 this is the released
    model's exact computation graph.
    """

    def __init__(self, node_types, relations, hidden, aggr, layers=2):
        super().__init__()
        self.node_types = list(node_types)
        self.relations = list(relations)
        self.hidden = hidden
        self.aggr = aggr
        self.n_layers = layers
        # Layer 0 reads raw attributes (lazy in_channels); later layers read the
        # hidden output of the previous layer (explicit).  The released
        # GNNEncoder's two lazy GeneralConvs materialise the same way under
        # to_hetero's trace.
        in_dims = [-1] + [hidden] * (layers - 1)
        self.lins_msg = torch.nn.ModuleList([
            torch.nn.ModuleDict(
                {f'{s}__{rel}__{t}': Linear(d, hidden) for s, rel, t in relations})
            for d in in_dims])
        self.lins_self = torch.nn.ModuleList([
            torch.nn.ModuleDict(
                {f'{s}__{rel}__{t}': Linear(d, hidden) for s, rel, t in relations})
            for d in in_dims])

    def _layer(self, layer_idx, x_dict, edge_index_dict, edge_weight_dict,
               out_dim):
        """One message-passing layer.  Returns a new x_dict."""
        lmsg, lself = self.lins_msg[layer_idx], self.lins_self[layer_idx]
        acc = {nt: [] for nt in self.node_types}
        for (s, rel, t), ei in edge_index_dict.items():
            k = f'{s}__{rel}__{t}'
            if ei.size(1) == 0:
                continue
            msg = lmsg[k](x_dict[s])
            if edge_weight_dict is not None and (s, rel, t) in edge_weight_dict:
                w = edge_weight_dict[(s, rel, t)].to(ei.device).view(-1)
            else:
                w = torch.ones(ei.size(1), device=ei.device)
            m = msg[ei[0]] * w.unsqueeze(-1)          # [E, hidden]
            contrib = scatter(m, ei[1], dim=0,
                              dim_size=x_dict[t].size(0), reduce='sum')
            acc[t].append(contrib + lself[k](x_dict[t]))
        out = {}
        for t in self.node_types:
            if not acc[t]:
                out[t] = torch.zeros(x_dict[t].size(0), out_dim,
                                     device=x_dict[t].device)
                continue
            stack = torch.stack(acc[t], dim=0)        # [n_rels, N_t, hidden]
            if self.aggr == 'min':
                h = stack.min(dim=0).values
            elif self.aggr == 'mean':
                h = stack.mean(dim=0)
            elif self.aggr == 'sum':
                h = stack.sum(dim=0)
            else:
                raise ValueError(self.aggr)
            out[t] = h
        return out

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        h = x_dict
        for i in range(self.n_layers):
            h = self._layer(i, h, edge_index_dict, edge_weight_dict,
                            self.hidden)
            if i < self.n_layers - 1:
                h = {t: F.leaky_relu(v, negative_slope=0.1)
                     for t, v in h.items()}
        return h


class WeightedModel(torch.nn.Module):
    def __init__(self, data, hidden_channels, aggr, rna_type, layers=2):
        super().__init__()
        self.encoder = ManualEncoder(data.node_types,
                                     list(data.edge_index_dict.keys()),
                                     hidden_channels, aggr, layers=layers)
        self.rna_type = rna_type
        self.decoder = torch.nn.Sequential(
            Linear(2 * hidden_channels, 2 * hidden_channels),
            torch.nn.LeakyReLU(0.1),
            Linear(2 * hidden_channels, hidden_channels),
            torch.nn.LeakyReLU(0.1),
            Linear(hidden_channels, 1))

    def forward(self, x_dict, edge_index_dict, edge_weight_dict,
                edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict, edge_weight_dict)
        row, col = edge_label_index
        z = torch.cat([z_dict[self.rna_type][row], z_dict['drug'][col]], dim=-1)
        return self.decoder(z).view(-1)


def attach_weights(data, net_dir, mode, which=('lncRNA', 'miRNA', 'drug'),
                   lncmi='none', assoc_w=1.0):
    """Overlay continuous similarity weights on the three similarity relations.

    `mode`:
      bin    -- control: all weights 1.0 (reproduces the released binary graph)
      raw    -- w = |sim| value at the thresholded edge
      resid  -- w = max(|sim| - 0.3, 0): the margin above the 0.3 cutoff, so
                barely-edges are near zero and near-1 edges dominate strongly
      signed -- w = sim (signed, no abs): anti-correlated edges carry a
                negative weight, so their messages repel instead of attracting.
                Edge set is still |sim| > threshold (released topology).
      split  -- the released edge set is divided into `_pos` (sim > 0.3) and
                `_neg` (sim < -0.3) relations, each carrying w = |sim|.  The
                graph was built with `build(..., signed=True)`; the two
                semantics get separate transforms and are fused by the released
                per-relation aggregation.  Self-loops are dropped from `_neg`
                (an identity edge has no anti-correlation semantics).
    `which` restricts weighting to a subset of relations; the rest stay binary
    (all weights 1.0).  `lncmi` weights the cross-ncRNA bridge:
      none   -- binary (released)
      coex   -- w = |Pearson(lncRNA profile, miRNA profile)| over the shared
                938 samples, so the bridge carries co-regulation strength
                instead of ENCORI's binary interaction flag
    `assoc_w` scales the ASSOCIATION edges (ncRNA-drug, LncMi, and reverses)
    by a constant, so the relative strength of association vs similarity signal
    is a free axis -- the one construction knob that is symmetric by
    construction (both tasks share the same graph).  At 1.0 it is the released
    graph; below 1.0 it lets the similarity signal compete with the degree
    signal on the dense task, which is the file's measured MiDrug bottleneck.
    Association edges and self-loops always keep weight 1.0 unless `assoc_w`
    is set.
    """
    rels = {
        'lncRNA': ('LncLnc', 'LncSimMat_ken.csv'),
        'miRNA': ('MiMi', 'MiSimMat_ken.csv'),
        'drug': ('DrugDrug', 'drugSimMat.csv'),
    }
    for node_type, (rel, fname) in rels.items():
        keys = [(node_type, rel, node_type)]
        if mode == 'split':
            keys = [(node_type, f'{rel}_pos', node_type),
                    (node_type, f'{rel}_neg', node_type)]
        for key in keys:
            ei = data[key].edge_index
            if mode == 'bin' or node_type not in which:
                w = torch.ones(ei.size(1))
            elif mode == 'signed':
                M = pd.read_csv(f'{net_dir}/{fname}', header=None).values
                w = torch.tensor(M[ei[0], ei[1]], dtype=torch.float)
                w = torch.nan_to_num(w, nan=0.0)
                # Self-loops (AddSelfLoops) are identity edges, not similarity
                # edges -- the similarity matrix's zero diagonal must not kill
                # them.
                w[ei[0] == ei[1]] = 1.0
            else:
                M = np.abs(pd.read_csv(f'{net_dir}/{fname}', header=None).values)
                w = torch.tensor(M[ei[0], ei[1]], dtype=torch.float)
                w = torch.nan_to_num(w, nan=0.0)
                if mode == 'resid':
                    w = (w - 0.3).clamp(min=0.0)
                # Self-loops (AddSelfLoops) are identity edges, not similarity
                # edges -- the similarity matrix's zero diagonal must not kill
                # them.
                w[ei[0] == ei[1]] = 1.0
            if key[1].endswith('_neg'):
                keep = (ei[0] != ei[1])
                ei = ei[:, keep]
                data[key].edge_index = ei
                w = w[keep]
            data[key].edge_attr = w.unsqueeze(-1)

    if lncmi == 'coex':
        lm_key = ('lncRNA', 'LncMi', 'miRNA')
        ei = data[lm_key].edge_index
        le = pd.read_csv(f'{net_dir}/LncExp.csv', index_col=0)
        me = pd.read_csv(f'{net_dir}/MiExp.csv', index_col=0).reindex(le.index)
        mc = {c.upper(): c for c in me.columns}
        w = torch.ones(ei.size(1))
        for i in range(ei.size(1)):
            lc = le.columns[ei[0, i].item()]
            mic = me.columns[ei[1, i].item()].upper()
            if mic in mc:
                x = le[lc].values.astype(float)
                y = me[mc[mic]].values.astype(float)
                if np.std(x) > 0 and np.std(y) > 0:
                    w[i] = abs(float(np.corrcoef(x, y)[0, 1]))
        data[lm_key].edge_attr = w.unsqueeze(-1)

    if assoc_w != 1.0:
        for key in list(data.edge_index_dict.keys()):
            s, rel, t = key
            if s == t and rel in ('LncLnc', 'MiMi', 'DrugDrug'):
                continue
            n = data[key].edge_index.size(1)
            data[key].edge_attr = torch.full((n, 1), assoc_w)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--net-dir', required=True, help='2.construct_network')
    p.add_argument('--run-dir', default='.', help='3.heter_data (for shape check)')
    p.add_argument('--mode', choices=['bin', 'raw', 'resid', 'signed', 'split'],
                   default='raw')
    p.add_argument('--edge-set', choices=['thr', 'knn'], default='thr',
                   help='which edges exist: released threshold (|sim|>0.3) or '
                        'per-node top-k partners')
    p.add_argument('--k', type=int, default=3,
                   help='per-node top-k when --edge-set knn (all three '
                        'relations)')
    p.add_argument('--which', nargs='+', default=['lncRNA', 'miRNA', 'drug'],
                   choices=['lncRNA', 'miRNA', 'drug'],
                   help='relations to weight; others stay binary')
    p.add_argument('--lncmi', choices=['none', 'coex'], default='none',
                   help='weight the LncMi bridge by co-expression')
    p.add_argument('--assoc-w', type=float, default=1.0,
                   help='scale association edges by this constant (default 1.0 '
                        '= released; <1 lets similarity compete with degree)')
    p.add_argument('--tasks', nargs='+', default=['LncDrug', 'MiDrug'])
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    seed_all()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.edge_set == 'knn':
        data = build(args.net_dir, args.k, args.k, args.k, knn=True,
                     signed=(args.mode == 'split'))
    else:
        data = build(args.net_dir, 0.3, 0.3, 0.3,
                     signed=(args.mode == 'split'))
    data = attach_weights(data, args.net_dir, args.mode, tuple(args.which),
                          args.lncmi, args.assoc_w)
    for key in data.edge_index_dict:
        n = data[key].edge_index.size(1)
        nt = key[0]
        print(f'  {str(key):48s} {n:7d} edges '
              f'(mean deg {n / data[nt].num_nodes:.2f})')
    splits = split_all(data, device)

    rows = []
    for task in args.tasks:
        t = TASKS[task]
        key = (t['rna'], task, 'drug')
        n_epochs = args.epochs or t['epochs']
        tr, va, te = splits[task]
        for seed in args.seeds:
            set_seed(seed)
            model = WeightedModel(tr, t['channels'], t['aggr'], t['rna']).to(device)
            opt = torch.optim.RMSprop(model.parameters(), lr=t['lr'],
                                      alpha=0.99)
            label_index = tr[key].edge_label_index
            label = tr[key].edge_label
            history = []
            for epoch in range(1, n_epochs):
                model.train()
                opt.zero_grad()
                ewd = {k: v.to(device) for k, v in tr.edge_attr_dict.items()}
                pred = model(tr.x_dict, tr.edge_index_dict, ewd, label_index)
                w = torch.full_like(label, 1.0)
                w[label > 0] = 2.0
                loss = F.binary_cross_entropy_with_logits(pred, label, weight=w)
                loss.backward()
                opt.step()

                if epoch % 20 == 0 or epoch == n_epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        for name, d in (('val', va), ('test', te)):
                            d_ewd = {k: v.to(device)
                                     for k, v in d.edge_attr_dict.items()}
                            s = model(d.x_dict, d.edge_index_dict, d_ewd,
                                      d[key].edge_label_index).cpu().numpy()
                            y = d[key].edge_label.cpu().numpy()
                            if name == 'val':
                                va_auc = roc_auc_score(y, s)
                                va_aupr = average_precision_score(y, s)
                            else:
                                te_auc = roc_auc_score(y, s)
                                te_aupr = average_precision_score(y, s)
                    history.append((epoch, va_auc, va_aupr, te_auc, te_aupr))
            h = pd.DataFrame(history, columns=['epoch', 'va_auc', 'va_aupr',
                                               'te_auc', 'te_aupr'])
            best = h.loc[h['va_auc'].idxmax()]
            rows.append(dict(task=task, mode=args.mode,
                             edge_set=args.edge_set, k=args.k, seed=seed,
                             val_auc=best['va_auc'], val_aupr=best['va_aupr'],
                             test_auc=best['te_auc'], test_aupr=best['te_aupr']))
            print(f"{task:8s} {args.mode:5s} {args.edge_set:3s} seed={seed} "
                  f"val {best['va_auc']:.4f}/{best['va_aupr']:.4f} "
                  f"test {best['te_auc']:.4f}/{best['te_aupr']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()