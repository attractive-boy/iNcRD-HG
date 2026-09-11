# NCNC-style member for the fusion pool: MPNN-then-SF on the heterogeneous graph.
#
# Neural Common Neighbor (NCN, ICLR 2024) runs an MPNN once, then lets the
# *node representations of the common neighbours* guide the link representation:
#
#   z_uv = [ z_u || z_v || sum_{c in CN(u,v)} w_c * z_c ]
#
# That is strictly more expressive than counting them (CN / RA / AA / BUDDY all
# reduce to counts), because the *properties* of the common neighbours are
# pooled, not just their existence.
#
# For a bipartite target relation (rna, drug) with similarity edges on both
# sides, the common neighbours of a candidate pair (r, d) are:
#
#   drug side:  drugs d' with S_d[d', d] > 0  and  A[r, d'] = 1
#               -- drugs chemically similar to d that r already hits.
#   ncRNA side: ncRNAs r' with S_r[r', r] > 0  and  A[r', d] = 1
#               -- ncRNAs similar to r that already hit d.
#
# So the pooling term is a LEARNED, similarity-weighted version of the
# hand-crafted `chem_transfer` projection: instead of summing the raw chemical
# similarity onto the association profile, it pools the message-passed node
# representations of those neighbours through an MLP.
#
# Leakage discipline follows gbdt.py:
#   * the similarity matrices carry a zero diagonal, so the first-order pooling
#     never reads the candidate pair's own label;
#   * training rows are scored K-fold -- for fold k the pooling matrix AND the
#     message-passing graph are built from the association matrix with fold k's
#     positives removed.  Validation and test rows use the full training matrix
#     and the training graph (no transductive val/test edges in the MPNN).
#
# The model: per-node-type linear over the released attributes, one
# symmetric-normalised propagation step over the hetero graph, then a 3-layer
# MLP over [z_r || z_d || pool_d || pool_r] plus the pair's own common-neighbour
# counts (CN_d / CN_r), which NCN keeps alongside the pooled features.
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.gbdt import dense, sim_matrix
from improve.run import TASKS, set_seed

EPS = 1e-9


class NCNScorer(torch.nn.Module):
    def __init__(self, node_types, attrs, hidden, rna_type, n_layers=1, cn=True,
                 pool=True):
        super().__init__()
        self.node_types = list(node_types)
        self.rna_type = rna_type
        self.n_layers = n_layers
        self.cn = cn
        self.pool = pool
        self.in_lins = torch.nn.ModuleDict({
            nt: torch.nn.Linear(attrs[nt].shape[1], hidden)
            for nt in self.node_types})
        # z_r, z_d (+ pool_d, pool_r if pooling) (+ cn_d, cn_r if counts)
        in_dim = 2 * hidden
        if self.pool:
            in_dim += 2 * hidden
        if self.cn:
            in_dim += 2
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 2 * hidden),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(2 * hidden, hidden),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hidden, 1))

    def build_offsets(self, num_nodes):
        self.offset, off = {}, 0
        for nt in self.node_types:
            self.offset[nt] = off
            off += num_nodes[nt]
        self.n_total = off

    def _adj(self, edge_index_dict, n_total, device):
        rows, cols = [], []
        for (s, _, t), ei in edge_index_dict.items():
            r = ei[0] + self.offset[s]
            c = ei[1] + self.offset[t]
            rows += [r, c]
            cols += [c, r]
        row = torch.cat(rows).to(device)
        col = torch.cat(cols).to(device)
        val = torch.ones(row.numel(), device=device)
        deg = torch.zeros(n_total, device=device).scatter_add_(0, row, val)
        d = deg.clamp(min=1).pow(-0.5)
        w = d[row] * d[col]
        return torch.sparse_coo_tensor(
            torch.stack([row, col]), w, (n_total, n_total)).coalesce()

    def embed(self, attrs, edge_index_dict):
        x = torch.cat([self.in_lins[nt](attrs[nt])
                       for nt in self.node_types], dim=0)
        adj = self._adj(edge_index_dict, self.n_total, x.device)
        out = x
        for _ in range(self.n_layers):
            out = out + torch.sparse.mm(adj, out)
        out = out / (self.n_layers + 1)
        z, off = {}, 0
        for nt in self.node_types:
            n = attrs[nt].shape[0]
            z[nt] = out[off:off + n]
            off += n
        return z

    def forward(self, attrs, edge_index_dict, pairs, A, S_r, S_d):
        z = self.embed(attrs, edge_index_dict)
        r, d = pairs[0], pairs[1]
        z_r = z[self.rna_type][r]
        z_d = z['drug'][d]
        Wd = torch.as_tensor(A[r] * S_d[d], device=z_r.device, dtype=torch.float)
        Wr = torch.as_tensor(S_r[r] * A[:, d].T, device=z_r.device,
                             dtype=torch.float)
        parts = [z_r, z_d]
        if self.pool:
            pool_d = (Wd @ z['drug']) / (Wd.sum(1, keepdim=True) + EPS)
            pool_r = (Wr @ z[self.rna_type]) / (Wr.sum(1, keepdim=True) + EPS)
            parts += [pool_d, pool_r]
        if self.cn:
            cn_d = (A[r] * S_d[d]).sum(1, keepdims=True)
            cn_r = (S_r[r] * A[:, d].T).sum(1, keepdims=True)
            parts += [torch.as_tensor(cn_d, device=z_r.device),
                      torch.as_tensor(cn_r, device=z_r.device)]
        return self.mlp(torch.cat(parts, dim=1)).view(-1)


def build_inputs(run_dir, net_dir, task, device):
    t = TASKS[task]
    rna = t['rna']
    key = (rna, task, 'drug')
    splits = [torch.load(f'{run_dir}/3.heter_data/{task}_{s}_data.pth',
                         map_location='cpu') for s in ('train', 'val', 'test')]
    train = splits[0]
    n_r, n_d = train[rna].num_nodes, train['drug'].num_nodes
    A = dense(train[key].edge_index, n_r, n_d)
    sim_file = {'lncRNA': 'LncSimMat_ken.csv', 'miRNA': 'MiSimMat_ken.csv'}[rna]
    S_r = sim_matrix(f'{net_dir}/{sim_file}', n_r)
    S_d = sim_matrix(f'{net_dir}/drugSimMat.csv', n_d)
    attrs = {nt: torch.as_tensor(train[nt].x, dtype=torch.float32)
             for nt in train.node_types}
    num_nodes = {nt: train[nt].num_nodes for nt in train.node_types}
    return splits, key, A, S_r, S_d, attrs, num_nodes


def guarded_graph(graph, key, rev_key, held):
    """The training graph with `held` (fold-k positive pairs) removed from the
    target relation and its reverse, so an OOF row's message passing never sees
    its own label's edge."""
    g = {k: v.clone() for k, v in graph.items()}
    if held.size(1):
        hd = set(map(tuple, held.t().tolist()))
        fwd = graph[key]
        keep = torch.tensor([tuple(e) not in hd for e in fwd.t().tolist()],
                            dtype=bool)
        g[key] = fwd[:, keep]
        g[rev_key] = g[key][[1, 0]]
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--cn', action='store_true', default=True)
    p.add_argument('--no-cn', dest='cn', action='store_false')
    p.add_argument('--pool', action='store_true', default=True)
    p.add_argument('--no-pool', dest='pool', action='store_false')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', required=True)
    p.add_argument('--dump-scores', default=None)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits, key, A, S_r, S_d, attrs, num_nodes = build_inputs(
        args.run_dir, args.net_dir, args.task, device)
    train, val, test = splits
    rna = TASKS[args.task]['rna']
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in (('train', train), ('val', val), ('test', test))}
    y = {f: rows[f][1] for f in ('val', 'test')}
    tr_idx, tr_y = rows['train']

    rev_key = ('drug', 'rev_' + args.task, rna)
    full_graph = train.edge_index_dict
    attrs_d = {k: v.to(device) for k, v in attrs.items()}
    Sr_d = torch.as_tensor(S_r, dtype=torch.float32, device=device)
    Sd_d = torch.as_tensor(S_d, dtype=torch.float32, device=device)

    def tensor_A(Ak):
        return torch.as_tensor(Ak, dtype=torch.float32, device=device)

    def train_model(pairs, yb, A_tr, graph, epochs):
        m = NCNScorer(num_nodes.keys(), attrs_d, args.hidden, rna,
                      n_layers=args.layers, cn=args.cn, pool=args.pool).to(device)
        m.build_offsets(num_nodes)
        opt = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=1e-4)
        pa = torch.as_tensor(pairs, device=device)
        yt = torch.as_tensor(yb, dtype=torch.float32, device=device)
        At = tensor_A(A_tr)
        g = {k: v.to(device) for k, v in graph.items()}
        for _ in range(epochs):
            m.train()
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                m(attrs_d, g, pa, At, Sr_d, Sd_d), yt)
            loss.backward()
            opt.step()
        m.eval()
        return m

    def predict(model, pairs, A_tr, graph):
        with torch.no_grad():
            return model(attrs_d, graph,
                         torch.as_tensor(pairs, device=device),
                         tensor_A(A_tr), Sr_d, Sd_d).cpu().numpy()

    set_seed(0)
    rng = np.random.RandomState(0)
    fold = rng.randint(0, args.folds, size=tr_idx.shape[1])

    oof = np.zeros(tr_idx.shape[1])
    for k in range(args.folds):
        msk = fold == k
        Ak = A.copy()
        held = tr_idx[:, msk & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        g = guarded_graph(full_graph, key, rev_key, torch.as_tensor(held))
        gd = {kk: v.to(device) for kk, v in g.items()}
        model = train_model(tr_idx[:, ~msk], tr_y[~msk], Ak, gd, args.epochs)
        oof[msk] = predict(model, tr_idx[:, msk], Ak, gd)
        print(f"  fold {k}: oof auc "
              f"{roc_auc_score(tr_y[msk], oof[msk]):.4f}", flush=True)

    acc = {'val': [], 'test': []}
    for seed in args.seeds:
        set_seed(seed)
        gd = {kk: v.to(device) for kk, v in full_graph.items()}
        model = train_model(tr_idx, tr_y, A, gd, args.epochs)
        for f in ('val', 'test'):
            acc[f].append(predict(model, rows[f][0], A, gd))
        print(f"  seed={seed} val/test done", flush=True)
    scores = {f: np.mean(acc[f], axis=0) for f in ('val', 'test')}

    print(f"\n{args.task} NCN member (5-fold OOF + {len(args.seeds)} seeds, "
          f"layers={args.layers} cn={args.cn})")
    for f in ('val', 'test'):
        print(f"  {f}: AUC {roc_auc_score(y[f], scores[f]):.4f} "
              f"AUPR {average_precision_score(y[f], scores[f]):.4f}")
    print(f"  OOF train: AUC {roc_auc_score(tr_y, oof):.4f}")
    if args.dump_scores:
        np.savez(args.dump_scores, ncn_val=scores['val'], ncn_test=scores['test'],
                 ncn_train=oof, y_val=y['val'], y_test=y['test'], y_train=tr_y)
        print(f'wrote {args.dump_scores}')


if __name__ == '__main__':
    main()