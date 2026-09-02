# A hybrid architecture: GNN embeddings + pairwise structural features.
#
# Why this and not another encoder.  The released decoder is
# `MLP([z_r || z_d])`, a function of the two node embeddings alone, so it cannot
# represent any pair-level quantity that does not factorise through them --
# common neighbours, Adamic-Adar, `(A A^T A)[r,d]`, similarity-weighted
# collaborative scores.  Two-layer message passing gives 2-hop *node*
# representations, not 3-hop *pair* statistics.  That is the standard
# expressiveness gap in link prediction (SEAL 2018, Neo-GNN 2021, BUDDY 2023),
# and it explains two results in improve/RESULTS.md that otherwise sit oddly
# together: feeding node degree in is exactly neutral (the GNN already has it),
# yet a tree on explicit pair features beats the GNN outright (it has what the
# GNN structurally cannot compute).
#
# So the fix is architectural rather than an ensemble: concatenate the pair
# feature vector onto the decoder input and train end to end.
#
#     score(r,d) = MLP([ z_r || z_d || phi(r,d) ])
#
# `phi` is improve/gbdt.py's 19 features, standardised with training statistics.
# Training rows get the same K-fold construction as the tree models -- a
# training positive is an edge of the matrix its own features come from, so
# fold k's features are computed with fold k's positives removed.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch_geometric.nn import to_hetero
from torch_geometric.utils import negative_sampling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improve.gbdt import build_inputs, featurise
from improve.model import GNNEncoder
from improve.run import ARMS, TASKS, compute_loss, set_seed


class PairDecoder(torch.nn.Module):
    """The author's three-layer MLP, plus a pathway for the pair features.

    `concat` widens the MLP input with phi.  It underperforms, and the reason is
    informative: 19 standardised features against 2x256 embedding dimensions in
    the first Linear, with one shared trunk that has to model both.

    `additive` gives phi its own head and sums the logits:

        score = MLP([z_r || z_d]) + g(phi)

    which is the differentiable analogue of the rank blend that actually works
    (`w1 * rank(GNN) + w2 * rank(tree)`).  Late fusion, trained end to end, so
    the encoder still gets gradient through the residual.
    """

    def __init__(self, hidden_channels, rna_type, n_pair, mode='additive'):
        super().__init__()
        assert mode in ('none', 'concat', 'additive')
        self.rna_type = rna_type
        self.mode = mode
        in_dim = 2 * hidden_channels + (n_pair if mode == 'concat' else 0)
        self.lin1 = Linear(in_dim, 2 * hidden_channels)
        self.lin2 = Linear(2 * hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, 1)
        if mode == 'additive':
            h = 64
            self.g = torch.nn.Sequential(
                Linear(n_pair, h), torch.nn.LeakyReLU(0.1),
                Linear(h, h), torch.nn.LeakyReLU(0.1), Linear(h, 1))

    def forward(self, z_dict, edge_label_index, phi):
        row, col = edge_label_index
        z = [z_dict[self.rna_type][row], z_dict['drug'][col]]
        if self.mode == 'concat':
            z.append(phi)
        z = torch.cat(z, dim=-1)
        z = F.leaky_relu(self.lin1(z), 0.1)
        z = F.leaky_relu(self.lin2(z), 0.1)
        out = self.lin3(z).view(-1)
        if self.mode == 'additive':
            out = out + self.g(phi).view(-1)
        return out


class Hybrid(torch.nn.Module):
    def __init__(self, data, hidden, aggr, rna_type, n_pair, mode='additive'):
        super().__init__()
        enc = GNNEncoder(hidden, hidden)
        self.encoder = to_hetero(enc, data.metadata(), aggr=aggr)
        self.decoder = PairDecoder(hidden, rna_type, n_pair, mode)

    def forward(self, x_dict, edge_index_dict, edge_label_index, phi):
        return self.decoder(self.encoder(x_dict, edge_index_dict),
                            edge_label_index, phi)


def pair_features(run_dir, net_dir, task, folds=5):
    """phi for train / val / test rows, with the K-fold guard on train."""
    splits, key, A, S_r, S_d, bridge, attrs = build_inputs(
        run_dir, net_dir, task, 'cpu')
    rows = {n: (d[key].edge_label_index.cpu().numpy(),
                d[key].edge_label.cpu().numpy())
            for n, d in zip(('train', 'val', 'test'), splits)}
    X = {f: featurise(rows[f][0], A, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1]) for f in ('val', 'test')}
    tr_idx, tr_y = rows['train']
    rng = np.random.RandomState(0)
    fold = rng.randint(0, folds, size=tr_idx.shape[1])
    parts = []
    for k in range(folds):
        m = fold == k
        Ak = A.copy()
        held = tr_idx[:, m & (tr_y == 1)]
        Ak[held[0], held[1]] = 0.0
        f = featurise(tr_idx[:, m], Ak, S_r, S_d, bridge, Xr=attrs[0], Xd=attrs[1])
        f.index = np.where(m)[0]
        parts.append(f)
    X['train'] = pd.concat(parts).sort_index()
    # Standardise on the training rows; a raw path count is O(100) and would
    # swamp the embeddings in the first Linear.
    mu, sd = X['train'].values.mean(0), X['train'].values.std(0) + 1e-8
    return {f: torch.tensor((X[f].values - mu) / sd, dtype=torch.float)
            for f in X}, splits, key


def run_one(run_dir, net_dir, task, seed, device, arm='posw2', lr=5e-4,
            channels=256, aggr='min', epochs=301, mode='additive'):
    t = TASKS[task]
    cfg = ARMS[arm]
    phi, splits, key = pair_features(run_dir, net_dir, task)
    phi = {f: v.to(device) for f, v in phi.items()}
    set_seed(seed)
    train_data, val_data, test_data = [d.to(device) for d in splits]

    model = Hybrid(train_data, channels, aggr, t['rna'],
                   phi['train'].size(1), mode).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)

    li, lb = train_data[key].edge_label_index, train_data[key].edge_label
    y = {f: d[key].edge_label.cpu().numpy()
         for f, d in (('val', val_data), ('test', test_data))}
    best = {'auc': -1.0, 'aupr': -1.0}
    cap = {}
    for _ in range(1, epochs):
        model.train()
        opt.zero_grad()
        pred = model(train_data.x_dict, train_data.edge_index_dict, li,
                     phi['train'])
        loss = compute_loss(pred, lb, cfg)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            s = {f: model(d.x_dict, d.edge_index_dict,
                          d[key].edge_label_index, phi[f]).cpu().numpy()
                 for f, d in (('val', val_data), ('test', test_data))}
        for m, v in (('auc', roc_auc_score(y['val'], s['val'])),
                     ('aupr', average_precision_score(y['val'], s['val']))):
            if v > best[m]:
                best[m] = v
                cap[m] = s
    return cap, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default='.')
    p.add_argument('--net-dir', required=True)
    p.add_argument('--task', required=True, choices=list(TASKS))
    p.add_argument('--arm', default='posw2')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--aggr', default='min')
    p.add_argument('--epochs', type=int, default=301)
    p.add_argument('--seeds', nargs='+', type=int, default=[666, 0, 1, 2, 3])
    p.add_argument('--mode', default='additive',
                   choices=['none', 'concat', 'additive'],
                   help="'none' is the control: identical code path with the "
                        'pair pathway removed')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res = []
    for seed in args.seeds:
        cap, y = run_one(args.run_dir, args.net_dir, args.task, seed, device,
                         args.arm, args.lr, args.channels, args.aggr,
                         args.epochs, mode=args.mode)
        r = dict(task=args.task, seed=seed, mode=args.mode)
        for f in ('val', 'test'):
            r[f'{f}_auc'] = roc_auc_score(y[f], cap['auc'][f])
            r[f'{f}_aupr'] = average_precision_score(y[f], cap['aupr'][f])
        res.append(r)
        print(f"  seed={seed:<4d} val {r['val_auc']:.4f}/{r['val_aupr']:.4f} "
              f"test {r['test_auc']:.4f}/{r['test_aupr']:.4f}", flush=True)
    df = pd.DataFrame(res)
    df.to_csv(args.out, index=False)
    print(f"\n{args.task} hybrid (mode={args.mode}), {len(res)} seeds")
    for f in ('val', 'test'):
        print(f"  {f}: AUC {df[f'{f}_auc'].mean():.4f} +- "
              f"{df[f'{f}_auc'].std():.4f}   AUPR {df[f'{f}_aupr'].mean():.4f} "
              f"+- {df[f'{f}_aupr'].std():.4f}")


if __name__ == '__main__':
    main()
