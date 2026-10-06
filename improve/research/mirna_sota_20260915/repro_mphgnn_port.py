"""PyG port of MPHGNN (metapath-guided heterogeneous GNN) for local assessment.

Faithful-in-structure reproduction of the official model in
/tmp/opencode/mphgnn_repo (layer.py, model.py, load_data.py):
  HeteroLinear -> 3x [HeteroGraphConv(GraphConv per relation, sum) + PReLU]
  -> layer attention over stacked layers -> BiTrans metapath aggregator
  -> MIL attention bag predictor + top-k instance predictor
  -> global GCN encoder + MLP edge decoder over the miRNA-drug pair graph.
Metapath instances follow meta_path_instance exactly (k*(k+2)+1 padding).
"""
import argparse, copy, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GraphConv, HeteroConv
from sklearn.metrics import roc_auc_score, average_precision_score

PTH = "/tmp/opencode/mphgnn_repo/MiDrug_data_end.pth"
EDGE = ("miRNA", "MiDrug", "drug")
REV = ("drug", "rev_MiDrug", "miRNA")
REL = {  # official graph_data relations
    ("miRNA", "Migene", "gene"): ("miRNA", "gene"),
    ("gene", "Genedrug", "drug"): ("gene", "drug"),
    EDGE: ("miRNA", "drug"),
    REV: ("drug", "miRNA"),
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load():
    return torch.load(PTH, weights_only=False).cpu()


class HeteroLinear(nn.Module):
    def __init__(self, in_feats, hidden):
        super().__init__()
        self.linears = nn.ModuleDict({str(k): nn.Linear(v, hidden) for k, v in in_feats.items()})
        for lin in self.linears.values():
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, feat):
        return {k: self.linears[str(k)](v) for k, v in feat.items()}


class NodeEmbedding(nn.Module):
    def __init__(self, rels, hidden):
        super().__init__()
        self.conv = HeteroConv({r: GraphConv(hidden, hidden) for r in rels}, aggr='sum')
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict):
        h = self.conv(x_dict, edge_index_dict)
        h = {k: self.prelu(v) for k, v in h.items()}
        return h


class LayerAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))

    def forward(self, z):  # z: [N, L, H]
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0).expand((z.shape[0],) + w.shape)
        return (beta * z).sum(1)


class MetaPathAggregator(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.a_dd = nn.Linear(hidden, hidden, bias=False)
        self.a_du = nn.Linear(hidden, hidden, bias=False)
        self.a_d = nn.Linear(hidden, hidden // 2, bias=False)
        self.a_u = nn.Linear(hidden, hidden // 2, bias=False)
        for m in (self.a_dd, self.a_du, self.a_d, self.a_u):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, feat, mp):  # mp: [B, n_inst, 4] (miRNA, gene1, gene2, drug)
        f = torch.cat([feat['miRNA'][mp[:, :, 0:1]], feat['gene'][mp[:, :, 1:2]],
                       feat['gene'][mp[:, :, 2:3]], feat['drug'][mp[:, :, 3:4]]], dim=2)  # [B,I,4,H]
        B, I, _, H = f.shape
        f = f.reshape(B, I, 4 * H)
        dis = (((self.a_dd((f[:, :, :H] + f[:, :, H:2 * H]) / 2) + f[:, :, 2 * H:3 * H]) / 2)
               + f[:, :, 3 * H:]) / 2
        dru = (((self.a_du((f[:, :, 3 * H:] + f[:, :, 2 * H:3 * H]) / 2) + f[:, :, H:2 * H]) / 2)
               + f[:, :, :H]) / 2
        return torch.cat((self.a_d(dru), self.a_u(dis)), dim=2)


class MILNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))

    def forward(self, ins):
        attn = torch.softmax(self.project(ins), dim=1)
        return (ins * attn).sum(1), attn


class InstanceNet(nn.Module):
    def __init__(self, hidden, k=3):
        super().__init__()
        self.k = k
        self.weights = nn.Linear(hidden // 2, hidden // 2, bias=False)
        nn.init.xavier_uniform_(self.weights.weight)

    def forward(self, ins, attn):
        m = ins[:, :, :ins.shape[-1] // 2]
        d = ins[:, :, ins.shape[-1] // 2:]
        pred = torch.matmul(self.weights(m).reshape(m.shape[0], m.shape[1], 1, m.shape[2]), d.unsqueeze(-1)).squeeze(3)
        return torch.mean((attn * pred).topk(k=self.k, dim=1)[0], dim=1)


class GNNEncoder(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(layers):
            self.convs.append(GCNConv(in_ch if i == 0 else hidden, out_ch if i == layers - 1 else hidden))
            self.bns.append(nn.BatchNorm1d(out_ch if i == layers - 1 else hidden))
        self.dropout = nn.Dropout(0.5)
        self.act = nn.ELU()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.act(self.bns[i](conv(self.dropout(x), edge_index)))
        return self.act(self.bns[-1](self.convs[-1](self.dropout(x), edge_index)))


class EdgeDecoder(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, layers=2):
        super().__init__()
        self.mlps = nn.ModuleList([nn.Linear(in_ch if i == 0 else hidden, out_ch if i == layers - 1 else hidden) for i in range(layers)])
        self.dropout = nn.Dropout(0.5)
        self.act = nn.ELU()

    def forward(self, z, edge):
        x = z[edge[0]] * z[edge[1]]
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.act(mlp(self.dropout(x)))
        return self.mlps[-1](x)


class Model(nn.Module):
    def __init__(self, in_feats, hidden=128, num_layers=3, k=3):
        super().__init__()
        self.lin = HeteroLinear(in_feats, hidden)
        self.layers = nn.ModuleList([NodeEmbedding(list(REL), hidden) for _ in range(num_layers)])
        self.att_mi = LayerAttention(hidden)
        self.att_dr = LayerAttention(hidden)
        self.att_ge = LayerAttention(hidden)
        self.agg = MetaPathAggregator(hidden)
        self.mil = MILNet(hidden)
        self.bag_pred = nn.Linear(hidden, 1, bias=False)
        self.ins_pred = InstanceNet(hidden, k)
        self.enc = GNNEncoder(hidden, hidden, 2 * hidden, 2)
        self.dec = EdgeDecoder(2 * hidden, 64, 1, 2)

    def forward(self, x_dict, edge_index_dict, mp):
        h = self.lin(x_dict)
        stack = {t: [h[t]] for t in h}
        for layer in self.layers:
            h = layer(h, edge_index_dict)
            for t in h:
                stack[t].append(h[t])
        hs = {t: torch.stack(v, dim=1) for t, v in stack.items()}
        h = {'miRNA': self.att_mi(hs['miRNA']), 'drug': self.att_dr(hs['drug']), 'gene': self.att_ge(hs['gene'])}
        ins = self.agg(h, mp)
        bag, attn = self.mil(ins)
        pred_bag = self.bag_pred(bag)
        pred_ins = self.ins_pred(ins, attn)
        pairs = mp[:, 0, [0, 3]].T.contiguous()
        pairs = torch.stack([pairs[0], pairs[1] + x_dict['miRNA'].shape[0]])
        z = self.dec(self.enc(torch.cat([h['miRNA'], h['drug']], 0), pairs), pairs)
        return (pred_ins + pred_bag + z) / 3


def build_links(D):
    gd = D["gene", "Genedrug", "drug"].edge_index.T.cpu().numpy()
    mg = D["miRNA", "Migene", "gene"].edge_index.T.cpu().numpy()
    gg = D["gene", "GeGe", "gene"].edge_index
    gg = torch.cat([gg, gg.flip(0)], 1).T.cpu().numpy()
    return gd, mg, gg


def meta_path_instance(mi, dr, gd, mg, gg, k, rng):
    mpi = [[mi, g1, g2, dr] for g1 in mg[mg[:, 0] == mi][:, 1] for g2 in gd[gd[:, 1] == dr][:, 0]]
    mpi += [[mi, g1, g2, dr] for g1 in mg[mg[:, 0] == mi][:, 1] for g2 in gg[gg[:, 0] == g1][:, 1]]
    mpi += [[mi, g1, g2, dr] for g2 in gd[gd[:, 1] == dr][:, 0] for g1 in gg[gg[:, 0] == g2][:, 1]]
    if not mpi:
        allg = np.unique(gd[:, 0]) if len(gd) else np.unique(mg[:, 1])
        rg = int(rng.choice(allg))
        mpi.append([mi, rg, rg, dr])
    target = k * (k + 2) + 1
    if len(mpi) < target:
        idx = rng.integers(0, len(mpi), size=target - len(mpi))
        mpi = mpi + [mpi[i] for i in idx]
    else:
        mpi = mpi[:target]
    return mpi


def build_instances(pairs, gd, mg, gg, k, seed):
    rng = np.random.default_rng(seed)
    return np.asarray([meta_path_instance(int(m), int(d), gd, mg, gg, k, rng) for m, d in pairs.T], dtype=np.int64)


def run(seed, fold_count, epochs, k, device, batch=256, lr=5e-4, transductive=False):
    D = load()
    n_mi = D["miRNA"].num_nodes
    bench = D[EDGE].edge_index
    pos = set(map(tuple, bench.T.tolist()))
    gd, mg, gg = build_links(D)
    x_dict = {'miRNA': D['miRNA'].x.float(), 'drug': D['drug'].x.float(), 'gene': D['gene'].x.float()}
    test_pairs = D[EDGE].edge_label_index
    test_y = D[EDGE].edge_label.float()

    # benchmark + equal random negatives, then 5-fold
    rng = np.random.default_rng(seed)
    negs = []
    while len(negs) < bench.shape[1]:
        i, j = int(rng.integers(0, n_mi)), int(rng.integers(0, D['drug'].num_nodes))
        if (i, j) not in pos:
            negs.append((i, j))
    bench_pairs = np.concatenate([bench.T.cpu().numpy(), np.asarray(negs)], 0)
    bench_y = np.concatenate([np.ones(bench.shape[1]), np.zeros(len(negs))])
    order = np.random.default_rng(seed + 1).permutation(len(bench_y))
    bench_pairs, bench_y = bench_pairs[order], bench_y[order]

    gd_t = torch.as_tensor(gd.T)
    mg_t = torch.as_tensor(mg.T)
    gg_t = torch.as_tensor(gg.T)
    # edges for message passing: benchmark positives only (official g uses benchmark)
    mp_edges = {('miRNA', 'Migene', 'gene'): mg_t, ('gene', 'Genedrug', 'drug'): gd_t,
                EDGE: torch.as_tensor(bench.cpu().numpy()), REV: torch.as_tensor(bench.flip(0).cpu().numpy())}
    x_dict = {kk: vv.to(device) for kk, vv in x_dict.items()}
    edge_index_dict = {kk: vv.to(device) for kk, vv in mp_edges.items()}

    # precompute metapath instances for benchmark and test
    all_bench_mp = build_instances(bench_pairs.T, gd, mg, gg, k, seed + 2)
    test_pos = test_pairs[:, test_y == 1].cpu().numpy()
    test_mp = build_instances(test_pairs.cpu().numpy(), gd, mg, gg, k, seed + 3)
    if transductive:
        te_all = torch.as_tensor(np.concatenate([bench.cpu().numpy(), test_pos], 1), device=device)
        te_graph = {('miRNA', 'Migene', 'gene'): mg_t.to(device), ('gene', 'Genedrug', 'drug'): gd_t.to(device),
                    EDGE: te_all, REV: te_all.flip(0)}
    else:
        te_graph = edge_index_dict
    bench_y_t = torch.as_tensor(bench_y, dtype=torch.float32, device=device)
    test_mp_t = torch.as_tensor(test_mp, device=device)

    folds = np.array_split(np.arange(len(bench_y)), fold_count)
    aucs, aprs = [], []
    for f, val_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(bench_y)), val_idx)
        tr_mp = torch.as_tensor(all_bench_mp[train_idx], device=device)
        tr_y = bench_y_t[train_idx]
        va_mp = torch.as_tensor(all_bench_mp[val_idx], device=device)
        va_y = bench_y_t[val_idx]
        set_seed(seed * 100 + f)
        model = Model({'miRNA': x_dict['miRNA'].shape[1], 'drug': x_dict['drug'].shape[1],
                       'gene': x_dict['gene'].shape[1]}).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        pw = torch.tensor(float((tr_y == 0).sum() / (tr_y == 1).sum()), device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        best = None
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(tr_y), device=device)
            for s in range(0, len(tr_y), batch):
                ix = perm[s:s + batch]
                opt.zero_grad()
                out = model(x_dict, edge_index_dict, tr_mp[ix]).view(-1)
                loss = crit(out, tr_y[ix])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                va_out = model(x_dict, edge_index_dict, va_mp).view(-1)
                va_auc = roc_auc_score(va_y.cpu().numpy(), va_out.cpu().numpy())
            if best is None or va_auc > best[0]:
                best = (va_auc, copy.deepcopy(model.state_dict()))
        model.load_state_dict(best[1])
        model.eval()
        with torch.no_grad():
            out = model(x_dict, te_graph, test_mp_t).view(-1).cpu().numpy()
        aucs.append(roc_auc_score(test_y.numpy(), out))
        aprs.append(average_precision_score(test_y.numpy(), out))
        print(f"  fold {f}: val AUC {best[0]:.4f}  test AUC/AUPR {aucs[-1]:.4f}/{aprs[-1]:.4f}", flush=True)
    return float(np.mean(aucs)), float(np.mean(aprs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[666, 0, 1])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--transductive", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    res = []
    for s in a.seeds:
        t = time.time()
        auc, ap_ = run(s, a.folds, a.epochs, a.k, dev, transductive=a.transductive)
        res.append((auc, ap_))
        print(f"seed {s}: AUC {auc:.4f} AUPR {ap_:.4f} ({time.time()-t:.0f}s)", flush=True)
    r = np.array(res)
    print("\n===== MPHGNN PyG port (5-fold, independent test) =====")
    print("AUC %.4f±%.4f  AUPR %.4f±%.4f" % (r[:, 0].mean(), r[:, 0].std(), r[:, 1].mean(), r[:, 1].std()))
    print("MPHGNN reported   AUC 0.9780  AUPR 0.9370")
    print("SubKNet reported  AUC 0.9717  AUPR 0.9341")
    print("prior weak proxy  AUC 0.9514  AUPR 0.8922")
