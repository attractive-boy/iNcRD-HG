import argparse, copy, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GeneralConv, to_hetero
from sklearn.metrics import roc_auc_score, average_precision_score

PTH = "/tmp/opencode/mphgnn_repo/MiDrug_data_end.pth"
EDGE = ("miRNA", "MiDrug", "drug")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load():
    d = torch.load(PTH, weights_only=False)
    return d.cpu()


def sample_neg(num_mir, num_drug, pos_set, n, rng):
    out = []
    while len(out) < n:
        i = int(rng.integers(0, num_mir)); j = int(rng.integers(0, num_drug))
        if (i, j) not in pos_set:
            out.append((i, j))
    return torch.tensor(out, dtype=torch.long).T


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden, out):
        super().__init__()
        self.conv1 = GeneralConv((-1, -1), hidden)
        self.conv2 = GeneralConv((-1, -1), out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        return self.conv2(x, edge_index)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lin1 = Linear(2 * hidden, 2 * hidden)
        self.lin2 = Linear(2 * hidden, hidden)
        self.lin3 = Linear(hidden, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict["miRNA"][row], z_dict["drug"][col]], dim=-1)
        z = F.leaky_relu(self.lin1(z), negative_slope=0.1)
        z = F.leaky_relu(self.lin2(z), negative_slope=0.1)
        return self.lin3(z).view(-1)


class Model(torch.nn.Module):
    def __init__(self, data, hidden, aggr):
        super().__init__()
        self.encoder = to_hetero(GNNEncoder(hidden, hidden), data.metadata(), aggr=aggr)
        self.decoder = EdgeDecoder(hidden)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        return self.decoder(self.encoder(x_dict, edge_index_dict), edge_label_index)


def run(seed, epochs, device):
    D = load()
    num_mir = D["miRNA"].num_nodes
    num_drug = D["drug"].num_nodes
    bench = D[EDGE].edge_index
    test_eli = D[EDGE].edge_label_index
    test_lab = D[EDGE].edge_label

    pos_set = set(map(tuple, bench.T.tolist()))
    rng = np.random.default_rng(seed)
    neg = sample_neg(num_mir, num_drug, pos_set, bench.shape[1], rng)

    train = copy.deepcopy(D)
    train[EDGE].edge_label_index = torch.cat([bench, torch.stack([neg[0], neg[1]])], dim=1)
    train[EDGE].edge_label = torch.cat([torch.ones(bench.shape[1]), torch.zeros(neg.shape[1])]).float()

    tpos = test_eli[:, test_lab == 1]
    test = copy.deepcopy(D)
    test[EDGE].edge_index = tpos
    test["drug", "rev_MiDrug", "miRNA"].edge_index = tpos.flip(0)
    test[EDGE].edge_label_index = test_eli
    test[EDGE].edge_label = test_lab.float()

    train = train.to(device); test = test.to(device)
    set_seed(seed)
    model = Model(train, 256, "mean").to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99)

    y = test[EDGE].edge_label.cpu().numpy()
    best = (-1, -1, 0); last = (0, 0, 0)
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        p = model(train.x_dict, train.edge_index_dict, train[EDGE].edge_label_index)
        loss = F.binary_cross_entropy_with_logits(p, train[EDGE].edge_label)
        loss.backward(); opt.step()
        if ep % 25 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                pp = model(test.x_dict, test.edge_index_dict, test[EDGE].edge_label_index).cpu().numpy()
            a = roc_auc_score(y, pp); r = average_precision_score(y, pp)
            last = (a, r, ep)
            if a > best[0]:
                best = (a, r, ep)
    return last, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[666, 0, 1, 2, 3])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lasts, bests = [], []
    for s in a.seeds:
        l, b = run(s, a.epochs, dev)
        lasts.append(l); bests.append(b)
        print(f"seed {s}: last AUC/AUPR={l[0]:.4f}/{l[1]:.4f} (ep{l[2]})  best={b[0]:.4f}/{b[1]:.4f} (ep{b[2]})", flush=True)
    lasts = np.array(lasts); bests = np.array(bests)
    print("\n===== iNcRD-HG architecture on MPHGNN data + MPHGNN independent test =====")
    print("iNcRD-HG (last epoch)  AUC %.4f±%.4f  AUPR %.4f±%.4f" % (lasts[:, 0].mean(), lasts[:, 0].std(), lasts[:, 1].mean(), lasts[:, 1].std()))
    print("iNcRD-HG (best-on-test) AUC %.4f±%.4f  AUPR %.4f±%.4f" % (bests[:, 0].mean(), bests[:, 0].std(), bests[:, 1].mean(), bests[:, 1].std()))
    print("MPHGNN reported                        AUC 0.9780       AUPR 0.9370")
