import copy, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GeneralConv, to_hetero
from sklearn.metrics import roc_auc_score, average_precision_score

PTH = "/tmp/opencode/mphgnn_repo/MiDrug_data_end.pth"
EDGE = ("miRNA", "MiDrug", "drug")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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

    def forward(self, z_dict, eli):
        r, c = eli
        z = torch.cat([z_dict["miRNA"][r], z_dict["drug"][c]], dim=-1)
        z = F.leaky_relu(self.lin1(z), negative_slope=0.1)
        z = F.leaky_relu(self.lin2(z), negative_slope=0.1)
        return self.lin3(z).view(-1)


class Model(torch.nn.Module):
    def __init__(self, data, hidden, aggr):
        super().__init__()
        self.encoder = to_hetero(GNNEncoder(hidden, hidden), data.metadata(), aggr=aggr)
        self.decoder = EdgeDecoder(hidden)

    def forward(self, xd, ed, eli):
        return self.decoder(self.encoder(xd, ed), eli)


def pair_feats(z, eli):
    r, c = eli
    zm = z["miRNA"][r]
    zd = z["drug"][c]
    return torch.cat([zm, zd, zm * zd, (zm - zd).abs()], dim=-1)


def main():
    dev = torch.device("cuda:0")
    D = torch.load(PTH, weights_only=False).cpu()
    nm, nd = D["miRNA"].num_nodes, D["drug"].num_nodes
    bench = D[EDGE].edge_index
    test_eli = D[EDGE].edge_label_index
    test_lab = D[EDGE].edge_label

    pos_set = set(map(tuple, bench.T.tolist()))
    rng = np.random.default_rng(666)
    neg = []
    while len(neg) < bench.shape[1]:
        i = int(rng.integers(0, nm)); j = int(rng.integers(0, nd))
        if (i, j) not in pos_set:
            neg.append((i, j))
    neg = torch.tensor(neg, dtype=torch.long).T

    train = copy.deepcopy(D)
    train[EDGE].edge_index = bench
    train[EDGE].edge_label_index = torch.cat([bench, neg], dim=1)
    train[EDGE].edge_label = torch.cat([torch.ones(bench.shape[1]), torch.zeros(neg.shape[1])]).float()

    tpos = test_eli[:, test_lab == 1]
    test = copy.deepcopy(D)
    test[EDGE].edge_index = tpos
    test["drug", "rev_MiDrug", "miRNA"].edge_index = tpos.flip(0)

    train = train.to(dev); test = test.to(dev)
    set_seed(666)
    model = Model(train, 256, "mean").to(dev)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99)
    for ep in range(1, 251):
        model.train(); opt.zero_grad()
        p = model(train.x_dict, train.edge_index_dict, train[EDGE].edge_label_index)
        F.binary_cross_entropy_with_logits(p, train[EDGE].edge_label).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        ztr = model.encoder(train.x_dict, train.edge_index_dict)
        zte = model.encoder(test.x_dict, test.edge_index_dict)
        Xtr = pair_feats(ztr, train[EDGE].edge_label_index).cpu().numpy()
        Xte = pair_feats(zte, test[EDGE].edge_label_index).cpu().numpy()
    ytr = train[EDGE].edge_label.cpu().numpy()
    yte = test_lab.numpy()

    np.savez_compressed("/tmp/opencode/mphgnn_pair_emb.npz",
                        X_train=Xtr.astype(np.float32), y_train=ytr,
                        X_test=Xte.astype(np.float32), y_test=yte)
    # quick sklearn reference on the embeddings
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=666)
    clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    print("HistGB on GNN pair embeddings: AUC %.4f  AUPR %.4f" % (roc_auc_score(yte, s), average_precision_score(yte, s)))
    print("saved /tmp/opencode/mphgnn_pair_emb.npz", Xtr.shape, Xte.shape)


if __name__ == "__main__":
    main()
