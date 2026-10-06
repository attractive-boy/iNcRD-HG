"""Assembles the frozen data, pair channels, fusion head and training recipe."""
import torch
import torch.nn.functional as F

from .config import ModelConfig
from .data import structure_data
from .model import Model
from .pair_channels import PairStruct

NOTE = ("Pinned original iNcRD-HG encoder (GeneralConv x2, hidden 256, relation mean); "
        "M1 QMP exact query-edge correction; M2 SP-RNCN (co,knn,dknn; extended path statistics); "
        "M3 MC-EASE (lambda=10,50, both sides); shared training-positive graph only; "
        "validation-AUC checkpoint selection; RMSprop 1e-4, alpha .99, 300 full-batch epochs.")


class Adapter:
    def __init__(self, data, graph, device, cfg=None):
        self.cfg = cfg or ModelConfig()
        self.graph = graph.to(device)
        self.D = structure_data(self.graph)
        pos = self.graph["miRNA", "MiDrug", "drug"].edge_index.t()
        self.ps = PairStruct(self.D, pos, self.cfg)
        self.pairs = {p: torch.as_tensor(data[p + "_pairs"].T, dtype=torch.long, device=device)
                      for p in ("train", "val", "test")}
        self.y = torch.as_tensor(data["train_y"], dtype=torch.float, device=device)
        raw = self.ps.qis(*self.pairs["train"])
        self.mu, self.sd = raw.mean(0, keepdim=True), raw.std(0, keepdim=True) + 1e-6
        self.qis = {p: (self.ps.qis(*idx) - self.mu) / self.sd for p, idx in self.pairs.items()}
        self.ease = {p: self.ps.ease(*idx) for p, idx in self.pairs.items()}
        self.dense = (self.D.xr, self.D.xd, self.D.Sr1, self.D.Sd1, self.ps.A)
        self.model = Model(self.graph, self.ps, raw.shape[1], self.ease["train"].shape[1], self.cfg).to(device)
        self.note = NOTE
        with torch.no_grad():
            self.model.encoder(self.graph.x_dict, self.graph.edge_index_dict)

    def make_optimizer(self, args):
        return torch.optim.RMSprop(self.model.parameters(), lr=args.lr, alpha=0.99, weight_decay=args.wd)

    def logits(self, part):
        return self.model(self.pairs[part], self.ps, self.qis[part], self.ease[part], self.dense)

    def train_epoch(self, optimizer, epoch):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(self.logits("train"), self.y)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def predict(self, part):
        return self.logits(part).sigmoid().cpu().numpy()
