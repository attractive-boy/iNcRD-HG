"""Fusion head of iNcRD-HG + M1 + M2 + M3.

The head concatenates the two query endpoint states (with the query edge's
messages removed, M1), the pooled intermediate states of the simple 2-hop paths
(M2), the query-invariant graph statistics and the masked-cell EASE scores
(M3), followed by the three-layer MLP.
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear

from .encoder import QueryHollowEncoder


class Model(torch.nn.Module):
    def __init__(self, graph, ps, n_qis, n_ease, cfg):
        super().__init__()
        self.encoder = QueryHollowEncoder(graph, cfg.hidden, cfg.layers)
        d = 2 * cfg.hidden
        self.ncn_lin = torch.nn.ModuleList([Linear(cfg.hidden, w) for w in ps.ncn_widths(cfg)])
        d += sum(ps.ncn_widths(cfg))
        self.pc = torch.nn.Sequential(Linear(n_qis, cfg.hidden), torch.nn.LeakyReLU(0.1),
                                      Linear(cfg.hidden, cfg.hidden))
        d += cfg.hidden
        self.eb = torch.nn.BatchNorm1d(n_ease)
        self.em = torch.nn.Sequential(Linear(n_ease, cfg.hidden), torch.nn.LeakyReLU(0.1),
                                      Linear(cfg.hidden, cfg.hidden))
        d += cfg.hidden
        self.lin1 = Linear(d, 2 * cfg.hidden)
        self.lin2 = Linear(2 * cfg.hidden, cfg.hidden)
        self.lin3 = Linear(cfg.hidden, 1)

    def forward(self, eli, ps, qis, ease, dense):
        r, d = eli
        xr, xd, Sr1, Sd1, A = dense
        zr, zd, zm, zdd, corr = self.encoder.forward_dense(xr, xd, Sr1, Sd1, A, r, d, ps.qvals(r, d))
        z = {"miRNA": zr, "drug": zd}
        parts = [zm, zdd]
        parts += [lin(e) for lin, e in zip(self.ncn_lin, ps.qncn(r, d, z, corr))]
        parts.append(self.pc(qis))
        parts.append(self.em(self.eb(ease)))
        h = torch.cat(parts, -1)
        h = F.leaky_relu(self.lin1(h), 0.1)
        h = F.leaky_relu(self.lin2(h), 0.1)
        return self.lin3(h).view(-1)
