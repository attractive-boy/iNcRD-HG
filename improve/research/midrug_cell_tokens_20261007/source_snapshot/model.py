"""Original expression columns as attribute tokens; no extra graph node types."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from common import Adapter, make_graph, Reconstruction, base

CONFIGS = {name+suffix: dict(conditioned=name == 'drug', hidden=32, chunk_size=256,
           branch_lr=3e-4, reconstruction_weight=weight)
           for name in ['drug', 'mean'] for suffix, weight in [('', 0.), ('_rec10', 10.)]}


class CellTokens(nn.Module):
    def __init__(self, n_rnas, drug_features, hidden=32, conditioned=True, chunk_size=256):
        super().__init__()
        self.hidden, self.conditioned, self.chunk_size = hidden, conditioned, chunk_size
        self.cell_encoder = nn.Sequential(nn.LayerNorm(n_rnas), nn.Linear(n_rnas, hidden), nn.LeakyReLU(.1))
        self.expression = nn.Linear(1, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.query = nn.Sequential(nn.LayerNorm(drug_features), nn.Linear(drug_features, hidden),
                                   nn.LeakyReLU(.1), nn.Linear(hidden, hidden, bias=False))
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.LeakyReLU(.1))

    def encode(self, xr, xd):
        cells = self.cell_encoder(xr.T)
        tokens = F.leaky_relu(cells[None, :, :]+self.expression(xr[:, :, None]), .1)
        queries = self.query(xd)
        if not self.conditioned:
            queries = queries.mean(0, keepdim=True).expand_as(queries)
        return self.key(tokens), self.value(tokens), queries

    def attend(self, keys, values, query, r, d):
        logits = torch.bmm(keys[r], query[d, :, None]).squeeze(-1)/math.sqrt(self.hidden)
        weights = logits.softmax(-1)
        pooled = torch.bmm(weights[:, None, :], values[r]).squeeze(1)
        return self.readout(pooled), weights

    def forward(self, xr, xd, r, d):
        keys, values, query = self.encode(xr, xd)
        pieces = []
        for start in range(0, len(r), self.chunk_size):
            out, _ = self.attend(keys, values, query, r[start:start+self.chunk_size], d[start:start+self.chunk_size])
            pieces.append(out)
        if not pieces:
            return xr.new_empty((0, self.hidden))
        return torch.cat(pieces)


class CellModel(base.Model):
    def forward(self, eli, ps, qis, ease, dense):
        r, d = eli
        addition = self.cell_projection(self.cell_context(dense[0], dense[1], r, d))
        hook = self.lin1.register_forward_hook(lambda module, args, out: out+addition)
        try:
            return super().forward(eli, ps, qis, ease, dense)
        finally:
            hook.remove()


def make_adapter(data, name, device):
    a = Adapter(data, make_graph(data), device)
    if name == 'control':
        return a
    if name == 'unmasked10' or CONFIGS[name]['reconstruction_weight']:
        a.model.add_module('reconstruction', Reconstruction(a.graph, a.cfg.hidden).to(device))
    if name == 'unmasked10':
        return a
    cfg = CONFIGS[name]
    m = a.model; m.__class__ = CellModel
    m.cell_context = CellTokens(a.graph['miRNA'].num_nodes, a.graph['drug'].x.shape[1],
                               cfg['hidden'], cfg['conditioned'], cfg['chunk_size']).to(device)
    m.cell_projection = nn.Linear(cfg['hidden'], m.lin1.out_features, bias=False).to(device)
    nn.init.zeros_(m.cell_projection.weight)
    return a


def make_optimizer(m, cfg):
    groups = [dict(params=[p for name, p in m.named_parameters()
                          if not name.startswith(('cell_context.', 'cell_projection.', 'reconstruction.'))], lr=1e-4),
              dict(params=list(m.cell_context.parameters())+list(m.cell_projection.parameters()),
                   lr=cfg['branch_lr'], weight_decay=1e-4)]
    if cfg['reconstruction_weight']:
        groups.append(dict(params=m.reconstruction.parameters(), lr=1e-4, weight_decay=1e-4))
    return torch.optim.RMSprop(groups, alpha=.99)


def mask_generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)
