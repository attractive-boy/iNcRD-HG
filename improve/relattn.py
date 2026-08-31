# Learnable relation-level semantic attention, replacing to_hetero's fixed aggr.
#
# The released model wraps GeneralConv in `to_hetero(..., aggr=...)`, which
# gives each relation its own parameters and then fuses the per-relation
# messages arriving at a node with a FIXED elementwise reduction -- 'min' for
# LncDrug, 'mean' for MiDrug.  'min' is the strong claim: any one relation can
# suppress every other relation on any dimension, and nothing in training can
# change that.
#
# The paper's "semantic enhancement" is that per-relation parameterisation.  The
# fusion step, where a semantic-attention model would actually live, is not
# learned at all.  This module makes it learned: HAN-style attention over the
# relation dimension, with the weights produced from the messages themselves.
#
#     beta_r = softmax_r( mean_n q^T tanh(W h_r[n] + b) )
#     z[n]   = sum_r beta_r * h_r[n]
#
# beta is per (node type, layer) and shared across nodes of that type, which is
# HAN's semantic attention exactly -- it asks "which relations matter for this
# node type", not "which neighbour matters", and it stays interpretable.
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleDict, Sequential, Tanh
from torch_geometric.nn import GeneralConv


def _key(rel):
    return '__'.join(rel)


class SemanticAttention(torch.nn.Module):
    """Attention over the relation axis for one node type."""

    def __init__(self, hidden=128):
        super().__init__()
        self.project = Sequential(torch.nn.LazyLinear(hidden), Tanh(),
                                  Linear(hidden, 1, bias=False))

    def forward(self, zs):
        if len(zs) == 1:
            return zs[0], None
        stacked = torch.stack(zs, dim=1)            # [N, R, D]
        w = self.project(stacked).mean(dim=0)       # [R, 1]
        beta = torch.softmax(w, dim=0)
        return (stacked * beta.unsqueeze(0)).sum(dim=1), beta.detach().squeeze(-1)


class RelAttnEncoder(torch.nn.Module):
    """Two GeneralConv layers, relation messages fused by learned attention."""

    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        node_types, edge_types = metadata
        self.edge_types = list(edge_types)
        self.conv1 = ModuleDict(
            {_key(r): GeneralConv((-1, -1), hidden_channels) for r in edge_types})
        self.conv2 = ModuleDict(
            {_key(r): GeneralConv((-1, -1), out_channels) for r in edge_types})
        self.att1 = ModuleDict({nt: SemanticAttention() for nt in node_types})
        self.att2 = ModuleDict({nt: SemanticAttention() for nt in node_types})
        self.last_beta = {}

    def _layer(self, convs, atts, x_dict, edge_index_dict, tag):
        msgs = {nt: [] for nt in x_dict}
        for rel in self.edge_types:
            if rel not in edge_index_dict:
                continue
            src, _, dst = rel
            h = convs[_key(rel)]((x_dict[src], x_dict[dst]), edge_index_dict[rel])
            msgs[dst].append(h)
        out = {}
        for nt, zs in msgs.items():
            if not zs:
                continue
            out[nt], beta = atts[nt](zs)
            if beta is not None:
                self.last_beta[f'{tag}:{nt}'] = beta
        return out

    def forward(self, x_dict, edge_index_dict):
        h = self._layer(self.conv1, self.att1, x_dict, edge_index_dict, 'l1')
        h = {k: F.leaky_relu(v, negative_slope=0.1) for k, v in h.items()}
        return self._layer(self.conv2, self.att2, h, edge_index_dict, 'l2')
