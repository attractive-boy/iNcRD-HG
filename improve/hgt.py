# Heterogeneous Graph Transformer encoder (Hu et al. 2020).
#
# The released model is to_hetero(GeneralConv, aggr=...): each relation gets its
# own weights, then a FIXED elementwise reduction fuses them.  HGT replaces both
# halves -- relation-specific key/query/value projections plus learned attention
# over the incoming edges, and attention-weighted fusion across relations.  It
# is the standard realisation of "semantic-enhanced heterogeneous graph
# learning" and is the largest architectural change tested here.
import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class HGTEncoder(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, heads=2, n_layers=2):
        super().__init__()
        self.lin = torch.nn.ModuleDict(
            {nt: torch.nn.LazyLinear(hidden_channels) for nt in metadata[0]})
        self.convs = torch.nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, heads=heads)
            for _ in range(n_layers)])

    def forward(self, x_dict, edge_index_dict):
        h = {nt: F.leaky_relu(self.lin[nt](x), 0.1) for nt, x in x_dict.items()}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {k: F.leaky_relu(v, 0.1) for k, v in h.items()}
        return h
