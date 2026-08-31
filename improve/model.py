# Model variants for iNcRD-HG, evaluated on the released protocol.
#
# The author's architecture is the `base` arm and is reproduced verbatim from
# predict_association.py: two lazy GeneralConv layers wrapped by to_hetero,
# LeakyReLU(0.1) between them, and a three-layer MLP decoder over the
# concatenated ncRNA/drug embeddings.  Every other arm is a single named
# switch on top of that, so a delta can be attributed to one change.
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GeneralConv, to_hetero


class GNNEncoder(torch.nn.Module):
    """The author's encoder.  `norm` and `residual` are off by default."""

    def __init__(self, hidden_channels, out_channels, norm=False, residual=False):
        super().__init__()
        self.conv1 = GeneralConv((-1, -1), hidden_channels)
        self.conv2 = GeneralConv((-1, -1), out_channels)
        self.norm = torch.nn.LayerNorm(hidden_channels) if norm else None
        self.residual = residual

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        if self.norm is not None:
            h = self.norm(h)
        h = F.leaky_relu(h, negative_slope=0.1)
        out = self.conv2(h, edge_index)
        if self.residual:
            out = out + h
        return out


class EdgeDecoder(torch.nn.Module):
    """Three-layer MLP decoder.

    `mode='concat'` is the author's [z_rna || z_drug].  `mode='rich'` also feeds
    the Hadamard product and absolute difference, so the MLP does not have to
    learn multiplicative interaction terms from a concatenation alone.
    """

    def __init__(self, hidden_channels, rna_type, mode='concat'):
        super().__init__()
        assert mode in ('concat', 'rich')
        self.mode = mode
        self.rna_type = rna_type
        in_dim = (2 if mode == 'concat' else 4) * hidden_channels
        self.lin1 = Linear(in_dim, 2 * hidden_channels)
        self.lin2 = Linear(2 * hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z_r = z_dict[self.rna_type][row]
        z_d = z_dict['drug'][col]
        if self.mode == 'concat':
            z = torch.cat([z_r, z_d], dim=-1)
        else:
            z = torch.cat([z_r, z_d, z_r * z_d, (z_r - z_d).abs()], dim=-1)
        z = self.lin1(z)
        z = F.leaky_relu(z, negative_slope=0.1)
        z = self.lin2(z)
        z = F.leaky_relu(z, negative_slope=0.1)
        return self.lin3(z).view(-1)


class Model(torch.nn.Module):
    def __init__(self, data, hidden_channels, aggr_str, rna_type,
                 decoder='concat', norm=False, residual=False, encoder='hetero'):
        super().__init__()
        if encoder == 'relattn':
            # Learned relation-level fusion instead of to_hetero's fixed aggr.
            from improve.relattn import RelAttnEncoder
            self.encoder = RelAttnEncoder(data.metadata(), hidden_channels,
                                          hidden_channels)
        else:
            enc = GNNEncoder(hidden_channels, hidden_channels,
                             norm=norm, residual=residual)
            self.encoder = to_hetero(enc, data.metadata(), aggr=aggr_str)
        self.decoder = EdgeDecoder(hidden_channels, rna_type, mode=decoder)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
