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
    """The author's encoder.  `norm` and `residual` are off by default.

    `layers=2` and `dropout=0` reproduce the released encoder exactly:
    conv1 -> LeakyReLU(0.1) -> conv2.  Depth and dropout are axes the author's
    search never had -- `find_best_params.py` sweeps only lr, epochs, channels
    and aggr, so two layers with no regularisation is an untested constant, not
    a tuned choice.

    Dropout is an `nn.Dropout` module rather than `F.dropout` on purpose:
    `to_hetero` traces this with torch.fx, and a functional call would bake
    `self.training` in as a constant at trace time, leaving dropout active
    during evaluation.
    """

    def __init__(self, hidden_channels, out_channels, norm=False,
                 residual=False, layers=2, dropout=0.0):
        super().__init__()
        assert layers >= 1
        dims = [hidden_channels] * (layers - 1) + [out_channels]
        self.convs = torch.nn.ModuleList(
            [GeneralConv((-1, -1), d) for d in dims])
        self.drops = torch.nn.ModuleList(
            [torch.nn.Dropout(dropout) for _ in range(len(dims) - 1)])
        self.norm = torch.nn.LayerNorm(hidden_channels) if norm else None
        self.residual = residual

    def forward(self, x, edge_index):
        h = x
        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            prev = h
            h = conv(h, edge_index)
            if i < last:
                if self.norm is not None and i == 0:
                    h = self.norm(h)
                h = F.leaky_relu(h, negative_slope=0.1)
                h = self.drops[i](h)
            elif self.residual:
                h = h + prev
        return h


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
                 decoder='concat', norm=False, residual=False, encoder='hetero',
                 layers=2, dropout=0.0):
        super().__init__()
        if encoder == 'hyper':
            from improve.hyper import HyperEncoder
            self.encoder = HyperEncoder(list(data.node_types), hidden_channels)
        elif encoder == 'hgt':
            # Heterogeneous Graph Transformer (Hu et al. 2020): per-relation
            # attention with type-specific K/Q/V projections.  This is the
            # architecture the paper's "semantic-enhanced heterogeneous graph"
            # framing implies; to_hetero(GeneralConv) only gives per-relation
            # weights with a fixed fusion.
            from improve.hgt import HGTEncoder
            self.encoder = HGTEncoder(data.metadata(), hidden_channels)
        elif encoder == 'relattn':
            # Learned relation-level fusion instead of to_hetero's fixed aggr.
            from improve.relattn import RelAttnEncoder
            self.encoder = RelAttnEncoder(data.metadata(), hidden_channels,
                                          hidden_channels)
        else:
            enc = GNNEncoder(hidden_channels, hidden_channels,
                             norm=norm, residual=residual, layers=layers,
                             dropout=dropout)
            self.encoder = to_hetero(enc, data.metadata(), aggr=aggr_str)
        self.decoder = EdgeDecoder(hidden_channels, rna_type, mode=decoder)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
