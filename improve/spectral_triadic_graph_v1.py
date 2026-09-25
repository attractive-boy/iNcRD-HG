"""SHONN spectral sharing of quadratic coefficients in a typed graph pair head.

The layer follows Eq. (7) of arXiv:2603.28420v1. Its fixed random tensor
reduces trainable parameters, not tensor storage or quadratic multiply-adds.
Graph attention, query normalization, activations and residual prediction are
local adaptations; no source universal-approximation claim is made here.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'linear_only', 'direct_quadratic')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}


def parameters():
    return dict(width=64, heads=4, depth=2, dropout=.1, folds=5,
        pair_depth=2, pair_inputs='concatenated RNA and drug states',
        pair_normalization='LayerNorm after each spectral layer',
        pair_activation='GELU after each normalized spectral layer',
        quadratic_bias=False, quadratic_pairs='all i <= j, including squares',
        spectral_initialization='author Xavier uniform bases and uniform[-1,1] eigenvalues',
        spectral_linear_basis_trainable=True, spectral_quadratic_basis_trainable=False,
        lnc_policy='identity_strong_reference')


def empty_diagnostics():
    return dict(calls=0, layer_calls=0, query_rows=0, quadratic_terms=0,
        maximum_absolute_quadratic_weight=0., maximum_absolute_activation=0.)


class SpectralTriadic(nn.Module):
    """Linear spectral weights plus explicit upper-triangular input products.

All arms consume exactly the same initialization RNG. For direct_quadratic,
the fixed basis is replaced deterministically by the initial effective tensor;
that tensor becomes trainable and its spectral coefficients become frozen.
Thus full and direct_quadratic start at the same function and input Jacobian,
but have intentionally different trainable parameter counts and tangent spaces.
"""
    def __init__(self, in_features, out_features, kind='full', bias=True):
        super().__init__()
        if kind not in KINDS or min(in_features, out_features) < 1:
            raise ValueError('Invalid spectral-triadic layer')
        self.in_features, self.out_features = in_features, out_features
        self.kind, self.bias = kind, bias
        self.phi = nn.Parameter(torch.empty(out_features, in_features + int(bias)))
        self.l_in = nn.Parameter(torch.empty(in_features + int(bias)))
        self.l_out = nn.Parameter(torch.empty(out_features))
        pairs = in_features * (in_features + 1) // 2
        self.phi_triadic = nn.Parameter(torch.empty(out_features, pairs), requires_grad=False)
        self.l_in_triadic = nn.Parameter(torch.empty(pairs))
        self.l_out_triadic = nn.Parameter(torch.empty(out_features))
        indices = torch.triu_indices(in_features, in_features)
        self.register_buffer('triu_rows', indices[0])
        self.register_buffer('triu_cols', indices[1])
        # The draw order is the pinned author's SpectralTriadic initialization.
        nn.init.xavier_uniform_(self.phi)
        nn.init.uniform_(self.l_in, -1., 1.)
        nn.init.uniform_(self.l_out, -1., 1.)
        nn.init.xavier_uniform_(self.phi_triadic)
        nn.init.uniform_(self.l_in_triadic, -1., 1.)
        nn.init.uniform_(self.l_out_triadic, -1., 1.)
        if kind == 'direct_quadratic':
            with torch.no_grad():
                effective = self.spectral_quadratic_weight().clone()
                self.phi_triadic.copy_(effective)
            self.phi_triadic.requires_grad_(True)
            self.l_in_triadic.requires_grad_(False)
            self.l_out_triadic.requires_grad_(False)

    def linear_weight(self):
        return (self.l_in[None, :] - self.l_out[:, None]) * self.phi

    def spectral_quadratic_weight(self):
        return (self.l_in_triadic[None, :] - self.l_out_triadic[:, None]) * self.phi_triadic

    def quadratic_weight(self):
        return self.phi_triadic if self.kind == 'direct_quadratic' else self.spectral_quadratic_weight()

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError('Expected a matrix with the declared input width')
        linear_inputs = (torch.cat((inputs, inputs.new_ones((len(inputs), 1))), dim=1)
                         if self.bias else inputs)
        pairs = inputs[:, self.triu_rows] * inputs[:, self.triu_cols]
        quadratic = F.linear(pairs, self.quadratic_weight())
        # Keep the same actual pair construction and matrix multiplication in
        # the linear control. Zero gradient does not imply equal active capacity.
        if self.kind == 'linear_only':
            quadratic = 0. * quadratic
        return F.linear(linear_inputs, self.linear_weight()) + quadratic


class SpectralTriadicGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS or p['pair_depth'] != 2:
            raise ValueError('Unregistered spectral-triadic architecture')
        self.p, self.kind = dict(p), kind
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([graph_lib.SparseTypedAttention(width, p['heads'],
            2 * relations + 2, p['dropout']) for _ in range(p['depth'])])
        self.pair_layers = nn.ModuleList([SpectralTriadic(2 * width, width, kind),
                                         SpectralTriadic(width, width, kind)])
        self.pair_norms = nn.ModuleList([nn.LayerNorm(width) for _ in self.pair_layers])
        self.dropout = nn.Dropout(p['dropout'])
        self.readout = nn.Linear(width, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        self.triadic_diagnostics = empty_diagnostics()

    def encode(self, xs, graph):
        edges, relation, counts = graph
        parts, start = [], 0
        for x, projection in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(projection(torch.cat((x / rms, counts[start:start + len(x)]), dim=1)))
            start += len(x)
        state = F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers:
            state = layer(state, edges, relation)
        return state

    def forward(self, xs, graph, pairs):
        hidden = self.encode(xs, graph)
        state = torch.cat((hidden[pairs[0]], hidden[pairs[1]]), dim=1)
        d = self.triadic_diagnostics
        d['calls'] += 1
        d['query_rows'] += pairs.shape[1]
        for layer, norm in zip(self.pair_layers, self.pair_norms):
            state = self.dropout(F.gelu(norm(layer(state))))
            with torch.no_grad():
                d['layer_calls'] += 1
                d['quadratic_terms'] += len(state) * layer.triu_rows.numel()
                d['maximum_absolute_quadratic_weight'] = max(
                    d['maximum_absolute_quadratic_weight'], float(layer.quadratic_weight().abs().max()))
                if state.numel():
                    d['maximum_absolute_activation'] = max(
                        d['maximum_absolute_activation'], float(state.abs().max()))
        return self.readout(state).squeeze(-1)


def setup(a, p, kind, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Spectral-triadic setup accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered spectral-triadic arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    graphs = []
    for f in range(p['folds']):
        edges, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        graphs.append(graph_lib.graph_inputs(edges, labels, np.empty((2, 0), dtype=np.int64),
                                            n, relations, device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = SpectralTriadicGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
