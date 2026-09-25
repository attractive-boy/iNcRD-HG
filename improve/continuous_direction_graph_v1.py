"""Independent continuous edge directions, adapted from released CoED code.

The released operator uses squared cosine/sine masses, whereas the inspected
2024 preprint defines unsquared weights. We follow and independently test the
released operator, without claiming the paper's exact fuzzy-Laplacian theorem.
All input relations start neutral; storage order is never a biological arrow.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve.sharpness_graph_v1 import IdentityResidual, identity_task


KINDS = ('full', 'symmetric_weights', 'fixed_direction')


@dataclass
class DirectionContext:
    edges: torch.Tensor
    edge_ids: torch.Tensor
    relation: torch.Tensor
    counts: torch.Tensor
    node_count: int


def context(a, omitted, device):
    omitted = np.asarray(omitted, dtype=np.int64)
    pairs = a['pairs_train'][:, omitted]
    codes = pairs[0] * int(a['nd']) + pairs[1]
    live = np.flatnonzero(~np.isin(a['target_codes'], codes))
    edges, kinds = a['edges'][:, live], a['relation'][live]
    if np.isin(a['target_codes'][live], codes).any():
        raise ValueError('A query fact remains visible')
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    if np.any(edges[0] == edges[1]):
        raise ValueError('Continuous directions require physical edges without self-loops')
    counts = np.zeros((n, nr), dtype=np.float32)
    np.add.at(counts, (edges[0], kinds), 1.)
    np.add.at(counts, (edges[1], kinds), 1.)
    return DirectionContext(torch.as_tensor(edges, device=device),
        torch.as_tensor(live, device=device), torch.as_tensor(kinds, device=device),
        torch.as_tensor(np.log1p(counts), device=device), n)


def normalized_directions(theta, edges, n, kind):
    """Author squared-mass operator with explicit source-to-target convention.

    Every listed physical edge contributes both arcs. At a neutral theta=pi/4
    the two matrices coincide. The complete matrices are mutual transposes,
    with in/out degree normalization coupled across their two endpoints.
    """
    if kind not in KINDS or theta.ndim != 1 or theta.shape[0] != edges.shape[1]:
        raise ValueError('Invalid direction vector or arm')
    if not torch.isfinite(theta).all() or bool(((theta < -1e-6) | (theta > np.pi / 2 + 1e-6)).any()):
        raise ValueError('Unprojected direction parameter')
    if kind == 'fixed_direction':
        theta = torch.full_like(theta, np.pi / 4)
    forward = theta.cos().square()
    reverse = theta.sin().square() if kind == 'full' else forward
    source, target = torch.cat((edges, edges.flip(0)), dim=1)
    incoming = torch.cat((forward, reverse))
    outgoing = torch.cat((reverse, forward))
    # Keep the released epsilon and explicit zero-degree boundary.
    send_degree = theta.new_full((n,), 1e-12).index_add(0, source, incoming)
    receive_degree = theta.new_full((n,), 1e-12).index_add(0, source, outgoing)
    send_scale = torch.where(send_degree < 1e-11, 0., send_degree.rsqrt())
    receive_scale = torch.where(receive_degree < 1e-11, 0., receive_degree.rsqrt())
    in_weight = send_scale[source] * incoming * receive_scale[target]
    out_weight = receive_scale[source] * outgoing * send_scale[target]
    return source, target, in_weight, out_weight


def aggregate(value, source, target, weight):
    return torch.zeros_like(value).index_add(0, target, value[source] * weight[:, None])


class DirectionLayer(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.incoming = nn.Linear(width, width, bias=False)
        self.outgoing = nn.Linear(width, width, bias=False)
        self.self_feature = nn.Linear(width, width, bias=True)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, propagation):
        source, target, in_weight, out_weight = propagation
        incoming = aggregate(value, source, target, in_weight)
        outgoing = aggregate(value, source, target, out_weight)
        combined = self.self_feature(value) + .5 * (self.incoming(incoming) + self.outgoing(outgoing))
        return self.dropout(F.relu(self.norm(combined)))


class ContinuousDirectionGraph(nn.Module):
    def __init__(self, dimensions, relations, edges, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unregistered direction arm')
        self.p, self.kind = dict(p), kind
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.input_norm = nn.LayerNorm(width)
        self.theta = nn.Parameter(torch.full((edges,), np.pi / 4))
        self.layers = nn.ModuleList([DirectionLayer(width, p['dropout']) for _ in range(p['depth'])])
        self.readout = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
                                    nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def encode(self, xs, graph):
        parts, start = [], 0
        for x, projection in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(projection(torch.cat((x / rms, graph.counts[start:start + len(x)]), dim=-1)))
            start += len(x)
        value = F.gelu(self.input_norm(torch.cat(parts)))
        propagation = normalized_directions(self.theta[graph.edge_ids], graph.edges,
                                           graph.node_count, self.kind)
        for layer in self.layers:
            value = layer(value, propagation)
        return value

    def forward(self, xs, graph, pairs):
        h = self.encode(xs, graph)
        left, right = h[pairs[0]], h[pairs[1]]
        return self.readout(torch.cat((left, right, left * right, (left - right).abs()), dim=-1)).squeeze(-1)

    @torch.no_grad()
    def project_directions(self):
        self.theta.clamp_(0., np.pi / 2)


def parameters():
    return dict(width=64, depth=3, dropout=.1, folds=5, direction_learning_rate=.01,
        direction_weight_decay=0., direction_initialization='neutral_pi_over_4',
        direction_parameterization='projected_angle_shared_across_layers',
        propagation='author_squared_cosine_sine_with_coupled_in_out_degree_normalization',
        lnc_policy='identity_strong_reference')


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered direction arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None] * p['folds'], folds
    graphs = [context(a, np.flatnonzero(a['folds'] == f), device) for f in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = ContinuousDirectionGraph([x.shape[1] for x in xs], int(a['relation'].max()) + 1,
        a['edges'].shape[1], p, kind).to(device)
    return model, xs, graphs, folds


def direction_summary(model):
    if not hasattr(model, 'theta'):
        return dict(stored_direction_parameters=0, active_direction_parameters=0,
                    mean_abs_angle_change=0., min_angle=0., max_angle=0., boundary_fraction=0.)
    theta = model.theta.detach()
    return dict(stored_direction_parameters=theta.numel(),
        active_direction_parameters=theta.numel() if model.kind != 'fixed_direction' else 0,
        mean_abs_angle_change=float((theta - np.pi / 4).abs().mean()),
        min_angle=float(theta.min()), max_angle=float(theta.max()),
        boundary_fraction=float(((theta < 1e-6) | (theta > np.pi / 2 - 1e-6)).float().mean()))
