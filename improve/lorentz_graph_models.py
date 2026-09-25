"""Typed geometric graph attention inspired by Hypformer (KDD 2024).

This is a sparse, fixed-curvature bipartite adaptation, not Hypformer's full
linear-attention model. Lorentz attention, centroids, residual aggregation and
distance readout are tested against matched Euclidean operations. The control
also sees the derived radial coordinate, so a radial feature alone cannot
establish a geometric advantage.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from improve.overlap_block_models import edge_softmax


def lift(space, radius_sq=1.):
    time = (space.square().sum(-1, keepdim=True) + radius_sq).sqrt()
    return torch.cat((time, space), dim=-1)


def lorentz_inner(x, y):
    return (x[..., 1:] * y[..., 1:]).sum(-1) - x[..., 0] * y[..., 0]


def chordal_sq(x, y, radius_sq=1.):
    """Lorentz squared chordal distance, monotone in geodesic distance."""
    return (-2*radius_sq - 2*lorentz_inner(x, y)).clamp_min(0.)


def normalize_midpoint(weighted_sum, radius_sq=1.):
    norm_sq = -lorentz_inner(weighted_sum, weighted_sum)
    return math.sqrt(radius_sq) * weighted_sum / norm_sq.clamp_min(1e-10).sqrt()[..., None]


def aggregate(values, weights, destinations, n, curved, radius_sq):
    """Weighted Lorentz centroid vs matched spatial arithmetic mean."""
    result = values.new_zeros((n, *values.shape[1:]))
    result.index_add_(0, destinations, weights[..., None]*values)
    if curved:
        return normalize_midpoint(result, radius_sq)
    return lift(result[..., 1:], radius_sq)


class GeometricLayer(nn.Module):
    def __init__(self, width, heads, edge_types, dropout, curved_attention=True,
                 radius_sq=1.):
        super().__init__()
        self.heads, self.dim = heads, width//heads
        self.curved_attention, self.radius_sq = curved_attention, radius_sq
        self.qkv = nn.Linear(width+1, 3*width)
        self.edge_bias = nn.Embedding(edge_types, heads)
        nn.init.zeros_(self.edge_bias.weight)
        self.output = nn.Linear(width+heads, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width+1, 2*width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2*width, width))
        self.dropout = nn.Dropout(dropout)
        self.attention_scale = nn.Parameter(torch.tensor(0.))

    def merge(self, x, y):
        both = .5*x + .5*y
        return normalize_midpoint(both, self.radius_sq) if self.curved_attention else lift(
            both[..., 1:], self.radius_sq)

    def forward(self, h, edges, relations):
        # HRC-style spatial normalization and time-coordinate reconstruction.
        normalized = lift(self.norm1(h[..., 1:]), self.radius_sq)
        q, k, v = self.qkv(normalized).reshape(len(h), 3, self.heads, self.dim).unbind(1)
        q, k, v = (lift(t, self.radius_sq) for t in (q, k, v))
        src, dst = edges
        if self.curved_attention:
            distance = chordal_sq(q[dst], k[src], self.radius_sq)
        else:
            distance = (q[dst, ..., 1:]-k[src, ..., 1:]).square().sum(-1)
        scale = F.softplus(self.attention_scale)+.1
        weight = edge_softmax(-distance/(math.sqrt(self.dim)*scale)+self.edge_bias(relations),
                              dst, len(h))
        message = aggregate(v[src], weight, dst, len(h), self.curved_attention, self.radius_sq)
        message = lift(self.dropout(self.output(message.flatten(1))), self.radius_sq)
        h = self.merge(h, message)
        refined = lift(self.norm2(h[..., 1:]), self.radius_sq)
        y = lift(self.dropout(self.ff(refined)), self.radius_sq)
        return self.merge(h, y)


class LorentzGraph(nn.Module):
    def __init__(self, dims, node_counts, relations, width=64, depth=3, heads=4,
                 dropout=.1, kind='lorentz', radius_sq=1.):
        super().__init__()
        if kind not in ('lorentz', 'euclidean', 'attention_only', 'distance_only'):
            raise ValueError('Unknown geometry control')
        self.kind, self.radius_sq = kind, radius_sq
        self.node_counts = node_counts
        self.curved_readout = kind in ('lorentz', 'distance_only')
        curved_attention = kind in ('lorentz', 'attention_only')
        self.project = nn.ModuleList([nn.Linear(d+relations, width) for d in dims])
        self.input_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([GeometricLayer(width, heads, 2*relations+2, dropout,
                curved_attention, radius_sq) for _ in range(depth)])
        self.rna_projection = nn.Linear(width+1, width//2)
        self.drug_projection = nn.Linear(width+1, width//2)
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.scale = nn.Parameter(torch.tensor(0.))
        self.bias = nn.Parameter(torch.tensor(0.))

    def encode(self, xs, graph):
        edges, relations, counts = graph
        projected, offset = [], 0
        for x, project, n in zip(xs, self.project, self.node_counts):
            norm = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            projected.append(project(torch.cat((x/norm, counts[offset:offset+n]), 1)))
            offset += n
        h = lift(F.gelu(self.input_norm(torch.cat(projected))), self.radius_sq)
        for layer in self.layers:
            h = layer(h, edges, relations)
        return h, counts

    def decode(self, h, counts, pairs):
        r, d = pairs
        hr = lift(self.rna_projection(h[r]), self.radius_sq)
        hd = lift(self.drug_projection(h[d]), self.radius_sq)
        distance = chordal_sq(hr, hd, self.radius_sq) if self.curved_readout else (
            hr[..., 1:]-hd[..., 1:]).square().sum(-1)
        degree = self.rna_degree(counts[r]).squeeze(-1) + self.drug_degree(counts[d]).squeeze(-1)
        return self.bias + degree - F.softplus(self.scale)*distance

    def forward(self, xs, graph, pairs):
        return self.decode(*self.encode(xs, graph), pairs)


@torch.no_grad()
def predict(model, xs, graphs, pairs):
    model.eval()
    probabilities = [model(xs, graph, pairs).sigmoid() for graph in graphs]
    probability = torch.stack(probabilities).mean(0).clamp(1e-7, 1-1e-7)
    return torch.logit(probability)
