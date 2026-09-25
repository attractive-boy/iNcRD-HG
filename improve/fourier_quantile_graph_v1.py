"""Typed graph message aggregation using Fourier quantile embeddings.

The uniform-mass FSW integral follows arXiv:2405.16519v3, Eq. 24, and
FSW-GNN arXiv:2410.09118v2. Ragged neighborhoods are bucketed only for
storage, with no truncation or padding mass. This finite-width link model
does not claim the source's dimension-dependent bi-Lipschitz guarantees.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'mean_only', 'mean_std')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}


def parameters():
    return dict(width=64, depth=2, dropout=.1, folds=5, edge_width=8,
        embedding_width=128, frequency_initialization='spread quantiles u/(1-u), u=(i+.5)/m',
        positive_frequency='softplus', direction_normalization='unit L2 each forward',
        moment_variance_ridge=1e-8, neighborhood_cap=None,
        aggregation_dtype='float64', degree_encoding='log1p',
        self_message='concatenated own state; diagonal arcs omitted from neighbor multiset',
        bucket_storage='next power of two, padding has zero mass',
        lnc_policy='identity_strong_reference')


def stable_sinc(value):
    """Normalized sinc with its accurate near-zero value and derivative."""
    small = value.abs() < 1e-3
    z = torch.pi * torch.where(small, value, torch.zeros_like(value))
    squared = z.square()
    series = 1. - squared / 6. + squared.square() / 120. - squared.pow(3) / 5040.
    regular = torch.sinc(torch.where(small, torch.ones_like(value), value))
    return torch.where(small, series, regular)


def uniform_quantile_features(projected, valid, frequencies, kind, variance_ridge=1e-8):
    """Exact piecewise-constant quantile integral on uniform valid elements.

    projected: batch x padded-neighbors x coordinates; valid: batch x neighbors.
    Empty sets map to zero. Sort is differentiated away from order crossings;
    no derivative is claimed for the permutation itself at ties.
    """
    if (projected.ndim != 3 or projected.dtype != torch.float64
            or valid.shape != projected.shape[:2] or valid.dtype != torch.bool
            or frequencies.shape != (projected.shape[-1],) or kind not in KINDS
            or projected.shape[-1] % 2 or variance_ridge <= 0):
        raise ValueError('Invalid Fourier quantile inputs')
    if (not torch.isfinite(projected[valid]).all() or not torch.isfinite(frequencies).all()
            or torch.any(frequencies < 0)):
        raise FloatingPointError('Nonfinite features or invalid frequencies')
    count = valid.sum(1)
    divisor = count.clamp_min(1).to(projected.dtype)[:, None, None]
    finite = torch.where(valid[..., None], projected, torch.zeros_like(projected))
    values = torch.sort(torch.where(valid[..., None], finite,
        torch.full_like(projected, torch.inf)), dim=1, stable=True).values
    rank = torch.arange(projected.shape[1], dtype=projected.dtype, device=projected.device)[None, :, None]
    ranked_valid = rank < count[:, None, None]
    values = torch.where(ranked_valid, values, torch.zeros_like(values))
    xi = frequencies[None, None, :]
    # Integral over [r/n,(r+1)/n] via sum-to-product, avoiding cancellation
    # of two close sine values and the removable division by zero at xi=0.
    coefficient = (2. * (1. + xi) / divisor * stable_sinc(xi / divisor)
        * torch.cos(torch.pi * xi * (2. * rank + 1.) / divisor))
    fsw = (values * coefficient).sum(1)
    mean = finite.sum(1) / divisor[:, 0]
    centered = torch.where(valid[..., None], finite - mean[:, None], torch.zeros_like(finite))
    variance = centered.square().sum(1) / divisor[:, 0]
    deviation = (variance + variance_ridge).sqrt() - variance_ridge**.5
    middle = projected.shape[-1] // 2
    mean_only = 2. * mean
    mean_std = 2. * torch.cat((mean[:, :middle], deviation[:, middle:]), dim=1)
    # Every arm computes identical sorts, frequencies, means and variances.
    # Unused paths have zero gradient, while remaining connected for auditing.
    if kind == 'full':
        result = fsw + 0. * (mean_only + mean_std)
    elif kind == 'mean_only':
        result = mean_only + 0. * (fsw + mean_std)
    else:
        result = mean_std + 0. * (fsw + mean_only)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Fourier quantile aggregation lost finiteness')
    return result


@dataclass
class QuantileBucket:
    recipients: torch.Tensor
    messages: torch.Tensor
    valid: torch.Tensor


@dataclass
class QuantileContext:
    physical: tuple
    senders: torch.Tensor
    relation: torch.Tensor
    degree: torch.Tensor
    buckets: tuple


def pack_graph(physical):
    edges, relation, counts = physical
    keep = edges[0] != edges[1]
    senders, recipients = edges[:, keep]
    kinds = relation[keep]
    destination = recipients.detach().cpu().numpy()
    nodes = len(counts)
    degree = np.bincount(destination, minlength=nodes)
    order = np.argsort(destination, kind='stable')
    starts = np.r_[0, np.cumsum(degree)]
    capacities = np.where(degree > 0, 2**np.ceil(np.log2(np.maximum(degree, 1))).astype(np.int64), 0)
    packs = []
    for capacity in sorted(set(capacities) - {0}):
        ids = np.flatnonzero(capacities == capacity)
        messages = np.zeros((len(ids), capacity), dtype=np.int64)
        valid = np.zeros(messages.shape, dtype=bool)
        for row, node in enumerate(ids):
            messages[row, :degree[node]] = order[starts[node]:starts[node + 1]]
            valid[row, :degree[node]] = True
        packs.append(QuantileBucket(torch.as_tensor(ids, device=edges.device),
            torch.as_tensor(messages, device=edges.device), torch.as_tensor(valid, device=edges.device)))
    return QuantileContext(physical, senders, kinds,
        torch.as_tensor(degree, device=edges.device), tuple(packs))


class FourierQuantileLayer(nn.Module):
    def __init__(self, width, relations, p):
        super().__init__()
        size = p['embedding_width']
        if size < 2 or size % 2:
            raise ValueError('The registered embedding needs an even positive width')
        self.p = p
        self.relation = nn.Embedding(relations, p['edge_width'])
        self.directions = nn.Parameter(torch.randn(size, width + p['edge_width']))
        uniform = (torch.arange(size, dtype=torch.float64) + .5) / size
        frequency = uniform / (1. - uniform)
        self.raw_frequency = nn.Parameter((frequency + torch.log(-torch.expm1(-frequency))).float())
        self.combine = nn.Sequential(nn.Linear(width + size + 1, width),
            nn.LeakyReLU(.2), nn.Dropout(p['dropout']))
        self.norm = nn.LayerNorm(width)

    def forward(self, state, context, kind):
        features = torch.cat((state[context.senders], self.relation(context.relation)), dim=1)
        directions = F.normalize(self.directions.double(), dim=1, eps=1e-12)
        frequency = F.softplus(self.raw_frequency.double())
        projected = F.linear(features.double(), directions)
        message = projected.new_zeros((len(state), self.p['embedding_width'])) + 0. * projected.sum()
        for bucket in context.buckets:
            encoded = uniform_quantile_features(projected[bucket.messages], bucket.valid,
                frequency, kind, self.p['moment_variance_ridge'])
            message = message.index_copy(0, bucket.recipients, encoded)
        with torch.no_grad():
            diagnostic = dict(layer_calls=1, sort_calls=len(context.buckets),
                nonempty_neighborhoods=int((context.degree > 0).sum()),
                empty_neighborhoods=int((context.degree == 0).sum()),
                messages=len(context.senders), padded_slots=sum(v.valid.numel() for v in context.buckets),
                maximum_degree=int(context.degree.max()),
                maximum_absolute_embedding=float(message.abs().max()),
                minimum_frequency=float(frequency.min()), maximum_frequency=float(frequency.max()))
        degree = context.degree.double().log1p()[:, None]
        combined = torch.cat((state, (message / self.p['embedding_width']**.5).to(state.dtype),
                              degree.to(state.dtype)), dim=1)
        return self.norm(state + self.combine(combined)), diagnostic


class FourierQuantileGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unregistered Fourier quantile arm')
        self.p, self.kind = dict(p), kind
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([FourierQuantileLayer(width, 2 * relations + 2, p)
                                     for _ in range(p['depth'])])
        self.readout = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        self.quantile_diagnostics = dict(calls=0, layer_calls=0, sort_calls=0,
            nonempty_neighborhoods=0, empty_neighborhoods=0, messages=0, padded_slots=0,
            maximum_degree=0, maximum_absolute_embedding=0.,
            minimum_frequency=float('inf'), maximum_frequency=0.)

    def encode(self, xs, context):
        counts = context.physical[2]
        parts, start = [], 0
        for x, projection in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(projection(torch.cat((x / rms, counts[start:start + len(x)]), dim=1)))
            start += len(x)
        state = F.gelu(self.norm(torch.cat(parts)))
        self.quantile_diagnostics['calls'] += 1
        for layer in self.layers:
            state, audit = layer(state, context, self.kind)
            for key in ('layer_calls', 'sort_calls', 'nonempty_neighborhoods', 'empty_neighborhoods',
                        'messages', 'padded_slots'):
                self.quantile_diagnostics[key] += audit[key]
            for key in ('maximum_degree', 'maximum_absolute_embedding', 'maximum_frequency'):
                self.quantile_diagnostics[key] = max(self.quantile_diagnostics[key], audit[key])
            self.quantile_diagnostics['minimum_frequency'] = min(
                self.quantile_diagnostics['minimum_frequency'], audit['minimum_frequency'])
        return state

    def forward(self, xs, context, pairs):
        hidden = self.encode(xs, context)
        left, right = hidden[pairs[0]], hidden[pairs[1]]
        return self.readout(torch.cat((left, right, left * right, (left - right).abs()), dim=1)).squeeze(-1)


def setup(a, p, kind, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Fourier quantile setup accepts training arrays only')
    if kind not in KINDS or p['neighborhood_cap'] is not None:
        raise ValueError('Unregistered Fourier quantile deployment')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    graphs = []
    for fold in range(p['folds']):
        edges, kinds = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == fold))
        physical = graph_lib.graph_inputs(edges, kinds, np.empty((2, 0), dtype=np.int64), n, relations, device)
        graphs.append(pack_graph(physical))
    xs = [torch.as_tensor(a['x_' + str(index)], device=device) for index in range(len(a['node_counts']))]
    model = FourierQuantileGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
