"""All-path resistance-curvature flow of learned entity affinities.

The numerical flow follows the pinned cqfei/resistance_curvature library's
distance-weighted normalization, curvature scaling, and leaf convention. It is
not a reproduction of the empty RCF application repository or its benchmarks.
Only positive edge coordinates are divided, keeping exact finite derivatives.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_models as graph_lib
from improve import nonlinear_pair_field_v1 as feature_graph
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'local_resistance', 'no_flow')


def parameters():
    return dict(width=32, heads=4, depth=2, dropout=.1, folds=5,
        neighbors=10, neighbor_tie_tolerance=1e-12, metric_width=16,
        affinity_bandwidth=1., flow_iterations=3, flow_step=.1,
        laplacian_ridge=.001, inverse_residual_tolerance=1e-10,
        flow_normalization='edge-distance-weighted curvature mean from pinned library',
        curvature_scale='minimum positive edge resistance divided by two',
        leaf_policy='flip negative curvature on edges touching an unweighted leaf',
        topology='fixed-attribute 10NN symmetrized union retaining all ties',
        updated_types=[1, 2], lnc_policy='identity_strong_reference')


def resistance_curvature(weight, support, p, local=False):
    """Differentiate the SPD inverse and divide only actual edge coordinates."""
    if weight.dtype != torch.float64 or weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError('Resistance flow requires a square FP64 affinity matrix')
    if support.shape != weight.shape or support.dtype != torch.bool:
        raise ValueError('Invalid fixed affinity support')
    if (torch.any(torch.diag(support)) or not torch.equal(support, support.T)
            or not torch.equal(weight, weight.T) or torch.any(weight[~support] != 0)
            or not torch.isfinite(weight).all() or torch.any(weight[support] <= 0)):
        raise ValueError('Expected a positive, symmetric, loop-free supported graph')
    row, col = torch.where(support)
    if len(row) == 0:
        raise ValueError('The registered entity graph must contain an edge')
    eye = torch.eye(len(weight), device=weight.device, dtype=weight.dtype)
    laplacian = torch.diag(weight.sum(1)) - weight
    system = laplacian + p['laplacian_ridge'] * eye
    factor = torch.linalg.cholesky(system)
    inverse = torch.cholesky_inverse(factor)
    diagonal = inverse.diagonal()
    true_resistance = diagonal[row] + diagonal[col] - inverse[row, col] - inverse[col, row]
    if not torch.isfinite(true_resistance).all() or torch.any(true_resistance <= 0):
        raise FloatingPointError('Nonpositive resistance on an active edge')
    # Compute the inverse in every arm. The local control discards alternative
    # paths, replacing resistance by the direct edge's reciprocal conductance.
    resistance = (1. / weight[row, col] + 0. * true_resistance) if local else true_resistance
    node = 1. - .5 * weight.new_zeros(len(weight)).index_add(0, row, weight[row, col] * resistance)
    curvature = (node[row] + node[col]) * resistance.min() / resistance
    degree = support.sum(1)
    leaf = (degree[row] == 1) | (degree[col] == 1)
    curvature = torch.where(leaf & (curvature < 0.), -curvature, curvature)
    with torch.no_grad():
        residual = torch.linalg.vector_norm(system @ inverse - eye)
        residual /= 1. + torch.linalg.vector_norm(system) * torch.linalg.vector_norm(inverse)
        error = float(residual)
        if not np.isfinite(error) or error > p['inverse_residual_tolerance']:
            raise FloatingPointError('Resistance inverse failed its residual check')
        audit = dict(inverse_relative_residual=error,
            minimum_true_resistance=float(true_resistance.min()),
            minimum_curvature=float(curvature.min()), maximum_curvature=float(curvature.max()),
            leaf_edges=int(leaf.sum()) // 2)
    return row, col, resistance, curvature, audit


def evolve(weight, support, p, kind):
    if kind not in KINDS or not 0 < p['flow_step'] < 1 or p['flow_iterations'] < 1:
        raise ValueError('Unregistered flow or invalid fixed step')
    initial, records = weight, []
    for iteration in range(p['flow_iterations']):
        row, col, resistance, curvature, audit = resistance_curvature(
            weight, support, p, local=(kind == 'local_resistance'))
        distance = 1. / weight[row, col]
        center = (distance * curvature).sum() / distance.sum()
        multiplier = 1. - p['flow_step'] * (curvature - center)
        if not torch.isfinite(multiplier).all() or torch.any(multiplier <= 0):
            raise FloatingPointError('Registered curvature-flow step lost positivity')
        updated = 1. / (distance * multiplier)
        weight = weight.new_zeros(weight.shape).index_put((row, col), updated)
        # Symmetric paired arcs are formed from exactly the same arithmetic.
        # Explicit averaging also prevents accumulated roundoff from changing
        # the SPD contract of subsequent iterations.
        weight = .5 * (weight + weight.T)
        audit.update(iteration=iteration, minimum_multiplier=float(multiplier.detach().min()),
            maximum_multiplier=float(multiplier.detach().max()),
            curvature_center=float(center.detach()), directed_edges=len(row))
        records.append(audit)
    if kind == 'no_flow':
        weight = initial + 0. * weight
    return weight, records


def normalize_affinity(weight):
    matrix = weight + torch.eye(len(weight), device=weight.device, dtype=weight.dtype)
    degree = matrix.sum(1)
    scale = degree.rsqrt()
    return scale[:, None] * matrix * scale[None, :]


@dataclass
class FlowContext:
    physical: tuple
    supports: tuple


def fixed_support(features, p, device):
    graph, audit = feature_graph.entity_neighbors(features, p['neighbors'], p['neighbor_tie_tolerance'])
    support = torch.as_tensor(graph.toarray() > 0, device=device, dtype=torch.bool)
    return support, audit


class ResistanceFlowGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown curvature-flow arm')
        self.p, self.kind = dict(p), kind
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([graph_lib.SparseTypedAttention(width, p['heads'],
            2 * relations + 2, p['dropout']) for _ in range(p['depth'])])
        self.metric = nn.ModuleList([nn.Linear(width, p['metric_width'], bias=False) for _ in (1, 2)])
        self.flow_project = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in (1, 2)])
        self.flow_norm = nn.ModuleList([nn.LayerNorm(width) for _ in (1, 2)])
        self.readout = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
                                    nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        self.flow_diagnostics = dict(calls=0, entity_flows=0, inverse_calls=0,
            flow_steps=0, maximum_inverse_relative_residual=0., minimum_multiplier=1.,
            minimum_true_resistance=float('inf'), maximum_absolute_curvature=0.)

    def encode(self, xs, context):
        edges, relation, counts = context.physical
        parts, start = [], 0
        for x, projection in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(projection(torch.cat((x / rms, counts[start:start + len(x)]), dim=-1)))
            start += len(x)
        value = F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers:
            value = layer(value, edges, relation)
        blocks = list(torch.split(value, [len(x) for x in xs]))
        self.flow_diagnostics['calls'] += 1
        for index, node_type in enumerate(self.p['updated_types']):
            state = blocks[node_type]
            metric = F.normalize(self.metric[index](state).double(), dim=1, eps=1e-8)
            differences = metric[:, None, :] - metric[None, :, :]
            weight = torch.exp(-differences.square().sum(-1) / self.p['affinity_bandwidth'])
            support = context.supports[index]
            weight = weight * support
            evolved, records = evolve(weight, support, self.p, self.kind)
            message = (normalize_affinity(evolved) @ state.double()).to(state.dtype)
            blocks[node_type] = self.flow_norm[index](state + self.flow_project[index](message))
            d = self.flow_diagnostics
            d['entity_flows'] += 1
            d['inverse_calls'] += len(records)
            d['flow_steps'] += len(records)
            for row in records:
                d['maximum_inverse_relative_residual'] = max(d['maximum_inverse_relative_residual'], row['inverse_relative_residual'])
                d['minimum_multiplier'] = min(d['minimum_multiplier'], row['minimum_multiplier'])
                d['minimum_true_resistance'] = min(d['minimum_true_resistance'], row['minimum_true_resistance'])
                d['maximum_absolute_curvature'] = max(d['maximum_absolute_curvature'],
                    abs(row['minimum_curvature']), abs(row['maximum_curvature']))
        return torch.cat(blocks)

    def forward(self, xs, graph, pairs):
        hidden = self.encode(xs, graph)
        left, right = hidden[pairs[0]], hidden[pairs[1]]
        features = torch.cat((left, right, left * right, (left - right).abs()), dim=-1)
        return self.readout(features).squeeze(-1)


def setup(a, p, kind, device):
    if set(a) - feature_graph.TRAIN_KEYS:
        raise ValueError('Resistance flow setup accepts training arrays only')
    if kind not in KINDS or p['updated_types'] != [1, 2]:
        raise ValueError('Unregistered resistance-flow deployment')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    supports = tuple(fixed_support(a['x_' + str(node_type)], p, device)[0] for node_type in p['updated_types'])
    graphs = []
    for fold in range(p['folds']):
        edges, kinds = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == fold))
        physical = graph_lib.graph_inputs(edges, kinds, np.empty((2, 0), dtype=np.int64), n, relations, device)
        graphs.append(FlowContext(physical, supports))
    xs = [torch.as_tensor(a['x_' + str(index)], device=device) for index in range(len(a['node_counts']))]
    model = ResistanceFlowGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
