"""Convex reference-offset pair fields with nonlinear graph p-energy.

The nonlinear p-law and damped continuation are adapted from arXiv:2607.03503v1.
The local fidelity is Bernoulli logistic with a fixed nested OOF offset and a
small all-node ridge. This is not the author's near-linear solver or original
squared-loss SSL experiment. No target edge is used to construct the graph.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested

KINDS = ('full', 'quadratic')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}


def parameters():
    return dict(neighbors=10, neighbor_tie_tolerance=1e-12,
        graph_bandwidth='median positive squared distance of retained directed entity neighbors',
        graph_axis_normalization='each entity axis has mean weighted degree one',
        regularization=1., ridge=.001, primary_exponent=3., control_exponent=2.,
        continuation=[2., 3.], conductance_floor=1e-6, gradient_tolerance=1e-8,
        maximum_newton_steps=60, cg_relative_tolerance=1e-9, cg_absolute_tolerance=1e-12,
        maximum_cg_steps=1500, armijo=.0001, maximum_backtracks=40,
        residual_clip=2., strong_weight=.1, dtype='float64', lnc_policy='identity_strong_reference')


def entity_neighbors(features, k=10, tie_tolerance=1e-12):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() or k < 1 or tie_tolerance < 0:
        raise ValueError('Invalid fixed entity features or neighbor budget')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
    norm2 = np.square(x).sum(1)
    distance = np.maximum(norm2[:, None] + norm2[None, :] - 2. * (x @ x.T), 0.)
    np.fill_diagonal(distance, np.inf)
    k = min(int(k), len(x) - 1)
    cutoff = np.partition(distance, k - 1, axis=1)[:, k - 1]
    selected = distance <= cutoff[:, None] + tie_tolerance
    np.fill_diagonal(selected, False)
    row, col = np.nonzero(selected)
    positive = distance[row, col][distance[row, col] > tie_tolerance]
    bandwidth = float(np.median(positive)) if len(positive) else 1.
    weight = np.exp(-distance[row, col] / bandwidth)
    graph = sparse.csr_matrix((weight, (row, col)), shape=distance.shape)
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()
    average_degree = float(graph.sum()) / len(x)
    if not np.isfinite(average_degree) or average_degree <= 0:
        raise ValueError('The entity graph has no positive similarity weight')
    graph /= average_degree
    graph.sort_indices()
    return graph, dict(nodes=len(x), neighbor_budget=k, directed_retained=int(len(row)),
        undirected_edges=graph.nnz // 2, ties_kept=int(len(row) - k * len(x)),
        bandwidth=bandwidth, average_weighted_degree=float(graph.sum()) / len(x),
        maximum_degree=int(np.diff(graph.indptr).max()))


def cartesian_graph(rna_features, drug_features, p=None):
    p = parameters() if p is None else p
    rna, rna_audit = entity_neighbors(rna_features, p['neighbors'], p['neighbor_tie_tolerance'])
    drug, drug_audit = entity_neighbors(drug_features, p['neighbors'], p['neighbor_tie_tolerance'])
    graph = sparse.kron(rna, sparse.eye(len(drug_features)), format='csr') + sparse.kron(
        sparse.eye(len(rna_features)), drug, format='csr')
    graph.sum_duplicates()
    graph.sort_indices()
    return graph, dict(rna=rna_audit, drug=drug_audit, nodes=graph.shape[0],
        undirected_edges=graph.nnz // 2, maximum_degree=int(np.diff(graph.indptr).max()),
        graph_uses_only_fixed_entity_features=True, target_edges_used=False, labels_used=False)


@dataclass
class PairGraph:
    nodes: int
    u: torch.Tensor
    v: torch.Tensor
    weight: torch.Tensor
    crow: torch.Tensor
    columns: torch.Tensor
    arc_edge: torch.Tensor

    @classmethod
    def from_scipy(cls, graph, device):
        graph = sparse.csr_matrix(graph, dtype=np.float64)
        graph.sum_duplicates()
        graph.eliminate_zeros()
        graph.sort_indices()
        n = graph.shape[0]
        if (graph.shape != (n, n) or np.any(graph.diagonal()) or not np.isfinite(graph.data).all()
                or np.any(graph.data <= 0) or (graph - graph.T).nnz):
            raise ValueError('Expected a symmetric positive graph without self loops')
        upper = sparse.triu(graph, k=1, format='coo')
        order = np.lexsort((upper.col, upper.row))
        row, col, values = upper.row[order], upper.col[order], upper.data[order]
        arcs = graph.tocoo()
        code = row.astype(np.int64) * n + col
        arc_codes = np.minimum(arcs.row, arcs.col).astype(np.int64) * n + np.maximum(arcs.row, arcs.col)
        arc_edge = np.searchsorted(code, arc_codes)
        if len(code):
            np.testing.assert_array_equal(code[arc_edge], arc_codes)
        return cls(n, torch.as_tensor(row, device=device, dtype=torch.long),
            torch.as_tensor(col, device=device, dtype=torch.long),
            torch.as_tensor(values, device=device, dtype=torch.float64),
            torch.as_tensor(graph.indptr, device=device, dtype=torch.long),
            torch.as_tensor(graph.indices, device=device, dtype=torch.long),
            torch.as_tensor(arc_edge, device=device, dtype=torch.long))

    def divergence(self, flow):
        return flow.new_zeros(self.nodes).index_add(0, self.u, flow).index_add(0, self.v, -flow)

    def weighted_operator(self, conductance, mass):
        degree = conductance.new_zeros(self.nodes).index_add(0, self.u, conductance).index_add(0, self.v, conductance)
        adjacency = torch.sparse_csr_tensor(self.crow, self.columns, conductance[self.arc_edge],
            size=(self.nodes, self.nodes), dtype=conductance.dtype, device=conductance.device)
        diagonal = degree + mass
        def multiply(value):
            return diagonal * value - torch.sparse.mm(adjacency, value[:, None]).flatten()
        return multiply, diagonal


def value_and_gradient(field, graph, train, y, offset, exponent, regularization, ridge):
    difference = field[graph.u] - field[graph.v]
    logits = offset + field[train]
    value = F.binary_cross_entropy_with_logits(logits, y, reduction='sum')
    value = value + regularization / exponent * (graph.weight * difference.abs().pow(exponent)).sum()
    value = value + .5 * ridge * field.square().sum()
    flow = regularization * graph.weight * difference.abs().pow(exponent - 2.) * difference
    gradient = graph.divergence(flow) + ridge * field
    gradient = gradient.index_add(0, train, torch.sigmoid(logits) - y)
    return value, gradient


@torch.no_grad()
def pcg(multiply, diagonal, b, relative_tolerance, absolute_tolerance, maximum_steps):
    if not torch.all(diagonal > 0) or not torch.isfinite(diagonal).all():
        raise ValueError('Nonpositive Jacobi preconditioner')
    x = torch.zeros_like(b)
    r = b.clone()
    target = max(absolute_tolerance, relative_tolerance * float(torch.linalg.vector_norm(b)))
    initial = float(torch.linalg.vector_norm(r))
    if initial <= target:
        return x, dict(steps=0, residual_norm=initial, tolerance=target)
    z = r / diagonal
    direction, product = z.clone(), r @ z
    for step in range(1, maximum_steps + 1):
        image = multiply(direction)
        curvature = direction @ image
        if not torch.isfinite(curvature) or curvature <= 0:
            raise FloatingPointError('CG lost positive curvature')
        alpha = product / curvature
        x.add_(direction, alpha=float(alpha))
        r.sub_(image, alpha=float(alpha))
        norm = float(torch.linalg.vector_norm(r))
        if norm <= target:
            # Recompute a true residual before declaring convergence.
            true = b - multiply(x)
            true_norm = float(torch.linalg.vector_norm(true))
            if true_norm <= max(4. * target, absolute_tolerance):
                return x, dict(steps=step, residual_norm=true_norm, tolerance=target)
            r = true
            z = r / diagonal
            direction, product = z.clone(), r @ z
            continue
        z = r / diagonal
        updated = r @ z
        direction = z + (updated / product) * direction
        product = updated
    raise ValueError('CG did not reach the registered residual tolerance')


@torch.no_grad()
def solve(graph, train, y, offset, exponent, p=None, initial=None):
    p = parameters() if p is None else p
    device = graph.weight.device
    train = torch.as_tensor(train, device=device, dtype=torch.long)
    y = torch.as_tensor(y, device=device, dtype=torch.float64)
    offset = torch.as_tensor(offset, device=device, dtype=torch.float64)
    if (exponent not in (2., 3.) or y.ndim != 1 or y.shape != offset.shape or train.shape != y.shape
            or not len(train) or len(torch.unique(train)) != len(train) or int(train.min()) < 0
            or int(train.max()) >= graph.nodes or not torch.all((y == 0.) | (y == 1.))
            or not torch.isfinite(offset).all() or p['ridge'] <= 0 or p['regularization'] <= 0):
        raise ValueError('Invalid nonlinear-field training problem')
    field = torch.zeros(graph.nodes, device=device, dtype=torch.float64) if initial is None else initial.detach().clone()
    if field.shape != (graph.nodes,) or not torch.isfinite(field).all():
        raise ValueError('Invalid nonlinear-field initial iterate')
    trace = []
    for iteration in range(p['maximum_newton_steps'] + 1):
        value, gradient = value_and_gradient(field, graph, train, y, offset,
            exponent, p['regularization'], p['ridge'])
        norm = float(gradient.abs().max())
        row = dict(iteration=iteration, exponent=exponent, objective=float(value), maximum_gradient=norm)
        trace.append(row)
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite nonlinear-field energy or gradient')
        if norm <= p['gradient_tolerance']:
            return field, trace
        if iteration == p['maximum_newton_steps']:
            break
        difference = field[graph.u] - field[graph.v]
        conductance = p['regularization'] * graph.weight * (exponent - 1.) * difference.abs().pow(exponent - 2.)
        if exponent > 2. and len(conductance):
            conductance = conductance.clamp_min(p['conductance_floor'] * float(graph.weight.max()) * p['regularization'])
        probability = torch.sigmoid(offset + field[train])
        mass = field.new_full((graph.nodes,), p['ridge'])
        mass = mass.index_add(0, train, probability * (1. - probability))
        multiply, diagonal = graph.weighted_operator(conductance, mass)
        direction, cg = pcg(multiply, diagonal, -gradient, p['cg_relative_tolerance'],
            p['cg_absolute_tolerance'], p['maximum_cg_steps'])
        descent = float(gradient @ direction)
        if not np.isfinite(descent) or descent >= 0:
            raise FloatingPointError('The inexact Newton direction is not descending')
        row['cg'], row['directional_derivative'] = cg, descent
        for backtrack in range(p['maximum_backtracks']):
            step = 2. ** (-backtrack)
            proposed = field + step * direction
            proposed_value, _ = value_and_gradient(proposed, graph, train, y, offset,
                exponent, p['regularization'], p['ridge'])
            if float(proposed_value) <= float(value) + p['armijo'] * step * descent:
                field = proposed
                row.update(step=step, backtracks=backtrack)
                break
        else:
            raise ValueError('Nonlinear-field Armijo line search failed')
    raise ValueError('Nonlinear-field stationarity tolerance not reached')


def training_graph(a, p=None):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Nonlinear pair field accepts training arrays only')
    counts = a['node_counts']
    if not np.array_equal(a['pair_offset'], np.array([counts[0], counts[0] + counts[1]])):
        raise ValueError('The nonlinear field is registered for MiDrug only')
    return cartesian_graph(a['x_1'], a['x_2'], p)


@torch.no_grad()
def fit(a, device, p=None):
    p = parameters() if p is None else p
    graph, audit = training_graph(a, p)
    if a['anchor_oof'].shape != a['y_train'].shape:
        raise ValueError('Invalid fixed nested OOF reference')
    operator = PairGraph.from_scipy(graph, device)
    train = a['pairs_train'][0] * int(a['nd']) + a['pairs_train'][1]
    quadratic, q_trace = solve(operator, train, a['y_train'], a['anchor_oof'], 2., p)
    nonlinear, n_trace = solve(operator, train, a['y_train'], a['anchor_oof'], 3., p, initial=quadratic)
    return dict(full=nonlinear, quadratic=quadratic), dict(full=n_trace, quadratic=q_trace), audit, operator


def predict(field, pairs, drugs, clip=2.):
    pairs = torch.as_tensor(pairs, dtype=torch.long, device=field.device)
    if pairs.ndim != 2 or pairs.shape[0] != 2 or torch.any(pairs < 0) or torch.any(pairs[1] >= drugs):
        raise ValueError('Invalid RNA-drug query shape or indices')
    index = pairs[0] * drugs + pairs[1]
    if len(index) and torch.any(index >= len(field)):
        raise ValueError('Query outside frozen pair universe')
    return field[index].clamp(-clip, clip)
