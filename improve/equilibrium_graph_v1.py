"""Typed-channel equilibrium propagation adapted from released MEGNN code.

The journal abstract and pinned author implementation were inspected; journal
full text was not available. The implemented linear fixed-point equation and
its implicit gradient are stated explicitly and checked independently. This
is not a claim to reproduce all of the paper's architecture or theory.
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task


KINDS = ('full', 'two_hops')


def normalized_gram(factor):
    gram = factor.T @ factor
    return gram / (torch.linalg.vector_norm(gram) + 1e-12)


def relative_residual(value, rhs, adjacency, coupling, gamma):
    residual = rhs + gamma * (torch.sparse.mm(adjacency, value) @ coupling) - value
    return torch.linalg.vector_norm(residual) / (1. + torch.linalg.vector_norm(rhs))


def solve_fixed_point(rhs, adjacency, coupling, gamma, tolerance, max_iterations):
    value = rhs.clone()
    denominator = 1. + torch.linalg.vector_norm(rhs)
    for iteration in range(1, max_iterations + 1):
        updated = rhs + gamma * (torch.sparse.mm(adjacency, value) @ coupling)
        difference = torch.linalg.vector_norm(updated - value) / denominator
        value = updated
        if float(difference) <= tolerance:
            residual = float(relative_residual(value, rhs, adjacency, coupling, gamma))
            if residual <= tolerance:
                return value, dict(iterations=iteration, relative_residual=residual)
    raise ArithmeticError('The registered equilibrium residual tolerance was not reached')


class ImplicitEquilibrium(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs, coupling, adjacency, transposed, gamma, tolerance, max_iterations, record):
        value, diagnostics = solve_fixed_point(rhs, adjacency, coupling, gamma, tolerance, max_iterations)
        record['forward'] = diagnostics
        ctx.save_for_backward(value, coupling, adjacency, transposed)
        ctx.gamma, ctx.tolerance, ctx.max_iterations, ctx.record = gamma, tolerance, max_iterations, record
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        value, coupling, adjacency, transposed = ctx.saved_tensors
        adjoint, diagnostics = solve_fixed_point(gradient, transposed, coupling.T,
            ctx.gamma, ctx.tolerance, ctx.max_iterations)
        ctx.record['backward'] = diagnostics
        coupling_gradient = ctx.gamma * torch.sparse.mm(adjacency, value).T @ adjoint
        return adjoint, coupling_gradient, None, None, None, None, None, None


def equilibrium(rhs, factor, adjacency, transposed, p, kind, record):
    if kind not in KINDS or not (0. < p['gamma'] < 1.):
        raise ValueError('Unknown arm or noncontractive shrinkage')
    rhs64 = rhs.double()
    coupling = normalized_gram(factor.double())
    if kind == 'full':
        value = ImplicitEquilibrium.apply(rhs64, coupling, adjacency, transposed,
            p['gamma'], p['solver_tolerance'], p['solver_max_iterations'], record)
    else:
        # Start with the injected attributes, then include exactly one- and
        # two-hop contributions. The full arm continues to equilibrium.
        value = rhs64
        for _ in range(p['control_propagation_steps']):
            value = rhs64 + p['gamma'] * (torch.sparse.mm(adjacency, value) @ coupling)
        record['finite_propagation_steps'] = p['control_propagation_steps']
    return value.to(rhs.dtype)


def torch_csr(matrix, device):
    matrix = matrix.tocsr()
    return torch.sparse_csr_tensor(
        torch.as_tensor(matrix.indptr, dtype=torch.long, device=device),
        torch.as_tensor(matrix.indices, dtype=torch.long, device=device),
        torch.as_tensor(matrix.data, dtype=torch.float64, device=device),
        size=matrix.shape, dtype=torch.float64, device=device)


def relation_channels(edges, relation, n, relations, device):
    counts = np.zeros((n, relations), dtype=np.float32)
    np.add.at(counts, (edges[0], relation), 1.)
    np.add.at(counts, (edges[1], relation), 1.)
    channels = []
    for kind in range(relations):
        selected = edges[:, relation == kind]
        row = np.r_[selected[0], selected[1], np.arange(n)]
        col = np.r_[selected[1], selected[0], np.arange(n)]
        physical = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n)).tocsr()
        # Binary, symmetric physical support plus an identity at every node.
        # This also specifies an unambiguous isolated-node boundary condition.
        physical.data[:] = 1.
        degree = np.asarray(physical.sum(axis=1)).ravel()
        transition = sp.diags(1. / degree) @ physical
        np.testing.assert_allclose(np.asarray(transition.sum(axis=1)).ravel(), 1., atol=1e-12, rtol=0)
        channels.append((torch_csr(transition, device), torch_csr(transition.T, device)))
    return channels, torch.as_tensor(np.log1p(counts), device=device)


class EquilibriumGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown equilibrium arm')
        self.p, self.kind = dict(p), kind
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.input_norm = nn.LayerNorm(width)
        self.factors = nn.ParameterList([nn.Parameter(torch.empty(width, width)) for _ in range(relations)])
        for factor in self.factors:
            nn.init.xavier_uniform_(factor)
        self.channel_project = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(relations)])
        self.attention = nn.Sequential(nn.Linear(width, 16), nn.Tanh(), nn.Linear(16, 1, bias=False))
        self.output_norm = nn.LayerNorm(width)
        self.readout = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
                                    nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        self.solver_records = []

    def encode(self, xs, graph):
        channels, counts = graph
        parts, start = [], 0
        for x, projection in zip(xs, self.project):
            scale = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(projection(torch.cat((x / scale, counts[start:start + len(x)]), dim=-1)))
            start += len(x)
        injected = F.gelu(self.input_norm(torch.cat(parts)))
        states = []
        for index, ((adj, transposed), factor, projection) in enumerate(
                zip(channels, self.factors, self.channel_project)):
            record = dict(channel=index, training=self.training)
            state = equilibrium(injected, factor, adj, transposed, self.p, self.kind, record)
            self.solver_records.append(record)
            states.append(projection(state))
        stack = torch.stack(states, dim=1)
        weights = self.attention(stack).softmax(dim=1)
        return self.output_norm((weights * stack).sum(dim=1))

    def forward(self, xs, graph, pairs):
        hidden = self.encode(xs, graph)
        left, right = hidden[pairs[0]], hidden[pairs[1]]
        features = torch.cat((left, right, left * right, (left - right).abs()), dim=-1)
        return self.readout(features).squeeze(-1)


def parameters():
    return dict(width=32, gamma=.8, solver_tolerance=1e-8, solver_max_iterations=200,
                control_propagation_steps=2, dropout=.1, folds=5,
                lnc_policy='identity_strong_reference')


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered equilibrium arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    graphs = []
    for fold in range(p['folds']):
        edges, kinds = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == fold))
        graphs.append(relation_channels(edges, kinds, n, relations, device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = EquilibriumGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds


def solver_summary(model):
    records = getattr(model, 'solver_records', [])
    result = dict(calls=len(records), training_calls=sum(r['training'] for r in records))
    for direction in ('forward', 'backward'):
        values = [r[direction] for r in records if direction in r]
        result[direction] = dict(calls=len(values),
            max_iterations=max([v['iterations'] for v in values], default=0),
            max_relative_residual=max([v['relative_residual'] for v in values], default=0.))
    return result
