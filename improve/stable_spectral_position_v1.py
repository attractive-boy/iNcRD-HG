"""SPE kernels and equivariant GIN pooling on the omitted-query bipartite graph.

The 821-node MiDrug graph uses its complete Laplacian eigendecomposition.
Learned positions feed a typed graph residual model, never context76 columns.
The constant-spectrum control keeps the same networks and rooted GIN pooling.
"""
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import torch
from torch import nn
from torch.nn import functional as F

from improve import sharpness_graph_v1 as backbone

KINDS = ('full', 'constant_spectrum')
identity_task = backbone.identity_task


def parameters():
    return dict(backbone.parameters(), pe_filters=8, psi_width=16, pe_width=16, pe_depth=2,
        spectral_scope='current_task_positive_bipartite', eigenvectors='complete',
        psi='elementwise_two_layer_tanh', rho='bias_free_dense_gin_then_column_sum')


def eigensystem(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
            or not np.isfinite(adjacency).all() or not np.array_equal(adjacency, adjacency.T)
            or (adjacency < 0).any()):
        raise ValueError('Expected a finite nonnegative symmetric adjacency')
    degree = adjacency.sum(1)
    inverse = np.divide(1., np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    # Symmetric normalized Laplacian convention: isolated nodes have L_ii = 1.
    laplacian = np.eye(len(degree)) - inverse[:, None] * adjacency * inverse[None, :]
    values, vectors = scipy.linalg.eigh(laplacian, driver='evd', check_finite=True)
    residual = float(np.max(np.abs(laplacian @ vectors - vectors*values[None, :])))
    orthogonal = float(np.max(np.abs(vectors.T @ vectors - np.eye(len(values)))))
    if residual > 1e-10 or orthogonal > 1e-10 or values.min() < -1e-10 or values.max() > 2+1e-10:
        raise ValueError('Full eigendecomposition failed numerical checks')
    audit = dict(nodes=len(values), retained_eigenvectors=len(values), truncated=False,
        undirected_edges=int(np.count_nonzero(adjacency)/2), isolated_nodes=int((degree == 0).sum()),
        eigen_residual_max_abs=residual, orthogonality_max_abs=orthogonal,
        minimum_eigenvalue=float(values.min()), maximum_eigenvalue=float(values.max()))
    return values.clip(0, 2), vectors, audit


@dataclass
class Context:
    physical: tuple
    adjacency: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    target_nodes: torch.Tensor
    audit: dict


def context(a, omitted, device):
    if backbone.target_type(a) != 1:
        raise ValueError('SPE is declared only for MiDrug')
    omitted = np.asarray(omitted, dtype=np.int64)
    nr, nd = int(a['node_counts'][1]), int(a['node_counts'][2])
    eligible = a['y_train'] == 1
    eligible = eligible.copy()
    eligible[omitted] = False
    pairs = a['pairs_train'][:, eligible]
    adjacency = np.zeros((nr+nd, nr+nd), dtype=np.float64)
    adjacency[pairs[0], nr+pairs[1]] = 1.
    adjacency[nr+pairs[1], pairs[0]] = 1.
    values, vectors, audit = eigensystem(adjacency)
    forbidden = a['pairs_train'][:, omitted]
    if np.any(adjacency[forbidden[0], nr+forbidden[1]] != 0):
        raise ValueError('A training query entered its spectral support')
    nodes = np.r_[np.arange(nr)+a['pair_offset'][0], np.arange(nd)+a['pair_offset'][1]]
    edges, relation = backbone.graph_lib.masked_physical(a, omitted)
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max())+1
    physical = backbone.graph_lib.graph_inputs(edges, relation, np.empty((2, 0), dtype=np.int64),
                                               n, relations, device)
    return Context(physical, torch.as_tensor(adjacency, device=device, dtype=torch.float32),
        torch.as_tensor(values, device=device, dtype=torch.float32),
        torch.as_tensor(vectors, device=device, dtype=torch.float32),
        torch.as_tensor(nodes, device=device), audit)


class DenseGINLayer(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.eps = nn.Parameter(torch.randn(()))
        self.mlp = nn.Sequential(nn.Linear(inputs, width, bias=False), nn.ReLU(),
                                 nn.Linear(width, width, bias=False))

    def forward(self, h, adjacency):
        n = h.shape[0]
        summed = (adjacency @ h.reshape(n, -1)).reshape_as(h)
        return self.mlp((1+self.eps)*h + summed)


class SpectralPosition(nn.Module):
    def __init__(self, p, kind):
        super().__init__()
        self.kind = kind
        self.psi = nn.ModuleList([nn.Sequential(nn.Linear(1, p['psi_width']), nn.Tanh(),
            nn.Linear(p['psi_width'], 1), nn.Tanh()) for _ in range(p['pe_filters'])])
        dimensions = [p['pe_filters']] + [p['pe_width']] * p['pe_depth']
        self.rho = nn.ModuleList([DenseGINLayer(dimensions[i], dimensions[i+1])
                                 for i in range(p['pe_depth'])])

    def coefficients(self, eigenvalues):
        weights = torch.cat([psi(eigenvalues[:, None]) for psi in self.psi], dim=1)
        if self.kind == 'constant_spectrum':
            weights = weights.mean(0, keepdim=True).expand_as(weights)
        return weights

    def kernels(self, eigenvalues, eigenvectors):
        weights = self.coefficients(eigenvalues)
        weighted = eigenvectors[None, :, :] * weights.T[:, None, :]
        return (weighted @ eigenvectors.T).permute(1, 2, 0).contiguous()

    def forward(self, eigenvalues, eigenvectors, adjacency):
        h = self.kernels(eigenvalues, eigenvectors)
        for layer in self.rho:
            h = layer(h, adjacency)
        return h.sum(1)


class PositionalGraph(backbone.OrdinaryGraph):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__(dimensions, relations, p)
        self.position = SpectralPosition(p, kind)
        self.position_norm = nn.LayerNorm(p['pe_width'])
        self.position_project = nn.Linear(p['pe_width'], p['width'], bias=False)

    def forward(self, xs, graph, pairs):
        edges, relation, counts = graph.physical
        parts, start = [], 0
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(project(torch.cat((x/rms, counts[start:start+len(x)]), dim=-1)))
            start += len(x)
        raw = torch.cat(parts)
        position = self.position(graph.eigenvalues, graph.eigenvectors, graph.adjacency)
        position = self.position_project(self.position_norm(position))
        padded = raw.new_zeros(raw.shape).index_copy(0, graph.target_nodes, position)
        h = F.gelu(self.norm(raw+padded))
        for layer in self.layers:
            h = layer(h, edges, relation)
        left, right = h[pairs[0]], h[pairs[1]]
        return self.readout(torch.cat((left, right, left*right, (left-right).abs()), dim=-1)).squeeze(-1)


def setup(a, p, kind, device):
    if (kind not in KINDS or p['eigenvectors'] != 'complete'
            or p['spectral_scope'] != 'current_task_positive_bipartite'):
        raise ValueError('Unregistered positional arm or truncated spectral recipe')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return backbone.IdentityResidual().to(device), [], [None]*p['folds'], folds
    graphs = [context(a, np.flatnonzero(a['folds'] == f), device) for f in range(p['folds'])]
    xs = [torch.as_tensor(a['x_'+str(i)], device=device) for i in range(len(a['node_counts']))]
    model = PositionalGraph([x.shape[1] for x in xs], int(a['relation'].max())+1, p, kind).to(device)
    return model, xs, graphs, folds
