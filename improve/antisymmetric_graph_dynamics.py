"""Space/weight antisymmetric propagation inspired by SWAN (AAAI 2025).

This independent link-prediction adaptation implements the mathematical
operators in equations 11 and 20, with shared weights across Euler steps.
The graph is Boolean, undirected and query-masked before normalization.
Continuous-time non-dissipativity is not a guarantee for the finite fitted
network, which also has damping, an encoder, a decoder and finite steps.
"""
from datetime import datetime, timezone
import math
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve.overlap_block_models import masked_physical, graph_inputs

ROOT = common.base.ROOT
OUT = ROOT / 'results/antisymmetric_dynamics_core_20260915'
KINDS = ('full', 'no_space', 'free_channels', 'shallow')


def spatial_operators(edges, n, device):
    """Return Ahat + Ahat.T and P - P.T, with P = D^-1 A.

    For a Boolean undirected edge (u,v), the second weight is exactly
    1/degree(u) - 1/degree(v). Node indices never index an edge-weight array.
    Parallel physical relation edges are deduplicated only for these spatial
    operators; relation counts remain available to every encoder.
    """
    arcs = np.unique(np.concatenate((edges, edges[::-1]), axis=1), axis=1)
    arcs = arcs[:, arcs[0] != arcs[1]]
    degrees = np.bincount(arcs[0], minlength=n).astype(np.float64)
    inv = np.divide(1., degrees, out=np.zeros_like(degrees), where=degrees > 0)
    weights = 2 * np.sqrt(inv[arcs[0]] * inv[arcs[1]])
    difference = inv[arcs[0]] - inv[arcs[1]]
    indices = torch.as_tensor(arcs, device=device)
    def sparse(values):
        return torch.sparse_coo_tensor(indices, torch.as_tensor(values, device=device, dtype=torch.float32),
                                      (n, n), device=device).coalesce()
    return sparse(weights), sparse(difference)


def context(a, fold, device):
    edges, relations = masked_physical(a, np.flatnonzero(a['folds'] == fold))
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max() + 1)
    graph = graph_inputs(edges, relations, np.empty((2, 0), dtype=np.int64), n, nr, device)
    symmetric, antisymmetric = spatial_operators(edges, n, device)
    return symmetric, antisymmetric, graph[2]


class AntisymmetricDynamics(nn.Module):
    def __init__(self, width, depth, epsilon, gamma, beta, kind):
        super().__init__()
        if kind not in KINDS or depth < 1 or epsilon <= 0 or gamma < 0:
            raise ValueError('Invalid antisymmetric-dynamics configuration')
        self.kind, self.depth = kind, depth
        self.epsilon, self.gamma, self.beta = epsilon, gamma, beta
        self.W = nn.Parameter(torch.empty(width, width))
        self.V = nn.Parameter(torch.empty(width, width))
        self.Z = nn.Parameter(torch.empty(width, width))
        self.bias = nn.Parameter(torch.zeros(width))
        for weight in (self.W, self.V, self.Z):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    def channel_matrices(self):
        if self.kind == 'free_channels':
            return self.W, self.V, self.Z
        return self.W - self.W.T, self.V - self.V.T, self.Z + self.Z.T

    def linear_field(self, hidden, symmetric, antisymmetric, matrices=None):
        """Undamped field; full-arm vectorized Jacobian is skew symmetric."""
        w, v, z = self.channel_matrices() if matrices is None else matrices
        result = hidden @ w + torch.sparse.mm(symmetric, hidden) @ v
        if self.kind != 'no_space':
            result = result + self.beta * (torch.sparse.mm(antisymmetric, hidden) @ z)
        return result

    def forward(self, hidden, symmetric, antisymmetric):
        matrices = self.channel_matrices()
        steps = 2 if self.kind == 'shallow' else self.depth
        for _ in range(steps):
            field = self.linear_field(hidden, symmetric, antisymmetric, matrices)
            hidden = hidden + self.epsilon * torch.tanh(field - self.gamma * hidden + self.bias)
        return hidden


class AntisymmetricGraph(nn.Module):
    def __init__(self, dims, node_counts, relations, kind, width=64, depth=16,
                 epsilon=.1, gamma=.1, beta=.5, dropout=.1):
        super().__init__()
        self.node_counts = list(map(int, node_counts))
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dims])
        self.input_norm, self.output_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dynamics = AntisymmetricDynamics(width, depth, epsilon, gamma, beta, kind)
        self.dropout = nn.Dropout(dropout)
        self.rna_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.drug_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.scale = math.sqrt(width)

    def forward(self, xs, graph, pairs):
        symmetric, antisymmetric, counts = graph
        pieces, offset = [], 0
        for x, project, size in zip(xs, self.project, self.node_counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            pieces.append(project(torch.cat((x / rms, counts[offset:offset + size]), 1)))
            offset += size
        field = self.dropout(torch.tanh(self.input_norm(torch.cat(pieces))))
        hidden = self.dynamics(field, symmetric, antisymmetric)
        hidden = self.dropout(self.output_norm(hidden))
        r, d = pairs
        score = (self.rna_readout(hidden)[r] * self.drug_readout(hidden)[d]).sum(-1) / self.scale
        return score + self.rna_degree(counts[r]).squeeze(-1) + self.drug_degree(counts[d]).squeeze(-1)


def parameters():
    return dict(kinds=list(KINDS), width=64, depth=16, epsilon=.1, gamma=.1,
                beta=.5, dropout=.1, folds=5, auxiliary_weight=0., primary_kind='full',
                primary_controls=['no_space', 'free_channels'])


def setup(a, p, kind, device):
    graphs = [context(a, fold, device) for fold in range(p['folds'])]
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = AntisymmetricGraph([x.shape[1] for x in xs], a['node_counts'], int(a['relation'].max() + 1),
        kind, p['width'], p['depth'], p['epsilon'], p['gamma'], p['beta'], p['dropout']).to(device)
    return model, xs, graphs, folds


def prepare():
    if OUT.exists():
        raise ValueError('Use a new antisymmetric-dynamics specification directory')
    parent = common.protocol(common.OUT)
    sources = ('antisymmetric_graph_dynamics.py', 'test_antisymmetric_graph_dynamics.py',
               'overlap_block_models.py', 'overlap_block_experiment.py', 'non_tabular_experiment.py')
    literature_files = [ROOT / 'research/new_mechanisms_20260915' / name for name in
                        ('swan_paper.pdf', 'swan_author_model.py', 'SWAN_AUTHOR_DISCOVERY.json')]
    p = dict(**parameters(), created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=parent['tasks'], selection=parent['selection'], specification_only=True,
        method='antisymmetric_dynamics_core', family='space_weight_antisymmetric_dynamics',
        literature=dict(title='On Oversquashing in Graph Neural Networks Through the Lens of Dynamical Systems',
                        venue='AAAI 2025', doi='10.1609/aaai.v39i16.33858', arxiv='2405.01009v2'),
        controls='Same raw matrices, encoder, decoder, initialization and training budget. no_space removes only the P-P.T branch; its unused Z is not counted as trained capacity. free_channels retains both spatial branches but uses unconstrained W,V,Z. shallow keeps the full constraints but takes 2 instead of 16 shared steps and is diagnostic. Primary full must beat no_space and free_channels on mean MiDrug AUC and AP, plus the offset null.',
        adaptation='Boolean undirected union of query-masked physical relations; original attributes and typed relation counts enter a 64-dimensional encoder. Shared W,V,Z for 16 tanh Euler steps, epsilon=0.1, gamma=0.1, beta=0.5. Ahat+Ahat.T and P-P.T follow equations 11/20. Endpoint readouts and fixed support offsets adapt graph-level SWAN to link prediction. Dropout and LayerNorm appear outside the dynamical steps only. No learned adjacency variant.',
        source_differences='Author get_adj uses edge_weight[col] inside antisym normalization, with col a node index rather than a reverse-edge index; this need not be antisymmetric. Local code constructs the exact P-P.T operator and tests it. Channel matrices use full W-W.T etc, whereas author V/Z parametrizations use triangular entries. Appendix A.3 prints a stability inequality >1 contrary to A.1; the Euler stability region is the unit disk. We do not execute author code.',
        theory_scope='The undamped linear vector field is exactly skew symmetric, and its frozen tanh Jacobian has imaginary eigenvalues. Gamma damping, finite Euler discretization and changing Jacobians mean these identities are not a proof of constant information flow for the complete trained predictor.',
        boundary='Remove fitting query facts before degree computation and both graph operators. Attributes are unchanged; complete support graph inference has no query-batch statistics, numeric IDs, added tabular columns, external biology, pretrained representations or kernels.',
        source_sha256={n: common.base.digest(ROOT / n) for n in sources},
        inputs={str(x): common.base.digest(x) for x in literature_files},
        innovation_counting='One constrained graph-dynamics family. ADGN, SWAN and future dynamics variants do not count separately. Classical offsets and composition do not count as innovations.')
    OUT.mkdir(parents=True)
    (OUT / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, OUT / 'source_snapshot' / name)
    common.base.write_json(OUT / 'protocol.json', p)
