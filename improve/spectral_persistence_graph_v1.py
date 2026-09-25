"""Learned SpectRe event spectra on complete one-hop common-neighbor graphs.

This is a local link-prediction adaptation, not the original graph-classification
experiment. All local nodes are retained; there is no node-ID truncation. The
ordinary typed support encoder and strong reference are shared infrastructure.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve import sharpness_graph_v1 as ordinary

ROOT = Path(__file__).resolve().parent
KINDS = ('full', 'no_spectrum', 'final_spectrum')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}
identity_task = ordinary.identity_task
_NATIVE = None


def native():
    global _NATIVE
    if _NATIVE is None:
        from improve.spectral_persistence_build_v1 import load_extension
        build = ROOT / 'results/spectral_persistence_native_v1_20260916'
        _NATIVE = load_extension('spectral_persistence_native_v1',
            ROOT / 'spectral_persistence_native_v1.cpp', build)
    return _NATIVE


def parameters():
    return dict(**ordinary.parameters(), filtration_count=1, filtration_hidden=16,
        spectrum_hidden=16, topology_width=32, filtration_epsilon=1e-4,
        neighborhood='all common one-hop neighbors plus both query endpoints',
        spectral_dtype='float64', spectral_zero_tolerance=1e-10,
        local_node_cap=None, equal_filtration_edges_processed_simultaneously=True)


def diagrams(vertex_values, edge_values, edges, vertex_slices, edge_slices):
    """Differentiable tuple values; combinatorial indices and spectra are detached.

    For unweighted filtered graphs, spectra are locally constant as long as the
    edge ordering does not change. Death, vertex color, and minimum incident edge
    retain their exact gather gradients. No derivative of a sort is invented.
    """
    if vertex_values.ndim != 1 or edge_values.ndim != 1:
        raise ValueError('This registered adapter uses one filtration')
    if (not torch.isfinite(vertex_values).all() or not torch.isfinite(edge_values).all()
            or not torch.all((edge_values > 0) & (edge_values < 1))):
        raise ValueError('Filtration values must be finite; edges must lie inside (0,1)')
    raw = native().event_spectra(vertex_values.detach().cpu().double().contiguous(),
        edge_values.detach().cpu().double().contiguous(), edges.cpu().long().contiguous(),
        vertex_slices.cpu().long().contiguous(), edge_slices.cpu().long().contiguous())
    device, dtype = vertex_values.device, vertex_values.dtype
    death, first, cycles = [value.to(device) for value in raw[:3]]
    padded = torch.cat((edge_values, edge_values.new_ones(1)))
    death_values = padded[torch.where(death >= 0, death, len(edge_values))]
    gamma_values = padded[torch.where(first >= 0, first, len(edge_values))]
    zero = torch.zeros_like(vertex_values)
    tuple0 = torch.stack((zero, death_values, vertex_values, gamma_values), dim=-1)
    tuple1 = torch.stack((torch.ones_like(edge_values), edge_values,
                         torch.zeros_like(edge_values), torch.zeros_like(edge_values)), dim=-1)
    rho0, rho1, final_rho = [value.to(device=device, dtype=dtype) for value in raw[3:6]]
    return dict(tuple0=tuple0, tuple1=tuple1, cycle_mask=cycles.bool(), rho0=rho0,
                rho1=rho1, final_rho=final_rho, counts=raw[6], native=raw)


class PersistenceContext:
    def __init__(self, edges, nodes, neural_graph):
        row, col = np.concatenate((edges, edges[::-1]), axis=1)
        keep = row != col
        graph = sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32),
                                   (row[keep], col[keep])), shape=(nodes, nodes))
        graph.sum_duplicates()
        graph.data[:] = 1.
        graph.sort_indices()
        self.adjacency, self.neural_graph, self.nodes = graph, neural_graph, int(nodes)
        self.items, self.packed = {}, OrderedDict()

    def extract(self, u, v):
        u, v = int(u), int(v)
        if u == v or not (0 <= u < self.nodes and 0 <= v < self.nodes):
            raise ValueError('Expected two distinct query endpoints')
        if self.adjacency[u, v] != 0 or self.adjacency[v, u] != 0:
            raise ValueError('Query fact must be removed before persistence extraction')
        key = (u, v)
        if key not in self.items:
            a = self.adjacency
            nu = a.indices[a.indptr[u]:a.indptr[u + 1]]
            nv = a.indices[a.indptr[v]:a.indptr[v + 1]]
            common = np.intersect1d(nu, nv, assume_unique=True)
            common = common[(common != u) & (common != v)]
            vertices = np.r_[u, v, common].astype(np.int64)
            induced = sparse.triu(a[vertices][:, vertices], k=1).tocoo()
            edges = np.stack((induced.row, induced.col), axis=1).astype(np.int64)
            if len(edges):
                order = np.lexsort((edges[:, 1], edges[:, 0]))
                edges = edges[order]
            self.items[key] = vertices, edges
        return self.items[key]

    def batch(self, pairs):
        array = pairs.detach().cpu().numpy()
        if array.shape[0] != 2 or array.ndim != 2 or not array.shape[1]:
            raise ValueError('Expected a nonempty pair matrix')
        key = (str(pairs.device), array.tobytes())
        if key not in self.packed:
            nodes, edges, node_batch, edge_batch = [], [], [], []
            vertex_slices, edge_slices = [0], [0]
            for i, (u, v) in enumerate(array.T):
                local_nodes, local_edges = self.extract(u, v)
                nodes.append(local_nodes)
                edges.append(local_edges + vertex_slices[-1])
                node_batch.append(np.full(len(local_nodes), i, dtype=np.int64))
                edge_batch.append(np.full(len(local_edges), i, dtype=np.int64))
                vertex_slices.append(vertex_slices[-1] + len(local_nodes))
                edge_slices.append(edge_slices[-1] + len(local_edges))
            result = dict(nodes=torch.as_tensor(np.concatenate(nodes), device=pairs.device),
                edges_cpu=torch.as_tensor(np.concatenate(edges), device='cpu'),
                vertex_slices=torch.tensor(vertex_slices), edge_slices=torch.tensor(edge_slices),
                node_batch=torch.as_tensor(np.concatenate(node_batch), device=pairs.device),
                edge_batch=torch.as_tensor(np.concatenate(edge_batch), device=pairs.device),
                count=array.shape[1])
            result['edges'] = result['edges_cpu'].to(pairs.device)
            result['node_counts'] = torch.as_tensor(np.diff(vertex_slices), device=pairs.device)
            self.packed[key] = result
            while len(self.packed) > 2:
                self.packed.popitem(last=False)
        self.packed.move_to_end(key)
        return self.packed[key]


class SpectralPersistenceGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        self.kind, self.p = kind, p
        width, top_width, spectrum_width = p['width'], p['topology_width'], p['spectrum_hidden']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([graph_lib.SparseTypedAttention(width, p['heads'],
            2 * relations + 2, p['dropout']) for _ in range(p['depth'])])
        def filtration():
            return nn.Sequential(nn.Linear(width, p['filtration_hidden']), nn.ReLU(),
                                 nn.Linear(p['filtration_hidden'], 1), nn.Sigmoid())
        self.vertex_filtration, self.edge_filtration = filtration(), filtration()
        self.spectrum = nn.ModuleList([nn.Sequential(nn.Linear(1, spectrum_width), nn.ReLU(),
            nn.Linear(spectrum_width, spectrum_width)) for _ in range(2)])
        self.diagram = nn.ModuleList([nn.Sequential(nn.Linear(4 + spectrum_width, top_width), nn.ReLU(),
            nn.Linear(top_width, top_width)) for _ in range(2)])
        self.topology = nn.Sequential(nn.LayerNorm(top_width), nn.Linear(top_width, top_width),
                                      nn.ReLU(), nn.Linear(top_width, top_width))
        self.readout = nn.Sequential(nn.Linear(4 * width + top_width, width), nn.GELU(),
                                    nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        self.persistence_diagnostics = dict(calls=0, graphs=0, components=0, cycles=0,
                                            eigendecompositions=0, max_local_nodes=0)

    def encode(self, xs, context):
        edges, relation, counts = context.neural_graph
        parts, start = [], 0
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(project(torch.cat((x / rms, counts[start:start + len(x)]), dim=-1)))
            start += len(x)
        h = F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        return h

    def topology_features(self, h, packed):
        local_h = h[packed['nodes']]
        eps = self.p['filtration_epsilon']
        fv = eps + (1. - 2. * eps) * self.vertex_filtration(local_h).squeeze(-1)
        e = packed['edges']
        fe = eps + (1. - 2. * eps) * self.edge_filtration(local_h[e[:, 0]] + local_h[e[:, 1]]).squeeze(-1)
        data = diagrams(fv, fe, packed['edges_cpu'], packed['vertex_slices'], packed['edge_slices'])
        stats = data['counts'].sum(0).tolist()
        self.persistence_diagnostics['calls'] += 1
        self.persistence_diagnostics['graphs'] += packed['count']
        for name, value in zip(('components', 'cycles', 'eigendecompositions'), stats):
            self.persistence_diagnostics[name] += int(value)
        self.persistence_diagnostics['max_local_nodes'] = max(self.persistence_diagnostics['max_local_nodes'],
            int(packed['node_counts'].max()))
        embeddings = []
        for dimension in (0, 1):
            index = packed['node_batch'] if dimension == 0 else packed['edge_batch']
            rho = data['rho' + str(dimension)]
            if self.kind == 'no_spectrum':
                rho = torch.zeros_like(rho)
            elif self.kind == 'final_spectrum':
                rho = data['final_rho'][index]
            valid = rho > 0.
            # A fixed invertible log transform controls scale without discarding
            # eigenvalues. Padding and the zero Laplacian eigenvalues are masked.
            encoded_rho = self.spectrum[dimension](torch.log1p(rho.clamp_min(0.))[..., None])
            encoded_rho = (encoded_rho * valid[..., None]).sum(1)
            values = self.diagram[dimension](torch.cat((data['tuple' + str(dimension)], encoded_rho), -1))
            if dimension == 1:
                values = values * data['cycle_mask'][:, None]
            pooled = values.new_zeros((packed['count'], values.shape[1])).index_add(0, index, values)
            if dimension == 0:
                pooled = pooled / packed['node_counts'][:, None]
            embeddings.append(pooled)
        return self.topology(embeddings[0] + embeddings[1])

    def forward(self, xs, context, pairs):
        h = self.encode(xs, context)
        topo = self.topology_features(h, context.batch(pairs))
        left, right = h[pairs[0]], h[pairs[1]]
        return self.readout(torch.cat((left, right, left * right, (left - right).abs(), topo), -1)).squeeze(-1)


def setup(a, p, kind, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Spectral persistence setup accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered spectral-persistence arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return ordinary.IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    graphs = []
    for f in range(p['folds']):
        edges, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        neural = graph_lib.graph_inputs(edges, labels, np.empty((2, 0), dtype=np.int64), n, relations, device)
        graphs.append(PersistenceContext(edges, n, neural))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = SpectralPersistenceGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
