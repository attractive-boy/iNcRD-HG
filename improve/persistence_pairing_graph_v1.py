"""Joint persistence tuples versus their separately sorted coordinate marginals.

The primary is exactly the previously observed no-spectrum SpectRe arm. This
narrows its hypothesis to tuple correspondence and does not restore a claim
that event spectra improved performance.
"""
import numpy as np
import torch

from improve import spectral_persistence_graph_v1 as parent

KINDS = ('full', 'marginal_only')
TRAIN_KEYS, parameters, identity_task = parent.TRAIN_KEYS, parent.parameters, parent.identity_task
PersistenceContext, graph_lib = parent.PersistenceContext, parent.graph_lib


def marginal_tuple(tuple0, node_batch, vertex_slices, node_counts):
    """Replace the joint empirical tuple measure by a comonotone coupling.

    Sorting each coordinate separately preserves every marginal and the number
    of events. It discards the observed association between death, vertex color
    and first incident edge. The resulting tuples need not describe a graph.
    """
    if tuple0.ndim != 2 or tuple0.shape[1] != 4 or node_batch.shape != (len(tuple0),):
        raise ValueError('Expected four-dimensional zero-persistence tuples')
    if len(vertex_slices) != len(node_counts) + 1 or not torch.all(node_counts > 0):
        raise ValueError('Expected nonempty ragged local graphs')
    start = vertex_slices[:-1].to(tuple0.device)
    position = torch.arange(len(tuple0), device=tuple0.device) - start[node_batch]
    width = int(node_counts.max())
    padded = tuple0.new_full((len(node_counts), width, 4), torch.inf)
    padded[node_batch, position] = tuple0
    sorted_coordinates = padded.sort(dim=1).values
    return sorted_coordinates[node_batch, position]


class PairingPersistenceGraph(parent.SpectralPersistenceGraph):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__(dimensions, relations, p, 'no_spectrum')
        self.pairing_kind = kind

    def topology_features(self, h, packed):
        if self.pairing_kind == 'full':
            return super().topology_features(h, packed)
        local_h = h[packed['nodes']]
        eps = self.p['filtration_epsilon']
        fv = eps + (1. - 2. * eps) * self.vertex_filtration(local_h).squeeze(-1)
        e = packed['edges']
        fe = eps + (1. - 2. * eps) * self.edge_filtration(local_h[e[:, 0]] + local_h[e[:, 1]]).squeeze(-1)
        data = parent.diagrams(fv, fe, packed['edges_cpu'], packed['vertex_slices'], packed['edge_slices'])
        stats = data['counts'].sum(0).tolist()
        self.persistence_diagnostics['calls'] += 1
        self.persistence_diagnostics['graphs'] += packed['count']
        for name, value in zip(('components', 'cycles', 'eigendecompositions'), stats):
            self.persistence_diagnostics[name] += int(value)
        self.persistence_diagnostics['max_local_nodes'] = max(self.persistence_diagnostics['max_local_nodes'],
            int(packed['node_counts'].max()))
        data['tuple0'] = marginal_tuple(data['tuple0'], packed['node_batch'],
                                        packed['vertex_slices'], packed['node_counts'])
        embeddings = []
        for dimension in (0, 1):
            index = packed['node_batch'] if dimension == 0 else packed['edge_batch']
            rho = torch.zeros_like(data['rho' + str(dimension)])
            valid = rho > 0.
            # Preserve even the zero spectrum computation and its zero gradients
            # so the module, initialization, AdamW and RNG budgets match parent.
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


def setup(a, p, kind, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Persistence pairing accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered persistence pairing arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return parent.ordinary.IdentityResidual().to(device), [], [None] * p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    graphs = []
    for f in range(p['folds']):
        edges, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        neural = graph_lib.graph_inputs(edges, labels, np.empty((2, 0), dtype=np.int64), n, relations, device)
        graphs.append(PersistenceContext(edges, n, neural))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = PairingPersistenceGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
