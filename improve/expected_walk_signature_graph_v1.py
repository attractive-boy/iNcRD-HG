"""Exact second-level signatures of support-graph random-walk bridges.

Independent link-prediction adaptation of the tensor-walk construction in
Toth et al., NeurIPS 2022, arXiv:2205.14092. A walk is conditioned to end at
the queried drug; its ordered increments are differences of learned node
attributes. This finite level-two model has no full-signature injectivity
guarantee. There is no physical-time channel or ordering of entity IDs.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'symmetrized', 'first_level')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}


def parameters():
    return dict(width=64, heads=4, depth=2, dropout=.1, folds=5,
        signature_rank=8, walk_steps=4, signature_level=2,
        transition='fixed uniform outgoing support arcs including reverse arcs and self loops',
        signature_dtype='float64', projection='two bounded learned node coordinates per rank',
        bridge_scope='all nodes to all drug endpoints, independent of query list',
        nonexistent_bridge='zero signature, zero reachability, unchanged endpoint decoder',
        lnc_policy='identity_strong_reference')


def sparse_left(matrix, values):
    """Left multiplication on the node axis; all remaining axes are batched."""
    shape = values.shape
    return torch.sparse.mm(matrix, values.reshape(shape[0], -1)).reshape(
        matrix.shape[0], *shape[1:])


def bridge_context(physical, target_indices, steps):
    """Fixed Markov transition and exact finite-horizon endpoint masses.

    Parallel arcs are kept as separate observations when calculating degrees
    and are then added by sparse coalescing. This preserves a uniform law on
    the original outgoing arc multiset. Nothing depends on query outcomes.
    """
    edges, relation, counts = physical
    n = len(counts)
    if steps < 1 or target_indices.ndim != 1:
        raise ValueError('Positive walk length and a vector of endpoints required')
    src = edges[0]
    degrees = torch.bincount(src, minlength=n).double()
    if torch.any(degrees == 0):
        raise ValueError('Each walk origin needs an outgoing arc (including self loops)')
    p = torch.sparse_coo_tensor(edges, degrees[src].reciprocal(), (n, n)).coalesce()
    total = torch.sparse.sum(p, dim=1).to_dense()
    # Sequential/coalesced summation error grows with the outgoing arc count.
    # A fixed 4e-14 limit was too strict for a high-degree real graph vertex.
    error_bound = 8*torch.finfo(torch.float64).eps*(degrees+1)
    if torch.any((total-1.).abs() > error_bound):
        raise FloatingPointError('Transition rows fail the positive-summation roundoff bound')
    initial = torch.zeros((n, len(target_indices)), device=edges.device, dtype=torch.float64)
    initial[target_indices, torch.arange(len(target_indices), device=edges.device)] = 1.
    masses = [initial]
    for _ in range(steps):
        masses.append(torch.sparse.mm(p, masses[-1]))
    if not all(torch.isfinite(x).all() and torch.all(x >= 0) for x in masses):
        raise FloatingPointError('Invalid bridge probability')
    return dict(physical=physical, transition=p, masses=masses, targets=target_indices)


def expected_levy_mass(transition, masses, u, v):
    """Unnormalized expected polygonal Levy area for every origin/endpoint.

    For a path x_0,...,x_K, the area in coordinate pair (u,v) is
      1/2 sum_t (u_t v_{t+1}-v_t u_{t+1})
        -1/2 (u_0 v_K-v_0 u_K).
    The recurrence below computes the first term exactly over all paths.
    The fixed-endpoint correction is applied by ``bridge_signature``.
    No sparse matrix with trainable values is needed: the two terms in
    diag(u) P diag(v)-diag(v) P diag(u) give an exact dense-gradient form.
    """
    if u.shape != v.shape or u.ndim != 2 or u.dtype != torch.float64 or v.dtype != torch.float64:
        raise ValueError('Expected float64 node/rank coordinate matrices')
    u, v = u[:, None, :], v[:, None, :]
    current = u.new_zeros((len(u), masses[0].shape[1], u.shape[-1]))
    for k in range(1, len(masses)):
        previous_mass = masses[k-1][..., None]
        local = .5*(u*sparse_left(transition, v*previous_mass)
                    -v*sparse_left(transition, u*previous_mass))
        current = sparse_left(transition, current)+local
    return current


def bridge_signature(context, u, v, kind):
    """Conditional first moments and one ordered level-two contraction/rank."""
    if kind not in KINDS:
        raise ValueError('Unregistered signature control')
    target = context['targets']
    mass = context['masses'][-1]
    reachable = mass > 0
    left_u, left_v = u[:, None, :], v[:, None, :]
    right_u, right_v = u[target][None, :, :], v[target][None, :, :]
    du, dv = right_u-left_u, right_v-left_v
    # The fixed-endpoint shuffle identity is Sym(S^2) = delta tensor^2/2.
    symmetric = .5*du*dv
    if kind == 'full':
        wedge = expected_levy_mass(context['transition'], context['masses'], u, v)
        safe_mass = torch.where(reachable, mass, torch.ones_like(mass))
        area = wedge/safe_mass[..., None]-.5*(left_u*right_v-left_v*right_u)
        second = symmetric+area
    elif kind == 'symmetrized':
        second = symmetric
    else:
        # Keep the same decoder shape; removing degree two makes its columns
        # inactive in this ablation. Parameter count is not an active-rank claim.
        second = symmetric*0.
    result = torch.cat((du, dv, second), -1)*reachable[..., None]
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite conditional walk signature')
    return result


class ExpectedWalkSignatureGraph(nn.Module):
    def __init__(self, dimensions, node_counts, relations, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unregistered signature control')
        self.p, self.kind = dict(p), kind
        width, rank = p['width'], p['signature_rank']
        self.project = nn.ModuleList([nn.Linear(d+relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([graph_lib.SparseTypedAttention(width, p['heads'],
            2*relations+2, p['dropout']) for _ in range(p['depth'])])
        self.coordinates = nn.Linear(width, 2*rank, bias=False)
        self.signature_norm = nn.LayerNorm(3*rank)
        self.readout = nn.Sequential(nn.Linear(4*width+3*rank+2, width), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        offsets = np.r_[0, np.cumsum(node_counts)]
        self.drug_start, self.drug_stop = int(offsets[2]), int(offsets[3])
        self.rna_start, self.rna_stop = int(offsets[1]), int(offsets[2])
        self.signature_diagnostics = dict(forwards=0, bridge_recurrences=0,
            bridge_sparse_products=0, maximum_absolute_signature=0.)

    def encode(self, xs, physical):
        edges, relation, counts = physical
        parts, start = [], 0
        for x, projection in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(projection(torch.cat((x/rms, counts[start:start+len(x)]), -1)))
            start += len(x)
        h = F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        return h

    def forward(self, xs, graph, pairs):
        r, d = pairs
        if torch.any((r < self.rna_start) | (r >= self.rna_stop)) or torch.any(
                (d < self.drug_start) | (d >= self.drug_stop)):
            raise ValueError('Walk-bridge decoder expects MiDrug pairs')
        h = self.encode(xs, graph['physical'])
        u, v = self.coordinates(h).tanh().double().chunk(2, -1)
        signature = bridge_signature(graph, u, v, self.kind)
        di = d-self.drug_start
        chosen = signature[r, di]
        chosen = self.signature_norm(chosen.to(h.dtype))
        mass = graph['masses'][-1][r, di]
        support = torch.stack((torch.log1p(len(graph['targets'])*mass), (mass > 0).double()), -1).to(h.dtype)
        left, right = h[r], h[d]
        feature = torch.cat((left, right, left*right, (left-right).abs(), chosen, support), -1)
        diag = self.signature_diagnostics
        diag['forwards'] += 1
        if self.kind == 'full':
            diag['bridge_recurrences'] += self.p['walk_steps']
            diag['bridge_sparse_products'] += 3*self.p['walk_steps']
        diag['maximum_absolute_signature'] = max(diag['maximum_absolute_signature'], float(signature.detach().abs().max()))
        return self.readout(feature).squeeze(-1)


def setup(a, p, kind, device):
    if set(a)-TRAIN_KEYS:
        raise ValueError('Expected-walk setup accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered signature control')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        model = IdentityResidual().to(device)
        model.signature_diagnostics = {}
        return model, [], [None]*p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max())+1
    starts = np.r_[0, np.cumsum(a['node_counts'])]
    targets = torch.arange(int(starts[2]), int(starts[3]), device=device)
    graphs = []
    for fold in range(p['folds']):
        e, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == fold))
        physical = graph_lib.graph_inputs(e, labels, np.empty((2, 0), dtype=np.int64), n, relations, device)
        graphs.append(bridge_context(physical, targets, p['walk_steps']))
    xs = [torch.as_tensor(a['x_'+str(i)], device=device) for i in range(len(a['node_counts']))]
    model = ExpectedWalkSignatureGraph([x.shape[1] for x in xs], a['node_counts'], relations, p, kind).to(device)
    return model, xs, graphs, folds
