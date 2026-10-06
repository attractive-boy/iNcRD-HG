"""Source-type contrastive learning on a query-masked bipartite graph.

Independent NT-SSM adaptation (arXiv:2605.24015v1): linear propagation
preserves the RNA/drug source decomposition; only negative logits are
type-reweighted. Attributes, a residual decoder and joint BCE are local
adaptations, shared by all controls. No recommendation benchmark data is used.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import sharpness_graph_v1 as identity

KINDS = ('full', 'type_shared', 'bce')


def parameters():
    return dict(width=64, depth=2, dropout=.1, folds=5,
        alpha_uu=1.2, alpha_ii=.8, alpha_ui=.8, alpha_iu=1.,
        temperature=.2, auxiliary_weight=.1,
        lnc_policy='identity_strong_reference')


def source_propagation(initial, adjacency, rna_count, depth):
    """Return N x 2 x D contributions from initial RNA and drug states."""
    if depth < 1 or not 0 < rna_count < len(initial):
        raise ValueError('Expected a nonempty bipartite graph and positive depth')
    mask = (torch.arange(len(initial), device=initial.device) < rna_count).to(initial.dtype)
    h = torch.stack((initial * mask[:, None], initial * (1-mask[:, None])), dim=1)
    total = h
    for _ in range(depth):
        h = torch.sparse.mm(adjacency, h.flatten(1)).reshape_as(h)
        total = total + h
    return total / (depth + 1)


def type_logits(left, right, coefficients):
    """Full cosine on diagonal; source-weighted cosine off diagonal.

left/right: B x 2 x D, with sources [RNA, drug]. The denominator uses
the complete state, identically for every bilinear term. Norm gradients
remain active; no approximate gradient theorem is asserted.
"""
    if left.shape != right.shape or left.ndim != 3 or left.shape[1] != 2:
        raise ValueError('Expected paired B x 2 x D source states')
    whole_left, whole_right = left.sum(1), right.sum(1)
    denominator = whole_left.norm(dim=-1)[:, None] * whole_right.norm(dim=-1)[None, :] + 1e-8
    positive = whole_left @ whole_right.T / denominator
    weighted = right * right.new_tensor(coefficients)[None, :, None]
    negative = whole_left @ weighted.sum(1).T / denominator
    diagonal = torch.eye(len(left), dtype=torch.bool, device=left.device)
    return torch.where(diagonal, positive, negative)


def directional_loss(left, right, coefficients, temperature, allowed_negative):
    if temperature <= 0 or allowed_negative.shape != (len(left), len(left)):
        raise ValueError('Invalid temperature or negative mask')
    logits = type_logits(left, right, coefficients) / temperature
    diagonal = torch.eye(len(left), dtype=torch.bool, device=left.device)
    logits = logits.masked_fill(~(allowed_negative | diagonal), -torch.inf)
    return (torch.logsumexp(logits, dim=1) - logits.diag()).mean()


def contrastive_loss(rna, drug, pairs, kind, p):
    """Two directions, avoiding false negatives from repeated batch endpoints.

An off-diagonal pair sharing either endpoint with its anchor positive is
excluded. This is a batch-supervision mask, never an inference graph input.
Other unobserved pairs remain implicit contrastive alternatives, not verified
biological negatives. All objective arms use the same mask.
"""
    if kind not in KINDS or len(rna) != pairs.shape[1]:
        raise ValueError('Invalid contrastive arm or positive pairs')
    if len(rna) < 2:
        return rna.sum() * 0.
    item_coeff = (p['alpha_iu'], p['alpha_ii'])
    rna_coeff = (p['alpha_uu'], p['alpha_ui'])
    if kind == 'type_shared':
        item_coeff = (sum(item_coeff)/2,) * 2
        rna_coeff = (sum(rna_coeff)/2,) * 2
    allowed = (pairs[0, :, None] != pairs[0, None, :]) & (pairs[1, :, None] != pairs[1, None, :])
    loss = .5 * (directional_loss(rna, drug, item_coeff, p['temperature'], allowed) +
                 directional_loss(drug, rna, rna_coeff, p['temperature'], allowed.T))
    return loss * (0. if kind == 'bce' else 1.)


class NeighborTypeGraph(nn.Module):
    def __init__(self, dimensions, node_counts, pair_offset, p, kind):
        super().__init__()
        self.p, self.kind = dict(p), kind
        self.rna_count, self.drug_count = map(int, node_counts)
        self.register_buffer('pair_offset', torch.as_tensor(pair_offset, dtype=torch.long))
        width = p['width']
        self.project = nn.ModuleList([nn.Linear(d, width, bias=False) for d in dimensions])
        self.node_state = nn.Parameter(torch.empty(sum(node_counts), width))
        nn.init.xavier_uniform_(self.node_state)
        self.readout = nn.Sequential(nn.Linear(4*width, width), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def encode(self, xs, graph):
        attributes = [F.normalize(project(x / x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()),
                                  dim=-1, eps=1e-8)
                      for project, x in zip(self.project, xs)]
        initial = torch.cat(attributes) + self.node_state
        return source_propagation(initial, graph['adjacency'], self.rna_count, self.p['depth'])

    def local_pairs(self, pairs):
        local = pairs - self.pair_offset[:, None]
        if ((local[0] < 0) | (local[0] >= self.rna_count) |
                (local[1] < 0) | (local[1] >= self.drug_count)).any():
            raise ValueError('Pair outside target endpoint dictionaries')
        return local

    def decode(self, states, local):
        whole = states.sum(1)
        left, right = whole[local[0]], whole[self.rna_count + local[1]]
        return self.readout(torch.cat((left, right, left*right, (left-right).abs()), -1)).squeeze(-1)

    def forward(self, xs, graph, pairs):
        return self.decode(self.encode(xs, graph), self.local_pairs(pairs))

    def training_output(self, xs, graph, pairs, labels):
        states, local = self.encode(xs, graph), self.local_pairs(pairs)
        positive = local[:, labels == 1]
        rna, drug = states[positive[0]], states[self.rna_count + positive[1]]
        auxiliary = contrastive_loss(rna, drug, positive, self.kind, self.p)
        return self.decode(states, local), auxiliary, dict(positive_pairs=int(positive.shape[1]))


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered contrastive arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity.identity_task(a, p):
        return identity.IdentityResidual().to(device), [], [None]*p['folds'], folds
    rna_count, drug_count = int(a['node_counts'][1]), int(a['node_counts'][2])
    n = rna_count + drug_count
    graphs = []
    for f in range(p['folds']):
        keep = (a['folds'] != f) & (a['y_train'] == 1)
        support = a['pairs_train'][:, keep]
        if np.unique(support[0]*drug_count + support[1]).size != support.shape[1]:
            raise ValueError('Duplicate target support fact')
        edges = support.copy()
        edges[1] += rna_count
        arcs = np.concatenate((edges, edges[::-1]), axis=1)
        degree = np.bincount(arcs[0], minlength=n).astype(np.float32)
        weights = 1. / np.sqrt(degree[arcs[0]] * degree[arcs[1]])
        adjacency = torch.sparse_coo_tensor(torch.as_tensor(arcs, device=device),
            torch.as_tensor(weights, device=device), (n, n)).coalesce()
        graphs.append(dict(adjacency=adjacency,
            global_edges=torch.as_tensor(support + a['pair_offset'][:, None], device=device)))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in (1, 2)]
    model = NeighborTypeGraph([x.shape[1] for x in xs], (rna_count, drug_count),
                              a['pair_offset'], p, kind).to(device)
    return model, xs, graphs, folds
