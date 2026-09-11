"""Learned three-pair messages for bipartite association completion.

The fourth (queried) pair is excluded from each elementary rectangle message.
These are latent relation features, not confirmed biological mechanisms. A
sampled zero label is distinct from an unlabelled pair and from sensitivity.
"""
import torch
from torch import nn

from improve.support_operator_modules import exclude_support_pairs


def rectangle_message(a, b, c, mode='product'):
    """Average over s != r, e != d of a[r,e] * b[s,e] * c[s,d].

All arrays are [RNA, drug, channel]. Two batched matrix multiplications and
inclusion/exclusion compute the product without materialising rectangles.
The additive control observes the same three positions, but averages their
sum instead of their product. Both operators exclude immediate backtracking.
"""
    if a.ndim != 3 or a.shape != b.shape or a.shape != c.shape:
        raise ValueError('Expected three identically shaped rank-three fields')
    if mode not in ('product', 'additive'):
        raise ValueError(mode)
    nr, nd, _ = a.shape
    if nr < 2 or nd < 2:
        return (a + b + c) * 0
    count = (nr - 1) * (nd - 1)
    if mode == 'additive':
        row = (a.sum(1, keepdim=True) - a) / (nd - 1)
        col = (c.sum(0, keepdim=True) - c) / (nr - 1)
        middle = (b.sum((0, 1), keepdim=True) - b.sum(1, keepdim=True)
                  - b.sum(0, keepdim=True) + b) / count
        return (row + middle + col) / 3
    ac, bc, cc = (v.permute(2, 0, 1) for v in (a, b, c))
    # Choose the smaller intermediate square matrix.
    if nd <= nr:
        full = torch.bmm(ac, torch.bmm(bc.transpose(1, 2), cc))
    else:
        full = torch.bmm(torch.bmm(ac, bc.transpose(1, 2)), cc)
    full = full.permute(1, 2, 0)
    repeated_rna = (a * b).sum(1, keepdim=True) * c
    repeated_drug = a * (b * c).sum(0, keepdim=True)
    return (full - repeated_rna - repeated_drug + a * b * c) / count


def evidence_grid(z, edges, rna_types, target_type, support_pairs,
                  support_labels, queries):
    """Two channels: known fitting positive and sampled fitting zero.

Other RNA tasks contribute their existing positive graph edges, following the
single-task protocol. Every current query is removed from the explicit grid,
including zero-labelled queries. The caller must also mask the GNN graph.
"""
    nd = len(z['drug'])
    offsets, offset = {}, 0
    for nt in rna_types:
        offsets[nt], offset = offset, offset + len(z[nt])
    grid = z['drug'].new_zeros((offset, nd, 2))
    tasks = {'lncRNA': 'LncDrug', 'miRNA': 'MiDrug'}
    for nt in rna_types:
        key = (nt, tasks[nt], 'drug')
        if key in edges:
            r, d = edges[key]
            grid[offsets[nt] + r, d, 0] = 1
    pairs, labels, _ = exclude_support_pairs(support_pairs, support_labels, queries, nd)
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('Only binary fitting labels are accepted')
    neg = pairs[:, labels == 0]
    grid[offsets[target_type] + neg[0], neg[1], 1] = 1
    grid[offsets[target_type] + queries[0], queries[1]] = 0
    if bool((grid.sum(-1) > 1).any()):
        raise ValueError('Conflicting fitting evidence')
    return grid, offsets


class RectangleLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.factors = nn.Linear(width, 3 * width)
        self.message_norm = nn.LayerNorm(width)
        self.update = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU(),
                                    nn.Linear(2 * width, width))
        self.step = nn.Parameter(torch.tensor(-2.197224577))  # sigmoid = .1

    def forward(self, h, mode, disable_message=False):
        a, b, c = self.factors(self.norm(h)).tanh().chunk(3, dim=-1)
        message = rectangle_message(a, b, c, mode)
        if disable_message:
            message = torch.zeros_like(message)
        update = self.update(torch.cat((h, self.message_norm(message)), dim=-1))
        return h + self.step.sigmoid() * update


class RectangleBranch(nn.Module):
    def __init__(self, hidden, rna_type, rna_types=('lncRNA', 'miRNA'),
                 width=16, depth=2, max_correction=1.):
        super().__init__()
        if depth < 1 or width < 2 or max_correction <= 0:
            raise ValueError('Invalid branch dimensions')
        self.rna_type, self.rna_types = rna_type, tuple(rna_types)
        self.max_correction = max_correction
        self.rna = nn.ModuleDict({nt: nn.Sequential(nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * width)) for nt in self.rna_types})
        self.drug = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * width))
        self.evidence = nn.Linear(2, width, bias=False)
        self.layers = nn.ModuleList([RectangleLayer(width) for _ in range(depth)])
        self.readout = nn.Sequential(nn.LayerNorm((depth + 1) * width),
            nn.Linear((depth + 1) * width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, z, edges, queries, support_pairs, support_labels,
                mode='product', ramp=1., intervention=None):
        if not 0 <= ramp <= 1:
            raise ValueError('Ramp must lie in [0,1]')
        if intervention not in (None, 'no_evidence', 'no_message', 'additive', 'no_branch'):
            raise ValueError(intervention)
        grid, offsets = evidence_grid(z, edges, self.rna_types, self.rna_type,
                                      support_pairs, support_labels, queries)
        if intervention == 'no_evidence':
            grid = torch.zeros_like(grid)
        dr, dm = self.drug(z['drug']).chunk(2, dim=-1)
        rows = []
        for nt in self.rna_types:
            rr, rm = self.rna[nt](z[nt]).chunk(2, dim=-1)
            rows.append(rr[:, None, :] + dr[None, :, :] + rm[:, None, :] * dm[None, :, :])
        h = (torch.cat(rows, dim=0) + self.evidence(grid)).tanh()
        r, d = offsets[self.rna_type] + queries[0], queries[1]
        selected = [h[r, d]]
        effective_mode = 'additive' if intervention == 'additive' else mode
        for layer in self.layers:
            h = layer(h, effective_mode, disable_message=intervention == 'no_message')
            selected.append(h[r, d])
        correction = ramp * self.max_correction * self.readout(torch.cat(selected, dim=-1)).flatten().tanh()
        return correction * 0 if intervention == 'no_branch' else correction


class RectangleModel(nn.Module):
    def __init__(self, base, hidden, rna_type, mode='rectangle', width=16,
                 depth=2, max_correction=1.):
        super().__init__()
        if mode not in ('pca_query', 'additive', 'rectangle'):
            raise ValueError(mode)
        self.base, self.mode = base, mode
        self.branch = (None if mode == 'pca_query' else RectangleBranch(
            hidden, rna_type, width=width, depth=depth, max_correction=max_correction))

    def forward(self, x, edges, queries, weights, support_pairs, support_labels,
                ramp=1., intervention=None, return_components=False):
        z = self.base.encoder(x, edges, weights)
        prior = self.base.decoder(z, queries)
        correction = torch.zeros_like(prior)
        if self.branch is not None:
            correction = self.branch(z, edges, queries, support_pairs, support_labels,
                mode='product' if self.mode == 'rectangle' else 'additive',
                ramp=ramp, intervention=intervention)
        if return_components:
            return prior + correction, prior, correction
        return prior + correction
