"""Typed directed-edge cavity propagation with observed-label support.

Based on NBA-GNN (TMLR 2024, arXiv:2310.07430v2). The local adaptation
includes sampled zero-labeled training pairs as a distinct observation type.
They are not asserted to be experimentally verified negative associations.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve.overlap_block_models import masked_physical

KINDS = ('full', 'backtracking', 'positive_only', 'edge_mlp')


def support_graph(a, query_indices, kind, device):
    if kind not in KINDS:
        raise ValueError('Unknown cavity control')
    physical, relation = masked_physical(a, query_indices)
    nr = int(a['relation'].max() + 1)
    n = int(a['node_counts'].sum())
    keep = a['y_train'] == 0
    keep = keep.copy()
    keep[np.asarray(query_indices)] = False
    zeros = a['pairs_train'][:, keep] + a['pair_offset'][:, None]
    all_edges = np.concatenate((physical, zeros), axis=1)
    all_types = np.concatenate((relation, np.full(zeros.shape[1], nr, dtype=np.int64)))
    # Identical observation-degree inputs are retained in positive_only. That
    # control removes zero-labeled propagation, not its marginal information.
    counts = np.zeros((n, nr + 1), dtype=np.float32)
    np.add.at(counts, (all_edges[0], all_types), 1)
    np.add.at(counts, (all_edges[1], all_types), 1)
    if kind == 'positive_only':
        edges, types = physical, relation
    else:
        edges, types = all_edges, all_types
    e = edges.shape[1]
    arcs = np.concatenate((edges, edges[::-1]), axis=1)
    types = np.concatenate((types, types + nr + 1))
    reverse = np.concatenate((np.arange(e, 2 * e), np.arange(e)))
    if not np.array_equal(arcs[:, reverse], arcs[::-1]):
        raise AssertionError('Directed-edge inverse mapping failed')
    codes, group = np.unique(arcs[0] * n + arcs[1], return_inverse=True)
    reverse_group = np.searchsorted(codes, arcs[1] * n + arcs[0])
    parallel = (torch.as_tensor(group, device=device),
                torch.as_tensor(reverse_group, device=device),
                torch.as_tensor(np.bincount(group), device=device))
    return (torch.as_tensor(arcs, device=device), torch.as_tensor(types, device=device),
            torch.as_tensor(reverse, device=device), torch.as_tensor(np.log1p(counts), device=device), parallel)


def cavity_mean(messages, edges, reverse, n, backtracking=False, leaf_return=True, parallel=None):
    """Mean over w->u for u->v, excluding every typed inverse from v.

    Incidence sums minus the inverse edge equal the explicit directed line-graph
    operation. No dense edge-by-edge adjacency is materialized. At degree one,
    the registered model uses the paper's begrudging return instead of a reset.
    """
    src, dst = edges
    sums = messages.new_zeros((n, messages.shape[1])).index_add(0, dst, messages)
    degree = torch.bincount(dst, minlength=n).to(messages.dtype)[src, None]
    if backtracking:
        return sums[src] / degree.clamp_min(1)
    if parallel is None:
        codes, group = torch.unique(src * n + dst, sorted=True, return_inverse=True)
        reverse_group = torch.searchsorted(codes, dst * n + src)
        group_size = torch.bincount(group, minlength=len(codes))
    else:
        group, reverse_group, group_size = parallel
    grouped = messages.new_zeros((len(group_size), messages.shape[1])).index_add(0, group, messages)
    excluded = grouped[reverse_group]
    multiplicity = group_size[reverse_group, None].to(messages.dtype)
    value = (sums[src] - excluded) / (degree - multiplicity).clamp_min(1)
    if leaf_return:
        value = torch.where(degree <= multiplicity, excluded / multiplicity.clamp_min(1), value)
    return value


class CavityLayer(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.message = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU(),
                                    nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edges, reverse, n, mode, parallel):
        normalized = self.norm(h)
        m = self.message(normalized)
        neighbor = (torch.zeros_like(m) if mode == 'edge_mlp' else
                    cavity_mean(m, edges, reverse, n, backtracking=mode == 'backtracking', parallel=parallel))
        return h + self.dropout(self.update(torch.cat((normalized, neighbor), dim=-1)))


class CavityGraph(nn.Module):
    def __init__(self, dims, node_counts, relations, width=64, depth=4, dropout=.1, kind='full'):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown cavity control')
        self.node_counts, self.kind = list(node_counts), kind
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dims])
        self.input_norm = nn.LayerNorm(width)
        self.relation = nn.Embedding(2 * relations, width)
        self.initial = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
        self.layers = nn.ModuleList([CavityLayer(width, dropout) for _ in range(depth)])
        self.pool = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.LayerNorm(width))
        self.rna_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width // 2))
        self.drug_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width // 2))
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.scale = math.sqrt(width // 2)

    def encode(self, xs, graph):
        edges, relations, reverse, counts, parallel = graph
        pieces, offset = [], 0
        for x, project, n in zip(xs, self.project, self.node_counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            pieces.append(project(torch.cat((x / rms, counts[offset:offset + n]), dim=1)))
            offset += n
        nodes = F.gelu(self.input_norm(torch.cat(pieces)))
        src, dst = edges
        h = self.initial(torch.cat((nodes[src], nodes[dst], self.relation(relations)), dim=1))
        for layer in self.layers:
            h = layer(h, edges, reverse, len(nodes), self.kind, parallel)
        incoming = h.new_zeros((len(nodes), h.shape[1])).index_add(0, dst, h)
        outgoing = h.new_zeros((len(nodes), h.shape[1])).index_add(0, src, h)
        degree = torch.bincount(src, minlength=len(nodes)).to(h.dtype)[:, None].clamp_min(1)
        return self.pool(torch.cat((nodes, incoming / degree, outgoing / degree), dim=1)), counts

    def forward(self, xs, graph, pairs):
        h, counts = self.encode(xs, graph)
        # Compute node readouts before query indexing for exact query batching.
        rna, drug = self.rna_readout(h), self.drug_readout(h)
        br, bd = self.rna_degree(counts).squeeze(-1), self.drug_degree(counts).squeeze(-1)
        r, d = pairs
        return (rna[r] * drug[d]).sum(-1) / self.scale + br[r] + bd[d]
