"""Dense pair-state triangular attention with exact row chunking.

Independent adaptation of Towards Principled Graph Transformers (NeurIPS 2024,
arXiv:2401.10119v4). Pair states cover the target RNA and drug dictionaries.
Every intermediate entity in these dictionaries is retained; row chunking and
activation checkpointing change memory use, not the triangular operator.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from improve import overlap_block_models as graph_lib

KINDS = ('full', 'additive_value')


def triangle_block(left_key, right_key, left_value, right_value, kind):
    """Contract through the SAME intermediate k, normalized over k.

    Inputs: [i,k,head,channel] and [k,j,head,channel]. Only i is chunked.
    The additive control preserves the scores and both value projections.
    """
    scores = torch.einsum('ikhd,kjhd->ikjh', left_key, right_key) / math.sqrt(left_key.shape[-1])
    weights = scores.softmax(dim=1)
    if kind == 'full':
        return torch.einsum('ikjh,ikhd,kjhd->ijhd', weights, left_value, right_value)
    if kind == 'additive_value':
        return .5 * (torch.einsum('ikjh,ikhd->ijhd', weights, left_value) +
                      torch.einsum('ikjh,kjhd->ijhd', weights, right_value))
    raise ValueError('Unknown triangular attention control')


class TriangleLayer(nn.Module):
    def __init__(self, width, heads, row_chunk, dropout, kind):
        super().__init__()
        if width % heads or row_chunk < 1 or kind not in KINDS:
            raise ValueError('Invalid triangular layer configuration')
        self.heads, self.width, self.row_chunk, self.kind = heads, width, row_chunk, kind
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.projections = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(4)])
        self.output = nn.Linear(width, width, bias=False)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, state):
        n = state.shape[0]
        normalized = self.norm1(state)
        q, k, v1, v2 = [layer(normalized).reshape(n, n, self.heads, self.width // self.heads)
                        for layer in self.projections]
        blocks = []
        for start in range(0, n, self.row_chunk):
            arguments = (q[start:start+self.row_chunk], k, v1[start:start+self.row_chunk], v2, self.kind)
            if self.training and torch.is_grad_enabled():
                value = checkpoint(triangle_block, *arguments, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                value = triangle_block(*arguments)
            blocks.append(value.flatten(-2))
        state = state + self.dropout(self.output(torch.cat(blocks)))
        return state + self.dropout(self.ffn(self.norm2(state)))


@dataclass
class PairContext:
    physical_relation: torch.Tensor
    observed_zero: torch.Tensor
    diagonal: torch.Tensor
    global_to_local: torch.Tensor
    omitted_indices: np.ndarray
    support_zero_indices: np.ndarray
    compact_global_nodes: np.ndarray


def target_types(a):
    offsets = np.r_[0, np.cumsum(a['node_counts'])[:-1]]
    return tuple(int(np.flatnonzero(offsets == offset)[0]) for offset in a['pair_offset'])


def context(a, omitted, device):
    omitted = np.asarray(omitted, dtype=np.int64)
    types = target_types(a)
    nodes = np.concatenate([np.arange(offset, offset + a['node_counts'][kind])
                            for offset, kind in zip(a['pair_offset'], types)])
    mapping = np.full(int(a['node_counts'].sum()), -1, dtype=np.int64)
    mapping[nodes] = np.arange(len(nodes))
    edges, relation = graph_lib.masked_physical(a, omitted)
    compact = mapping[edges]
    allowed = (compact >= 0).all(0)
    compact, relation = compact[:, allowed], relation[allowed]
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=np.int64)
    for (u, v), label in zip(compact.T, relation + 1):
        if matrix[u, v] not in (0, label) or matrix[v, u] not in (0, label):
            raise ValueError('Multiple physical relation types on the same compact pair')
        matrix[u, v] = matrix[v, u] = label
    fitting = np.ones(len(a['y_train']), dtype=bool); fitting[omitted] = False
    zero_ids = np.flatnonzero(fitting & (a['y_train'] == 0))
    zeros = np.zeros((n, n), dtype=bool)
    zero_pairs = mapping[a['pairs_train'][:, zero_ids] + a['pair_offset'][:, None]]
    if matrix[zero_pairs[0], zero_pairs[1]].any():
        raise ValueError('A recorded zero conflicts with a physical positive fact')
    zeros[zero_pairs[0], zero_pairs[1]] = True
    zeros[zero_pairs[1], zero_pairs[0]] = True
    return PairContext(torch.as_tensor(matrix, device=device), torch.as_tensor(zeros, device=device),
        torch.eye(n, dtype=torch.bool, device=device), torch.as_tensor(mapping, device=device),
        omitted.copy(), zero_ids, nodes)


class TriangularPairGraph(nn.Module):
    def __init__(self, dimensions, relations, parameters, kind):
        super().__init__()
        self.kind, self.config = kind, parameters
        width = parameters['width']
        self.project = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * width), nn.GELU(),
            nn.Linear(2 * width, width)) for d in dimensions])
        self.left, self.right = nn.Linear(width, width), nn.Linear(width, width)
        self.relation = nn.Embedding(relations + 1, width, padding_idx=0)
        self.zero_token = nn.Parameter(torch.randn(width) * .02)
        self.diagonal_token = nn.Parameter(torch.randn(width) * .02)
        self.layers = nn.ModuleList([TriangleLayer(width, parameters['heads'], parameters['row_chunk'],
            parameters['dropout'], kind) for _ in range(parameters['depth'])])
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                     nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def encode(self, xs, graph):
        nodes = []
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            nodes.append(project(x / rms))
        h = torch.cat(nodes)
        state = (self.left(h)[:, None, :] + self.right(h)[None, :, :] +
                 self.relation(graph.physical_relation) +
                 graph.observed_zero[:, :, None] * self.zero_token +
                 graph.diagonal[:, :, None] * self.diagonal_token)
        for layer in self.layers:
            state = layer(state)
        return state

    def forward(self, xs, graph, pairs):
        compact = graph.global_to_local[pairs]
        if (compact < 0).any():
            raise ValueError('Query is outside the target entity dictionaries')
        state = self.encode(xs, graph)
        return self.readout(state[compact[0], compact[1]]).squeeze(-1)


def parameters():
    return dict(width=16, depth=2, heads=2, row_chunk=8, dropout=.1, folds=5)


def setup(a, p, kind, device):
    graphs, folds = [], []
    for fold in range(p['folds']):
        omitted = np.flatnonzero(a['folds'] == fold)
        graphs.append(context(a, omitted, device))
        folds.append(torch.as_tensor(omitted, device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in target_types(a)]
    model = TriangularPairGraph([x.shape[1] for x in xs], int(a['relation'].max()) + 1, p, kind).to(device)
    return model, xs, graphs, folds
