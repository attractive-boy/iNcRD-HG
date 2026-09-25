"""Typed cooperative send/receive policies with query-independent MC inference.

Independent adaptation of Co-GNN (ICML 2024). The two binary actions follow the
factorization in the inspected author code; the typed trunk is shared locally.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'soft_actions', 'no_lookahead')


@dataclass
class Context:
    edges: torch.Tensor
    relation: torch.Tensor
    counts: torch.Tensor
    degree: torch.Tensor


def context(a, omitted, device):
    edges, relation = graph_lib.masked_physical(a, omitted)
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max()) + 1
    arcs, labels, counts = graph_lib.graph_inputs(edges, relation,
        np.empty((2, 0), dtype=np.int64), n, nr, device)
    # A separate self-state branch is always present. Count features retain the
    # frozen historical convention even though message self arcs are removed.
    mask = arcs[0] != arcs[1]
    arcs, labels = arcs[:, mask], labels[mask]
    degree = torch.bincount(arcs[1], minlength=n).to(counts.dtype).clamp_min(1)
    return Context(arcs, labels, counts, degree)


def weighted_mean(messages, destination, degree, weights=None):
    """Divide by original physical indegree, including zero-weight messages."""
    if weights is not None:
        messages = messages * weights[:, None]
    total = messages.new_zeros((len(degree), messages.shape[1]))
    total.index_add_(0, destination, messages)
    return total / degree[:, None]


def actions(logits, noise, temperature, hard):
    if logits.shape != noise.shape or logits.shape[-2:] != (2, 2):
        raise ValueError('Expected two binary actions for every node')
    if temperature <= 0:
        raise ValueError('Gumbel temperature must be positive')
    soft = ((logits + noise) / temperature).softmax(-1)
    if not hard:
        return soft
    discrete = F.one_hot(soft.argmax(-1), num_classes=2).to(soft.dtype)
    return discrete - soft.detach() + soft


def edge_weights(probabilities, edges):
    # Action 0 is receive, action 1 is send; category 0 means enabled.
    receive, send = probabilities[:, 0, 0], probabilities[:, 1, 0]
    return send[edges[0]] * receive[edges[1]]


class TypedMean(nn.Module):
    def __init__(self, width, relations):
        super().__init__()
        self.message = nn.Linear(width, width, bias=False)
        self.relation = nn.Embedding(relations, width)
        nn.init.normal_(self.relation.weight, std=.02)

    def forward(self, h, graph, weights=None, self_source=False):
        source = graph.edges[1] if self_source else graph.edges[0]
        values = self.message(h[source] + self.relation(graph.relation))
        return weighted_mean(values, graph.edges[1], graph.degree, weights)


class Policy(nn.Module):
    def __init__(self, width, relations):
        super().__init__()
        self.neighbor = TypedMean(width, relations)
        self.trunk = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 4))

    def forward(self, h, graph, lookahead=True):
        neighbor = self.neighbor(h, graph, self_source=not lookahead)
        return self.trunk(torch.cat((h, neighbor), dim=-1)).reshape(len(h), 2, 2)


class Environment(nn.Module):
    def __init__(self, width, relations, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.neighbor = TypedMean(width, relations)
        self.update = nn.Linear(2 * width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, graph, weights):
        z = self.norm(h)
        neighbor = self.neighbor(z, graph, weights=weights)
        update = F.gelu(self.update(torch.cat((z, neighbor), dim=-1)))
        return h + self.dropout(update)


class CooperativeGraph(nn.Module):
    def __init__(self, dimensions, relations, nodes, p, kind):
        super().__init__()
        self.kind, self.temperature = kind, p['temperature']
        width, depth = p['width'], p['depth']
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dimensions])
        self.input_norm = nn.LayerNorm(width)
        self.policy_norm = nn.LayerNorm(width)
        self.policy = Policy(width, 2 * relations + 2)
        self.layers = nn.ModuleList([Environment(width, 2 * relations + 2, p['dropout'])
                                    for _ in range(depth)])
        self.readout = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
                                    nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        generator = torch.Generator(device='cpu').manual_seed(p['evaluation_seed'])
        uniform = torch.rand((p['evaluation_samples'], depth, nodes, 2, 2), generator=generator)
        noise = -torch.log(-torch.log(uniform.clamp(1e-6, 1 - 1e-6)))
        self.register_buffer('evaluation_noise', noise)

    def initial(self, xs, graph):
        parts, start = [], 0
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(project(torch.cat((x / rms, graph.counts[start:start + len(x)]), dim=-1)))
            start += len(x)
        return F.gelu(self.input_norm(torch.cat(parts)))

    def one_draw(self, initial, graph, pairs, sample=None):
        h = initial
        for i, layer in enumerate(self.layers):
            logits = self.policy(self.policy_norm(h), graph, lookahead=self.kind != 'no_lookahead')
            if sample is None:
                uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
                noise = -torch.log(-torch.log(uniform))
            else:
                noise = self.evaluation_noise[sample, i]
            decisions = actions(logits, noise, self.temperature, hard=self.kind != 'soft_actions')
            h = layer(h, graph, edge_weights(decisions, graph.edges))
        left, right = h[pairs[0]], h[pairs[1]]
        return self.readout(torch.cat((left, right, left * right, (left - right).abs()), dim=-1)).squeeze(-1)

    def forward(self, xs, graph, pairs):
        h = self.initial(xs, graph)
        if self.training:
            return self.one_draw(h, graph, pairs)
        return torch.stack([self.one_draw(h, graph, pairs, sample).double()
                            for sample in range(len(self.evaluation_noise))]).mean(0)


def parameters():
    return dict(width=64, depth=4, dropout=.1, folds=5, temperature=1.,
                evaluation_samples=8, evaluation_seed=290916,
                lnc_policy='identity_strong_reference')


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered cooperative arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None] * p['folds'], folds
    graphs = [context(a, np.flatnonzero(a['folds'] == f), device) for f in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = CooperativeGraph([x.shape[1] for x in xs], int(a['relation'].max()) + 1,
                             int(a['node_counts'].sum()), p, kind).to(device)
    return model, xs, graphs, folds
