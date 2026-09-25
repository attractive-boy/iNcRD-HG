"""Exact query readout of triangular pair attention without unused final states.

All intermediate entities and every required pair token are retained. The final
layer evaluates only the requested output pairs. This equals the dense final
layer in evaluation and has the same training dropout law for distinct queries.
The prospective resource-bounded recipe uses one layer and width 32.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from improve import triangular_pair_graph_v1 as reference

KINDS = reference.KINDS
context = reference.context
target_types = reference.target_types


def query_triangle(q, k, v1, v2, source, destination, kind):
    left_q, right_k = q[source], k[:, destination].transpose(0, 1)
    left_v, right_v = v1[source], v2[:, destination].transpose(0, 1)
    scores = (left_q * right_k).sum(-1) / math.sqrt(q.shape[-1])
    weights = scores.softmax(1)
    if kind == 'full':
        values = left_v * right_v
    elif kind == 'additive_value':
        values = .5 * (left_v + right_v)
    else:
        raise ValueError('Unknown triangular attention control')
    return (weights[..., None] * values).sum(1)


class TriangleLayer(reference.TriangleLayer):
    def forward_queries(self, state, pairs, query_chunk):
        n = len(state)
        normalized = self.norm1(state)
        q, k, v1, v2 = [layer(normalized).reshape(n, n, self.heads, self.width // self.heads)
                        for layer in self.projections]
        blocks = []
        for start in range(0, pairs.shape[1], query_chunk):
            source, destination = pairs[:, start:start + query_chunk]
            arguments = (q, k, v1, v2, source, destination, self.kind)
            if self.training and torch.is_grad_enabled():
                value = checkpoint(query_triangle, *arguments, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                value = query_triangle(*arguments)
            blocks.append(value.flatten(-2))
        hidden = state[pairs[0], pairs[1]] + self.dropout(self.output(torch.cat(blocks)))
        return hidden + self.dropout(self.ffn(self.norm2(hidden)))


class TriangularPairGraph(reference.TriangularPairGraph):
    def __init__(self, dimensions, relations, parameters, kind):
        super().__init__(dimensions, relations, parameters, kind)
        replacements = []
        for layer in self.layers:
            replacement = TriangleLayer(parameters['width'], parameters['heads'], parameters['row_chunk'],
                                        parameters['dropout'], kind)
            replacement.load_state_dict(layer.state_dict())
            replacements.append(replacement)
        self.layers = nn.ModuleList(replacements)

    def initial_state(self, xs, graph):
        nodes = []
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            nodes.append(project(x / rms))
        h = torch.cat(nodes)
        return (self.left(h)[:, None, :] + self.right(h)[None, :, :] +
                self.relation(graph.physical_relation) +
                graph.observed_zero[:, :, None] * self.zero_token +
                graph.diagonal[:, :, None] * self.diagonal_token)

    def forward(self, xs, graph, pairs):
        compact = graph.global_to_local[pairs]
        if (compact < 0).any() or not self.layers:
            raise ValueError('Expected valid query pairs and at least one pair layer')
        if compact.shape[1] == 0:
            return self.zero_token.new_empty(0)
        state = self.initial_state(xs, graph)
        for layer in self.layers[:-1]:
            state = layer(state)
        queried = self.layers[-1].forward_queries(state, compact, self.config['query_chunk'])
        return self.readout(queried).squeeze(-1)


def parameters():
    return dict(width=32, depth=1, heads=4, row_chunk=8, query_chunk=128, dropout=.1, folds=5)


def setup(a, p, kind, device):
    graphs, folds = [], []
    for fold in range(p['folds']):
        omitted = reference.np.flatnonzero(a['folds'] == fold)
        graphs.append(context(a, omitted, device))
        folds.append(torch.as_tensor(omitted, device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in target_types(a)]
    model = TriangularPairGraph([x.shape[1] for x in xs], int(a['relation'].max()) + 1, p, kind).to(device)
    return model, xs, graphs, folds
