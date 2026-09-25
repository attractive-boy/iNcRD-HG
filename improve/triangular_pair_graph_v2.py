"""Equivalent compiled contraction for the v1 dense triangular pair model.

The intermediate-index reduction uses contiguous K axes and fused products.
There is no sampling, low-rank attention approximation, or change of graph.
CPU evaluation uses the previously checked direct einsum reference.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from improve import triangular_pair_graph_v1 as reference

KINDS = reference.KINDS
context = reference.context
target_types = reference.target_types


def contiguous_triangle(q, k, v1, v2, kind):
    scores = torch.einsum('ikhd,kjhd->ijhk', q, k).contiguous() / math.sqrt(q.shape[-1])
    weights = scores.softmax(-1)
    left = v1.permute(0, 2, 3, 1).contiguous()[:, None]
    right = v2.permute(1, 2, 3, 0).contiguous()[None]
    if kind == 'full':
        return (weights[:, :, :, None, :] * left * right).sum(-1)
    if kind == 'additive_value':
        return (weights[:, :, :, None, :] * (.5 * left + .5 * right)).sum(-1)
    raise ValueError('Unknown triangular attention control')


compiled_triangle = torch.compile(contiguous_triangle, dynamic=True, fullgraph=True)


def triangle_block(q, k, v1, v2, kind):
    if q.device.type == 'cpu':
        return reference.triangle_block(q, k, v1, v2, kind)
    if q.device != torch.device('cuda:1'):
        raise ValueError('Compiled graph mechanism is restricted to cuda:1')
    return compiled_triangle(q, k, v1, v2, kind)


class TriangleLayer(reference.TriangleLayer):
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


def parameters():
    return reference.parameters()


def setup(a, p, kind, device):
    graphs, folds = [], []
    for fold in range(p['folds']):
        omitted = reference.np.flatnonzero(a['folds'] == fold)
        graphs.append(context(a, omitted, device))
        folds.append(torch.as_tensor(omitted, device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in target_types(a)]
    model = TriangularPairGraph([x.shape[1] for x in xs], int(a['relation'].max()) + 1, p, kind).to(device)
    return model, xs, graphs, folds
