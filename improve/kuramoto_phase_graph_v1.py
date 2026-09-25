"""Attribute-initialized KoPE with sparse typed coupling and phase rotations.

Rotation, phase mixing and projected phase updates follow Microsoft's
Neuro-inspired_Phase_Encoding at commit
0eade3cebc76978d2944f3ebb9f50f57f1b7ee53 (vit_kope.py).
The graph has no image coordinates: phases are initialized from node states.
The source MIT license is retained in
research/oscillatory_next_20260916/kope_author_v1_LICENSE.

Copyright (c) Microsoft Corporation.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'static_phase', 'no_phase_rotation')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}


def parameters():
    return dict(width=64, heads=4, depth=4, dropout=.1, folds=5, gamma=.05,
        phase_initialization='learned bias-free projection of initial node state, shared across heads',
        phase_mixing='per-layer per-head identity plus Xavier normal gain0.1',
        mixed_squared_norm_floor=1e-5, phase_update_norm_floor=1e-5,
        phase_norm_tolerance=1e-5, coupling_qknorm=True, coupling_qk_scale=True,
        coupling_shared_across_layers=True, phase_update_order='after feature attention and feedforward',
        value_and_output_rotations=True, typed_edge_rotation='source phase, before key addition',
        phase_parameters='learned attribute initializer, head mixing, shared coupling projections and scales',
        lnc_policy='identity_strong_reference')


def empty_diagnostics():
    return dict(calls=0, attention_layers=0, phase_updates=0, phase_entries=0,
        feature_score_edge_heads=0, coupling_score_edge_heads=0, rotary_pairs=0,
        mixed_normalization_clamped=0, maximum_phase_norm_error=0.,
        maximum_abs_feature_score=0., maximum_abs_coupling_score=0., maximum_abs_state=0.)


def rotate_pairs(x, cosine, sine):
    """Real Cartesian rotation on adjacent coordinates of the last dimension."""
    if x.shape[-1] % 2 or cosine.shape[-1] != x.shape[-1]//2:
        raise ValueError('Rotary pair dimension mismatch')
    pair = x.reshape(*x.shape[:-1], -1, 2)
    real, imaginary = pair.unbind(-1)
    return torch.stack((real*cosine-imaginary*sine, real*sine+imaginary*cosine), dim=-1).reshape_as(x)


def mix_phase(cosine, sine, coefficient, floor):
    c = torch.einsum('nhc,hcd->nhd', cosine, coefficient)
    s = torch.einsum('nhc,hcd->nhd', sine, coefficient)
    squared = c.square()+s.square()
    inverse = squared.clamp_min(floor).rsqrt()
    return c*inverse, s*inverse, int((squared.detach() < floor).sum())


def phase_step(cosine, sine, mean_cosine, mean_sine, gamma, floor):
    phase = torch.stack((cosine, sine), dim=-1)
    delta = torch.stack((mean_cosine, mean_sine), dim=-1)
    tangent = delta - (delta*phase).sum(-1, keepdim=True)*phase
    updated = F.normalize(phase+gamma*tangent, dim=-1, eps=floor)
    return updated[..., 0], updated[..., 1]


class SharedPhaseCoupling(nn.Module):
    def __init__(self, width, heads, edge_types):
        super().__init__()
        self.heads, self.dim = heads, width//heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.qk_scale = nn.Parameter(torch.ones(heads, self.dim))
        self.edge = nn.Embedding(edge_types, width)
        nn.init.normal_(self.edge.weight, std=.02)

    def forward(self, h, cosine, sine, edges, relation, p):
        n = len(h)
        q = self.q_proj(h).reshape(n, self.heads, self.dim)
        k = self.k_proj(h).reshape(n, self.heads, self.dim)
        q = F.rms_norm(q, (self.dim,))*self.qk_scale
        k = F.rms_norm(k, (self.dim,))
        src, dst = edges
        edge = self.edge(relation).reshape(-1, self.heads, self.dim)
        score = (q[dst]*(k[src]+edge)).sum(-1)/math.sqrt(self.dim)
        if not torch.isfinite(score).all():
            raise FloatingPointError('Nonfinite Kuramoto coupling scores')
        weights = graph_lib.edge_softmax(score, dst, n)
        phase = torch.cat((cosine, sine), dim=-1)
        mean = phase.new_zeros(phase.shape).index_add(0, dst, weights[..., None]*phase[src])
        mean_c, mean_s = mean.chunk(2, dim=-1)
        c, s = phase_step(cosine, sine, mean_c, mean_s, p['gamma'], p['phase_update_norm_floor'])
        error = float((c.square()+s.square()-1.).detach().abs().max())
        if not torch.isfinite(c).all() or not torch.isfinite(s).all() or error > p['phase_norm_tolerance']:
            raise FloatingPointError('Kuramoto phase unit-norm gate failed')
        return c, s, error, float(score.detach().abs().max())


class PhaseAttention(nn.Module):
    def __init__(self, width, heads, edge_types, dropout, kind):
        super().__init__()
        self.heads, self.dim, self.kind = heads, width//heads, kind
        if self.dim % 4:
            raise ValueError('KoPE head dimension must be divisible by four')
        self.qkv = nn.Linear(width, 3*width, bias=False)
        self.edge = nn.Embedding(edge_types, width)
        nn.init.normal_(self.edge.weight, std=.02)
        self.output = nn.Linear(width, width)
        self.mix = nn.Parameter(torch.empty(heads, self.dim//2, self.dim//2))
        with torch.no_grad():
            nn.init.xavier_normal_(self.mix, gain=.1)
            self.mix.add_(torch.eye(self.dim//2).unsqueeze(0))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2*width, width))
        self.dropout = nn.Dropout(dropout)

    def attention(self, h, cosine, sine, edges, relation, p):
        n = len(h)
        q, k, v = self.qkv(h).reshape(n, 3, self.heads, self.dim).unbind(1)
        c, s, clamped = mix_phase(cosine, sine, self.mix, p['mixed_squared_norm_floor'])
        if self.kind == 'no_phase_rotation':
            c, s = torch.ones_like(c)+0.*c, 0.*s
        q, k, v = (rotate_pairs(x, c, s) for x in (q, k, v))
        src, dst = edges
        edge = self.edge(relation).reshape(-1, self.heads, self.dim)
        edge = rotate_pairs(edge, c[src], s[src])
        score = (q[dst]*(k[src]+edge)).sum(-1)/math.sqrt(self.dim)
        if not torch.isfinite(score).all():
            raise FloatingPointError('Nonfinite phase-rotated attention score')
        weight = graph_lib.edge_softmax(score, dst, n)
        mean = h.new_zeros((n, self.heads, self.dim)).index_add(0, dst, weight[..., None]*v[src])
        out = rotate_pairs(mean, c, -s)
        return self.output(out.flatten(1)), clamped, float(score.detach().abs().max())

    def forward(self, h, cosine, sine, edges, relation, p, diagnostics):
        out, clamped, maximum = self.attention(self.norm1(h), cosine, sine, edges, relation, p)
        h = h+self.dropout(out)
        h = h+self.dropout(self.ff(self.norm2(h)))
        if not torch.isfinite(h).all():
            raise FloatingPointError('Nonfinite phase-attention graph state')
        d = diagnostics
        d['attention_layers'] += 1
        d['feature_score_edge_heads'] += edges.shape[1]*self.heads
        d['rotary_pairs'] += (4*len(h)+edges.shape[1])*self.heads*(self.dim//2)
        d['mixed_normalization_clamped'] += clamped
        d['maximum_abs_feature_score'] = max(d['maximum_abs_feature_score'], maximum)
        d['maximum_abs_state'] = max(d['maximum_abs_state'], float(h.detach().abs().max()))
        return h


class KuramotoPhaseGraph(nn.Module):
    def __init__(self, dimensions, relations, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unregistered Kuramoto arm')
        self.p, self.kind = dict(p), kind
        width, heads = p['width'], p['heads']
        self.project = nn.ModuleList([nn.Linear(d+relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.phase_init = nn.Linear(width, width//heads//2, bias=False)
        self.coupling = SharedPhaseCoupling(width, heads, 2*relations+2)
        self.layers = nn.ModuleList([PhaseAttention(width, heads, 2*relations+2,
            p['dropout'], kind) for _ in range(p['depth'])])
        self.readout = nn.Sequential(nn.Linear(4*width, width), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        if kind != 'full':
            for value in self.coupling.parameters(): value.requires_grad_(False)
        if kind == 'no_phase_rotation':
            self.phase_init.weight.requires_grad_(False)
            for layer in self.layers: layer.mix.requires_grad_(False)
        self.phase_diagnostics = empty_diagnostics()

    def forward(self, xs, graph, pairs):
        edges, relation, counts = graph
        parts, start = [], 0
        for x, projection in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(projection(torch.cat((x/rms, counts[start:start+len(x)]), dim=-1)))
            start += len(x)
        h = F.gelu(self.norm(torch.cat(parts)))
        angle = self.phase_init(h)[:, None, :].expand(-1, self.p['heads'], -1)
        c, s = torch.cos(angle), torch.sin(angle)
        d = self.phase_diagnostics
        d['calls'] += 1
        for layer in self.layers:
            h = layer(h, c, s, edges, relation, self.p, d)
            new_c, new_s, error, maximum = self.coupling(h, c, s, edges, relation, self.p)
            d['phase_updates'] += 1
            d['phase_entries'] += c.numel()
            d['coupling_score_edge_heads'] += edges.shape[1]*self.p['heads']
            d['maximum_phase_norm_error'] = max(d['maximum_phase_norm_error'], error)
            d['maximum_abs_coupling_score'] = max(d['maximum_abs_coupling_score'], maximum)
            if self.kind == 'full':
                c, s = new_c, new_s
            else:
                c, s = c+0.*new_c, s+0.*new_s
        left, right = h[pairs[0]], h[pairs[1]]
        return self.readout(torch.cat((left, right, left*right, (left-right).abs()), dim=-1)).squeeze(-1)


def setup(a, p, kind, device):
    if set(a)-TRAIN_KEYS:
        raise ValueError('Kuramoto setup accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered Kuramoto arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        return IdentityResidual().to(device), [], [None]*p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max())+1
    graphs = []
    for f in range(p['folds']):
        edges, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == f))
        graphs.append(graph_lib.graph_inputs(edges, labels,
            np.empty((2, 0), dtype=np.int64), n, relations, device))
    xs = [torch.as_tensor(a['x_'+str(i)], device=device) for i in range(len(a['node_counts']))]
    model = KuramotoPhaseGraph([x.shape[1] for x in xs], relations, p, kind).to(device)
    return model, xs, graphs, folds
