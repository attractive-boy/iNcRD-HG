"""Typed masked feature reconstruction with an EMA latent target.

Independent GraphMAE2-inspired adaptation for query-masked bipartite learning.
Unlike GraphMAE2's pretrain/evaluate pipeline, this module regularizes supervised
link training jointly. It reconstructs fixed node attributes, not query labels.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from improve.overlap_block_models import SparseTypedAttention


def cosine_error(prediction, target, exponent=2):
    valid = target.square().sum(-1) > 1e-12
    if not valid.any():
        return prediction.sum() * 0.
    a = F.normalize(prediction[valid], dim=-1, eps=1e-8)
    b = F.normalize(target[valid], dim=-1, eps=1e-8)
    return (1 - (a * b).sum(-1)).clamp_min(0).pow(exponent).mean()


def node_mask(node_counts, fraction, device, generator):
    if not 0 < fraction < 1:
        raise ValueError('Mask fraction must be strictly between zero and one')
    mask = torch.zeros(sum(node_counts), dtype=torch.bool, device=device)
    offset = 0
    for n in node_counts:
        # Each node type participates, including the much smaller drug type.
        count = max(1, int(n * fraction))
        indices = torch.randperm(n, device=device, generator=generator)[:count]
        mask[indices + offset] = True
        offset += n
    return mask


class TypedEncoder(nn.Module):
    def __init__(self, dims, counts, relations, width, depth, heads, dropout):
        super().__init__()
        self.counts = list(counts)
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dims])
        self.mask_tokens = nn.ParameterList([nn.Parameter(torch.zeros(d + relations)) for d in dims])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([SparseTypedAttention(width, heads, 2 * relations + 2, dropout)
                                     for _ in range(depth)])

    def forward(self, xs, graph, masked=None):
        edges, relation, counts = graph
        pieces, offset = [], 0
        for x, projection, token, n in zip(xs, self.project, self.mask_tokens, self.counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            z = torch.cat((x / rms, counts[offset:offset + n]), dim=1)
            if masked is not None:
                z = torch.where(masked[offset:offset + n, None], token, z)
            pieces.append(projection(z))
            offset += n
        h = F.gelu(self.norm(torch.cat(pieces)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        return h


class MaskedGraphTeacher(nn.Module):
    KINDS = ('full', 'no_teacher', 'fixed_remask', 'bce')

    def __init__(self, dims, node_counts, relations, width=64, depth=3, heads=4,
                 dropout=.1, kind='full', mask_fraction=.5, remask_fraction=.5,
                 remask_views=2, momentum=.99):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError('Unknown masked teacher control')
        self.kind, self.node_counts = kind, list(node_counts)
        self.mask_fraction, self.remask_fraction = mask_fraction, remask_fraction
        self.remask_views, self.momentum = remask_views, momentum
        self.encoder = TypedEncoder(dims, node_counts, relations, width, depth, heads, dropout)
        self.projector = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.teacher_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.teacher_projector = copy.deepcopy(self.projector).requires_grad_(False)
        self.to_decoder = nn.Linear(width, width, bias=False)
        self.decoder_mask = nn.Parameter(torch.zeros(width))
        self.decoder = SparseTypedAttention(width, heads, 2 * relations + 2, dropout)
        self.reconstruction = nn.ModuleList([nn.Linear(width, d) for d in dims])
        self.rna_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width // 2))
        self.drug_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width // 2))
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.scale = math.sqrt(width // 2)
        self.teacher_encoder.eval()
        self.teacher_projector.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_projector.eval()
        return self

    @torch.no_grad()
    def update_teacher(self):
        for target, source in ((self.teacher_encoder, self.encoder),
                               (self.teacher_projector, self.projector)):
            for teacher, student in zip(target.parameters(), source.parameters()):
                teacher.mul_(self.momentum).add_(student, alpha=1 - self.momentum)
            for teacher, student in zip(target.buffers(), source.buffers()):
                teacher.copy_(student)

    def forward(self, xs, graph, pairs):
        h = self.encoder(xs, graph)
        rna, drug = pairs
        counts = graph[2]
        return (self.rna_readout(h[rna]) * self.drug_readout(h[drug])).sum(-1) / self.scale + \
            self.rna_degree(counts[rna]).squeeze(-1) + self.drug_degree(counts[drug]).squeeze(-1)

    def auxiliary(self, xs, graph, generator):
        if self.kind == 'bce':
            zero = self.decoder_mask.sum() * 0.
            return zero, dict(reconstruction=zero.detach(), latent=zero.detach())
        masked = node_mask(self.node_counts, self.mask_fraction, xs[0].device, generator)
        h = self.encoder(xs, graph, masked)
        codes = self.to_decoder(h)
        feature_losses = []
        for _ in range(self.remask_views):
            remasked = (masked if self.kind == 'fixed_remask' else
                        node_mask(self.node_counts, self.remask_fraction, h.device, generator))
            damaged = torch.where(remasked[:, None], self.decoder_mask, codes)
            decoded = self.decoder(damaged, graph[0], graph[1])
            offset, typed_losses = 0, []
            for x, projection, n in zip(xs, self.reconstruction, self.node_counts):
                keep = masked[offset:offset + n]
                predicted = projection(decoded[offset:offset + n][keep])
                typed_losses.append(cosine_error(predicted, x[keep].detach()))
                offset += n
            feature_losses.append(torch.stack(typed_losses).mean())
        reconstruction = torch.stack(feature_losses).mean()
        if self.kind == 'no_teacher':
            latent = h.sum() * 0.
        else:
            with torch.no_grad():
                target = self.teacher_projector(self.teacher_encoder(xs, graph))
            # The paper's Eq. (4) includes all nodes, masked and unmasked.
            latent = cosine_error(self.projector(h), target)
        return reconstruction + latent, dict(reconstruction=reconstruction.detach(), latent=latent.detach())
