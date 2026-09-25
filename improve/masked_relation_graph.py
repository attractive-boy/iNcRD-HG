"""Cross-relation masked structure learning, adapted from MaskGAE (KDD 2023).

Only non-target relation facts supply auxiliary targets. Their masked edges
and every parallel endpoint arc are removed before degrees and propagation.
The association query fold is independently removed first. Sampled absent
auxiliary pairs are reconstruction zeros, not measured biological negatives.
"""
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve.overlap_block_models import graph_inputs, SparseTypedAttention

ROOT = common.base.ROOT
OUT = ROOT / 'results/relation_mask_core_20260915'
KINDS = ('full', 'bce', 'shuffled_aux', 'visible_aux')


def auxiliary_zeros(edges, node_counts, relation_edges, count, rng):
    """Draw absent typed pairs without enumerating the full pair product."""
    n = int(np.sum(node_counts))
    borders = np.r_[0, np.cumsum(node_counts)]
    types = np.searchsorted(borders[1:], relation_edges, side='right')
    if len(np.unique(types[0])) != 1 or len(np.unique(types[1])) != 1:
        raise ValueError('A relation spans multiple endpoint type pairs')
    st, dt = int(types[0, 0]), int(types[1, 0])
    # All physical facts between these endpoint types are excluded, including
    # a differently typed parallel relation with the same endpoint pair.
    same_type = st == dt
    known = np.concatenate((edges[0] * n + edges[1], edges[1] * n + edges[0]))
    selected = np.empty(0, dtype=np.int64)
    for _ in range(100):
        r = rng.integers(borders[st], borders[st + 1], size=max(64, 4 * count))
        d = rng.integers(borders[dt], borders[dt + 1], size=len(r))
        if same_type:
            r, d = np.minimum(r, d), np.maximum(r, d)
        codes = r * n + d
        good = (r != d) & ~np.isin(codes, known)
        selected = np.unique(np.r_[selected, codes[good]])
        if len(selected) >= count:
            chosen = rng.permutation(selected)[:count]
            return np.stack((chosen // n, chosen % n))
    raise ValueError('Too few absent auxiliary pairs for the registered sampler')


def context(a, fold, kind, device, mask_fraction=.2):
    if kind not in KINDS:
        raise ValueError('Unknown relation-mask arm')
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max() + 1)
    query = a['pairs_train'][:, a['folds'] == fold]
    codes = query[0] * int(a['nd']) + query[1]
    main_keep = ~np.isin(a['target_codes'], codes)
    edges, relation, target = a['edges'][:, main_keep], a['relation'][main_keep], a['target_codes'][main_keep]
    rng = np.random.default_rng(20260915 + 1009 * fold)
    chosen, positives, negatives, types = [], [], [], []
    for rel in range(nr):
        available = np.flatnonzero((relation == rel) & (target < 0))
        if not len(available):
            continue
        size = max(1, int(round(mask_fraction * len(available))))
        ids = rng.permutation(available)[:size]
        pairs = edges[:, ids]
        zeros = auxiliary_zeros(edges, a['node_counts'], edges[:, available], len(ids), rng)
        chosen.extend(ids.tolist())
        positives.append(pairs)
        negatives.append(zeros)
        types.append(np.full(size, rel, dtype=np.int64))
    if not positives:
        raise ValueError('The graph has no auxiliary relation')
    positives, negatives = np.concatenate(positives, 1), np.concatenate(negatives, 1)
    auxiliary_relation = np.concatenate(types)
    removed = np.r_[positives[0] * n + positives[1], positives[1] * n + positives[0]]
    visible = np.ones(edges.shape[1], dtype=bool) if kind == 'visible_aux' else ~np.isin(edges[0] * n + edges[1], removed)
    if kind != 'visible_aux' and np.isin(edges[0, visible] * n + edges[1, visible], removed).any():
        raise AssertionError('Auxiliary fact remains in visible support')
    degree = np.zeros((n, nr), dtype=np.float32)
    np.add.at(degree, (positives[0], auxiliary_relation), 1)
    np.add.at(degree, (positives[1], auxiliary_relation), 1)
    if kind == 'shuffled_aux':
        # Preserve the marginal source and destination degrees per relation.
        # The reconstruction targets become an artificial structure null.
        shuffled = positives.copy()
        for rel in np.unique(auxiliary_relation):
            ix = np.flatnonzero(auxiliary_relation == rel)
            shuffled[1, ix] = positives[1, rng.permutation(ix)]
        positives = shuffled
    graph = graph_inputs(edges[:, visible], relation[visible], np.empty((2, 0), dtype=np.int64), n, nr, device)
    targets = tuple((torch.as_tensor(positives[:, auxiliary_relation == rel], device=device),
                     torch.as_tensor(negatives[:, auxiliary_relation == rel], device=device), int(rel))
                    for rel in np.unique(auxiliary_relation))
    return (*graph, targets, torch.as_tensor(np.log1p(degree), device=device))


class MaskedRelationGraph(nn.Module):
    def __init__(self, dims, node_counts, relations, kind, width=64, depth=3, dropout=.1):
        super().__init__()
        self.node_counts, self.kind, self.scale = list(node_counts), kind, math.sqrt(width // 2)
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dims])
        self.input_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([SparseTypedAttention(width, 4, 2 * relations + 2, dropout) for _ in range(depth)])
        self.rna_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width // 2))
        self.drug_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width // 2))
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.aux_project = nn.Linear(width, width // 2)
        self.aux_relation = nn.Embedding(relations, width // 2)
        self.aux_bias = nn.Embedding(relations, 1)
        self.aux_degree = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, relations))
        nn.init.normal_(self.aux_relation.weight, std=.1)
        nn.init.zeros_(self.aux_bias.weight)
        self._cached_h = None

    def encode(self, xs, graph):
        edges, relation, counts = graph[:3]
        pieces, offset = [], 0
        for x, project, size in zip(xs, self.project, self.node_counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            pieces.append(project(torch.cat((x / rms, counts[offset:offset + size]), 1)))
            offset += size
        h = F.gelu(self.input_norm(torch.cat(pieces)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        return h

    def forward(self, xs, graph, pairs):
        h = self.encode(xs, graph)
        if self.training:
            self._cached_h = h
        counts = graph[2]
        rna, drug = self.rna_readout(h), self.drug_readout(h)
        br, bd = self.rna_degree(counts).squeeze(-1), self.drug_degree(counts).squeeze(-1)
        r, d = pairs
        return (rna[r] * drug[d]).sum(-1) / self.scale + br[r] + bd[d]

    def auxiliary(self, xs, graph, generator):
        if self._cached_h is None:
            raise RuntimeError('Auxiliary loss requires the same-step main forward')
        h, self._cached_h = self._cached_h, None
        if self.kind == 'bce':
            zero = h.sum() * 0.
            return zero, dict(edge=zero.detach(), degree=zero.detach())
        z = self.aux_project(h)
        losses = []
        for positive, negative, rel in graph[3]:
            size = min(128, positive.shape[1], negative.shape[1])
            pi = torch.randperm(positive.shape[1], device=h.device, generator=generator)[:size]
            ni = torch.randperm(negative.shape[1], device=h.device, generator=generator)[:size]
            pairs = torch.cat((positive[:, pi], negative[:, ni]), 1)
            target = torch.cat((h.new_ones(size), h.new_zeros(size)))
            score = (z[pairs[0]] * z[pairs[1]] * self.aux_relation.weight[rel]).sum(-1) / self.scale + self.aux_bias.weight[rel, 0]
            losses.append(F.binary_cross_entropy_with_logits(score, target))
        edge = torch.stack(losses).mean()
        degree = F.mse_loss(self.aux_degree(h), graph[4])
        return edge + .01 * degree, dict(edge=edge.detach(), degree=degree.detach())


def setup(a, p, kind, device):
    graphs, folds = [], []
    for fold in range(p['folds']):
        graphs.append(context(a, fold, kind, device, p['mask_fraction']))
        folds.append(torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = MaskedRelationGraph([x.shape[1] for x in xs], a['node_counts'], int(a['relation'].max() + 1),
        kind, p['width'], p['depth'], p['dropout']).to(device)
    return model, xs, graphs, folds


def parameters():
    return dict(kinds=list(KINDS), width=64, depth=3, dropout=.1, folds=5,
                mask_fraction=.2, auxiliary_weight=.1, primary_kind='full',
                primary_controls=['bce', 'shuffled_aux'])


def prepare():
    if OUT.exists():
        raise ValueError('Core relation-mask specification already exists')
    parent = common.protocol(common.OUT)
    sources = ('masked_relation_graph.py', 'test_masked_relation_graph.py',
               'overlap_block_models.py', 'overlap_block_experiment.py', 'non_tabular_experiment.py')
    p = dict(**parameters(), created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=parent['tasks'], selection=parent['selection'], specification_only=True,
        controls='full and bce see identical masked graphs and initialized modules; bce has no auxiliary loss. shuffled_aux preserves auxiliary endpoint degrees but permutes positive destinations within each relation. visible_aux reconstructs visible facts as a diagnostic; main query facts remain absent in every arm.',
        method='masked_relation_core',
        literature=dict(title="What's Behind the Mask: Understanding Masked Graph Modeling for Graph Autoencoders",
                        doi='10.1145/3580305.3599546', arxiv='2205.10053v2'),
        adaptation='Joint supervised cross-relation regularization around fixed support likelihood, not pretraining. Only non-target relation edges are auxiliary targets. Fixed 20% edge masks in five supports, typed uniform absent-pair samples, relation-balanced DistMult decoders, log-degree reconstruction weighted 0.01. No path masking, no hierarchical pooling, no transfer of paper benchmark numbers.',
        new_innovation_family=False, family='graph_masked_self_supervision',
        source_sha256={n: common.base.digest(ROOT / n) for n in sources},
        inputs={str(ROOT / 'research/new_mechanisms_20260915/maskgae_paper.pdf'):
                common.base.digest(ROOT / 'research/new_mechanisms_20260915/maskgae_paper.pdf')})
    OUT.mkdir(parents=True)
    (OUT / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(ROOT / name, OUT / 'source_snapshot' / name)
    common.base.write_json(OUT / 'protocol.json', p)
