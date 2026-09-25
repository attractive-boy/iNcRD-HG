"""Typed relation-star hypergraph operators inspired by ED-HNN (ICLR 2023).

The hyperedges are explicit modeling groups derived from support relations,
not experimentally observed higher-order biological interactions. A shared
nonlinear operator sends a distinct message to each member of a hyperedge.
The invariant-message control has the identical inputs and parameters.
"""
from datetime import datetime, timezone
import math
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve.overlap_block_models import masked_physical, graph_inputs

ROOT = common.base.ROOT
OUT = ROOT / 'results/equivariant_hypergraph_core_20260915'
KINDS = ('full', 'invariant', 'pairwise', 'no_hypergraph')


def incidence_arrays(edges, relation, node_counts, pairwise=False):
    n = int(np.sum(node_counts))
    types = np.repeat(np.arange(len(node_counts)), node_counts)
    members, groups, kinds, owners = [], [], [], []
    if pairwise:
        facts = np.unique(np.column_stack((relation, edges.T)), axis=0)
        for rel, src, dst in facts:
            if src == dst:
                continue
            members.extend((int(src), int(dst)))
            groups.extend((len(kinds), len(kinds)))
            kinds.append(int(rel * len(node_counts) + types[src]))
            owners.append(int(src))
    else:
        stars = {}
        for (src, dst), rel in zip(edges.T, relation):
            if src == dst:
                continue
            stars.setdefault((int(rel), int(src)), {int(src)}).add(int(dst))
            stars.setdefault((int(rel), int(dst)), {int(dst)}).add(int(src))
        for (rel, owner), nodes in sorted(stars.items()):
            members.extend(sorted(nodes))
            groups.extend([len(kinds)] * len(nodes))
            kinds.append(rel * len(node_counts) + int(types[owner]))
            owners.append(owner)
    members, groups = np.asarray(members, dtype=np.int64), np.asarray(groups, dtype=np.int64)
    return dict(members=members, groups=groups, kinds=np.asarray(kinds, dtype=np.int64),
                owners=np.asarray(owners, dtype=np.int64),
                sizes=np.bincount(groups, minlength=len(kinds)).astype(np.float32),
                degrees=np.bincount(members, minlength=n).astype(np.float32))


def context(a, fold, device, pairwise=False):
    edges, relations = masked_physical(a, np.flatnonzero(a['folds'] == fold))
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max() + 1)
    counts = graph_inputs(edges, relations, np.empty((2, 0), dtype=np.int64), n, nr, device)[2]
    parts = incidence_arrays(edges, relations, a['node_counts'], pairwise)
    return {**{k: torch.as_tensor(v, device=device) for k, v in parts.items()}, 'counts': counts}


def segment_sum(values, index, count):
    result = values.new_zeros((count, values.shape[-1]))
    return result.index_add_(0, index, values)


def member_messages(hidden, graph, phi, rho, type_values, invariant=False):
    node, edge = graph['members'], graph['groups']
    sizes = graph['sizes']
    pooled = segment_sum(phi(hidden)[node], edge, len(sizes)) / sizes[:, None].clamp_min(1).sqrt()
    pooled = pooled + type_values
    if invariant:
        receiver = segment_sum(hidden[node], edge, len(sizes)) / sizes[:, None].clamp_min(1)
        receiver = receiver[edge]
    else:
        receiver = hidden[node]
    messages = rho(torch.cat((receiver, pooled[edge], sizes[edge, None].log1p()), 1))
    return messages


class SharedHypergraphOperator(nn.Module):
    def __init__(self, width, hyperedge_types, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.phi = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.rho = nn.Sequential(nn.Linear(2 * width + 1, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.update = nn.Sequential(nn.Linear(3 * width + 1, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.type_value = nn.Embedding(hyperedge_types, width)
        nn.init.normal_(self.type_value.weight, std=.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, initial, graph, invariant=False):
        current = self.norm(hidden)
        messages = member_messages(current, graph, self.phi, self.rho,
                                   self.type_value(graph['kinds']), invariant)
        aggregate = segment_sum(messages, graph['members'], len(hidden))
        aggregate = aggregate / graph['degrees'][:, None].clamp_min(1)
        updated = self.update(torch.cat((current, aggregate, initial, graph['degrees'][:, None].log1p()), 1))
        return hidden + self.dropout(updated)


class EquivariantHypergraph(nn.Module):
    def __init__(self, dims, node_counts, relations, kind, width=64, depth=3, dropout=.1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown equivariant-hypergraph arm')
        self.kind, self.node_counts, self.depth = kind, list(map(int, node_counts)), depth
        self.project = nn.ModuleList([nn.Linear(d + relations, width) for d in dims])
        self.input_norm, self.output_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.operator = SharedHypergraphOperator(width, relations * len(node_counts), dropout)
        self.rna_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.drug_readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.scale = math.sqrt(width)

    def forward(self, xs, graph, pairs):
        counts = graph['counts']
        pieces, offset = [], 0
        for x, project, size in zip(xs, self.project, self.node_counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            pieces.append(project(torch.cat((x / rms, counts[offset:offset + size]), 1)))
            offset += size
        initial = F.gelu(self.input_norm(torch.cat(pieces)))
        hidden = initial
        if self.kind != 'no_hypergraph':
            for _ in range(self.depth):
                hidden = self.operator(hidden, initial, graph, self.kind == 'invariant')
        hidden = self.output_norm(hidden)
        rna, drug = self.rna_readout(hidden), self.drug_readout(hidden)
        r, d = pairs
        score = (rna[r] * drug[d]).sum(-1) / self.scale
        return score + self.rna_degree(counts[r]).squeeze(-1) + self.drug_degree(counts[d]).squeeze(-1)


def parameters():
    return dict(kinds=list(KINDS), width=64, depth=3, dropout=.1, folds=5,
                auxiliary_weight=0., primary_kind='full', primary_controls=['invariant'])


def setup(a, p, kind, device):
    graphs = [context(a, fold, device, kind == 'pairwise') for fold in range(p['folds'])]
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = EquivariantHypergraph([x.shape[1] for x in xs], a['node_counts'], int(a['relation'].max() + 1),
        kind, p['width'], p['depth'], p['dropout']).to(device)
    return model, xs, graphs, folds


def prepare():
    if OUT.exists():
        raise ValueError('Equivariant-hypergraph specification already exists')
    parent = common.protocol(common.OUT)
    sources = ('equivariant_hypergraph.py', 'test_equivariant_hypergraph.py',
               'overlap_block_models.py', 'overlap_block_experiment.py', 'non_tabular_experiment.py')
    paper = ROOT / 'research/new_mechanisms_20260915/edhnn_paper.pdf'
    p = dict(**parameters(), created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=parent['tasks'], selection=parent['selection'], specification_only=True,
        controls='Primary full versus invariant uses identical relation-star memberships, type/count information, initial parameters and budgets. invariant replaces each receiver state in rho with the same hyperedge mean state, retaining the same rho architecture but equal messages for all members of that hyperedge. Pairwise size-two hyperedges and no_hypergraph are diagnostics, not additional innovations.',
        method='equivariant_hypergraph_core', family='member_conditioned_hypergraph_operators',
        literature=dict(title='Equivariant Hypergraph Diffusion Neural Operators', venue='ICLR 2023', arxiv='2207.06680v3'),
        adaptation='Independent typed link-prediction adaptation of Algorithm 1 and equation 5. Relation-star hyperedges include the center and its neighbors. Phi pooling uses sqrt(cardinality) normalization and rho receives log1p cardinality; node updates use mean incidence aggregation and log1p incidence degree. Relation/center-type embeddings, GELU, residual update and output LayerNorm are local adaptations. Three iterations share phi/rho/update parameters. This is not explicit potential optimization or ADMM and has no claimed convergence or universal-approximation guarantee for the chosen finite width.',
        boundary='Delete fitting query target facts before physical counts, star memberships and all messages. Hyperedges are modeling groups derived only from existing support facts, not measured multi-entity biological relations. Node IDs are array indices only.',
        source_sha256={n: common.base.digest(ROOT / n) for n in sources},
        inputs={str(paper): common.base.digest(paper)},
        innovation_counting='One higher-order equivariant-operator family; distinct from the earlier GW transport of hypergraphs and from generative diffusion.')
    OUT.mkdir(parents=True)
    (OUT / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, OUT / 'source_snapshot' / name)
    common.base.write_json(OUT / 'protocol.json', p)
