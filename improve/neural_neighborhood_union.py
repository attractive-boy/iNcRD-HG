"""Prospective union-pooling revision of the failed observed-NCN candidate.

This stays in the pair-conditioned neural structural-decoding family. The
NCN mechanism failed; its union diagnostic motivated a separately declared
candidate with a same-union type-count null. No claim of reproducing NCNC.
"""
from datetime import datetime, timezone
import shutil

import numpy as np
import torch

from improve import neural_common_neighbor as predecessor

common, ROOT = predecessor.common, predecessor.ROOT
OUT = ROOT / 'results/neural_neighborhood_union_core_20260915'
KINDS = ('full', 'count_only', 'node_only', 'shuffled')


def typed_permutation(node_counts, seed):
    rng = np.random.default_rng(seed)
    offset, pieces = 0, []
    for size in node_counts:
        pieces.append(rng.permutation(int(size)) + offset)
        offset += int(size)
    return np.concatenate(pieces)


def union_pool(adjacency, pairs, hidden, typed, kind, permutation):
    if kind not in KINDS:
        raise ValueError('Unknown neighborhood-union arm')
    if kind == 'node_only':
        return hidden.new_zeros((pairs.shape[1], hidden.shape[1]))
    r, d = pairs
    weights = (adjacency[r] | adjacency[d]).to(hidden.dtype)
    denominator = weights.sum(1, keepdim=True).clamp_min(1).sqrt()
    if kind == 'count_only':
        values = typed
    else:
        values = (hidden[permutation] if kind == 'shuffled' else hidden) + typed
    return (weights @ values) / denominator


class NeuralNeighborhoodUnion(predecessor.NeuralCommonNeighbor):
    def __init__(self, dims, node_counts, relations, kind, width=64, depth=3, dropout=.1):
        if kind not in KINDS:
            raise ValueError('Unknown neighborhood-union arm')
        super().__init__(dims, node_counts, relations,
                         'node_only' if kind == 'node_only' else 'union', width, depth, dropout)
        self.arm = kind

    def forward(self, xs, graph, pairs):
        if self.arm in ('full', 'node_only'):
            # Exact original arithmetic permits strict replay of existing fits.
            return super().forward(xs, graph[:4], pairs)
        h = self.encode(xs, graph)
        rna, drug = self.rna_readout(h), self.drug_readout(h)
        pooled = union_pool(graph[3], pairs, self.neighbor_readout(h),
                            self.type_value(self.node_types), self.arm, graph[4])
        r, d = pairs
        interaction = self.pair_readout(torch.cat((rna[r] * drug[d] / self.scale, pooled), 1)).squeeze(-1)
        counts = graph[2]
        return interaction + self.rna_degree(counts[r]).squeeze(-1) + self.drug_degree(counts[d]).squeeze(-1)


def parameters():
    return dict(kinds=list(KINDS), width=64, depth=3, dropout=.1, folds=5,
                auxiliary_weight=0., primary_kind='full',
                primary_controls=['count_only', 'node_only'], permutation_seed=41719)


def setup(a, p, kind, device):
    permutation = torch.as_tensor(typed_permutation(a['node_counts'], p['permutation_seed']), device=device)
    graphs = [(*predecessor.context(a, fold, device), permutation) for fold in range(p['folds'])]
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = NeuralNeighborhoodUnion([x.shape[1] for x in xs], a['node_counts'], int(a['relation'].max() + 1),
                                    kind, p['width'], p['depth'], p['dropout']).to(device)
    return model, xs, graphs, folds


def prepare():
    if OUT.exists():
        raise ValueError('Neighborhood union specification already exists')
    parent = common.protocol(predecessor.OUT)
    sources = ('neural_neighborhood_union.py', 'test_neural_neighborhood_union.py',
               *parent['source_sha256'])
    p = dict(**parameters(), created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=parent['tasks'], selection=parent['selection'], specification_only=True,
        controls='All arms use the same union support and matched initial parameters. count_only pools trainable node-type constants, without learned neighbor states. node_only uses the identical endpoint branch and zero pooling. shuffled permutes learned neighbor states within node type with a fixed label-independent permutation; it is an identity-alignment diagnostic.',
        method='neural_neighborhood_union_core', family='pair_conditioned_neural_structural_decoding',
        literature=parent['literature'],
        adaptation='Union of first-order endpoint neighborhoods, with the same typed sparse encoder, square-root-count normalization and pair readout as the predecessor diagnostic. This is a local revision inspired by MPNN-then-structural-function decoding, not the original NCN intersection or NCNC completion.',
        prior_results_seen='All 16 observed-NCN screening fits were evaluated. Full intersection failed; union diagnostic was promising. No union seeds 0/1 or fixed-partition predictions have been evaluated.',
        boundary=parent['boundary'], source_sha256={n: common.base.digest(ROOT / n) for n in sources},
        inputs={str(predecessor.OUT / 'protocol.json'): common.base.digest(predecessor.OUT / 'protocol.json'),
                **parent['inputs']},
        innovation_counting='This is a revision within NCN pair-conditioned structural decoding, not a separate family.')
    OUT.mkdir(parents=True)
    (OUT / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(ROOT / name, OUT / 'source_snapshot' / name)
    common.base.write_json(OUT / 'protocol.json', p)
