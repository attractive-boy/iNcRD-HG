"""Edge attributes and explicit RNA-mediated path instances on a fixed graph."""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve.attention_mechanisms import norm
from improve.drug_condition_attention import masked_weights
from improve.propagation_attention import WeightedMessages


ARMS = ('base', 'edge_fixed', 'edge_blind', 'edge_attributes', 'edge_shuffled',
        'path_count', 'path_uniform', 'path_attention')
PRIMARY = ('edge_attributes', 'path_attention')
SIMILARITIES = {
    ('lncRNA', 'LncLnc', 'lncRNA'): 'LncSimMat_ken.csv',
    ('miRNA', 'MiMi', 'miRNA'): 'MiSimMat_ken.csv',
    ('drug', 'DrugDrug', 'drug'): 'drugSimMat.csv',
}


def pack_mask(mask):
    """Pack every valid middle node, without top-k truncation or sampling."""
    counts = mask.sum(1).astype(np.int64)
    width = max(1, int(counts.max(initial=0)))
    ids = np.zeros((len(mask), width), dtype=np.int64)
    valid = np.zeros_like(ids, dtype=bool)
    rows, middle = np.nonzero(mask)
    positions = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
    ids[rows, positions] = middle
    valid[rows, positions] = True
    return dict(ids=torch.from_numpy(ids), valid=torch.from_numpy(valid),
                counts=torch.from_numpy(counts).float())


def build_assets(train, pairs, matrices, rna_type):
    attributes, families, packs = {}, [], {fold: [] for fold in pairs}
    for relation, matrix in matrices.items():
        if relation not in train.edge_types:
            continue
        n = train[relation[0]].num_nodes
        if matrix.shape != (n, n):
            raise ValueError('Similarity matrix / node count mismatch')
        edges = train[relation].edge_index.cpu().numpy()
        values = matrix[edges[0], edges[1]].astype(np.float32).copy()
        nonself = edges[0] != edges[1]
        if not np.isfinite(values[nonself]).all():
            raise ValueError('Nonfinite similarity on a retained edge')
        if (np.abs(values[nonself]) <= .3).any():
            raise ValueError('Similarity ordering does not match the frozen graph')
        values[~nonself] = 1.
        attributes[relation] = torch.from_numpy(np.stack([values, np.abs(values)], -1))
    for first, first_edges in train.edge_index_dict.items():
        source, _, middle = first
        if source != rna_type or middle == 'drug':
            continue
        for second in train.edge_types:
            if second[0] != middle or second[2] != 'drug':
                continue
            a = np.zeros((train[rna_type].num_nodes, train[middle].num_nodes), dtype=bool)
            e = first_edges.cpu().numpy()
            a[e[0], e[1]] = True
            if source == middle:
                # r -> r -> d would reveal the candidate edge during training.
                np.fill_diagonal(a, False)
            b = np.zeros((train['drug'].num_nodes, train[middle].num_nodes), dtype=bool)
            e = train[second].edge_index.cpu().numpy()
            b[e[1], e[0]] = True
            families.append(dict(first=list(first), second=list(second), middle=middle))
            for fold, indices in pairs.items():
                r, d = indices.cpu().numpy()
                packs[fold].append(pack_mask(a[r] & b[d]))
    if not families:
        raise ValueError('No RNA-mediated two-edge paths in this graph')
    return dict(attributes=attributes, families=families, packs=packs,
                pairs={f: p.cpu().clone() for f, p in pairs.items()})


def shuffle_attributes(attributes, edges, seed):
    """Permute undirected non-self edges together, preserving reciprocity."""
    result = attributes.clone()
    src, dst = edges.cpu().numpy()
    positions = np.flatnonzero(src != dst)
    canonical = np.stack([np.minimum(src[positions], dst[positions]),
                          np.maximum(src[positions], dst[positions])], -1)
    _, first, inverse = np.unique(canonical, axis=0, return_index=True, return_inverse=True)
    rng = np.random.RandomState(seed)
    representatives = attributes[torch.from_numpy(positions[first])]
    permuted = representatives[torch.from_numpy(rng.permutation(len(first)))]
    result[torch.from_numpy(positions)] = permuted[torch.from_numpy(inverse)]
    return result


class AttributeConv(nn.Module):
    def __init__(self, original, edges, attributes, n_target, mode, seed, width=16):
        super().__init__()
        dims = (original.lin_msg.in_channels, original.lin_self.in_channels)
        self.core = WeightedMessages(dims, original.out_channels)
        self.core.load_state_dict(original.state_dict())
        self.source = nn.Linear(dims[0], width, bias=False)
        self.target = nn.Linear(dims[1], width, bias=False)
        self.attribute = nn.Linear(2, width, bias=False)
        self.scorer = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.scorer.weight)
        self.mode, self.intervention = mode, None
        if mode == 'edge_fixed':
            for layer in (self.source, self.target, self.attribute, self.scorer):
                layer.requires_grad_(False)
        if mode == 'edge_blind':
            self.attribute.requires_grad_(False)
        if mode == 'edge_shuffled':
            attributes = shuffle_attributes(attributes, edges, seed)
        nonself = edges[0] != edges[1]
        destination = edges[1][nonself].cpu()
        # Fixed counts on CPU avoid unsupported deterministic CUDA bincount.
        degree = torch.bincount(destination, minlength=n_target).float()
        strength = attributes[nonself, 1]
        total = torch.zeros(n_target).index_add(0, destination, strength)
        fixed = torch.ones(edges.size(1))
        fixed[nonself] = strength * degree[destination] / total[destination].clamp_min(1e-12)
        self.register_buffer('attributes', attributes.clone())
        self.register_buffer('nonself', nonself.clone())
        self.register_buffer('destination', destination.clone())
        self.register_buffer('degree', degree)
        self.register_buffer('fixed_weights', fixed)
        self.last_weights = None

    def forward(self, x, edge_index, edge_attr=None, size=None):
        source, target = x if isinstance(x, (tuple, list)) else (x, x)
        if self.intervention == 'uniform':
            weights = source.new_ones(edge_index.size(1))
        elif self.mode == 'edge_fixed':
            weights = self.fixed_weights
        else:
            s, d = edge_index[:, self.nonself]
            hidden = self.source(norm(source))[s] + self.target(norm(target))[d]
            if self.mode != 'edge_blind' and self.intervention != 'no_attributes':
                hidden = hidden + self.attribute(self.attributes[self.nonself])
            scores = self.scorer(F.leaky_relu(hidden, .2)).squeeze(-1)
            positive = scores.sigmoid()
            total = torch.zeros_like(self.degree).index_add(0, self.destination, positive)
            scaled = positive * self.degree[self.destination] / total[self.destination].clamp_min(1e-12)
            weights = source.new_ones(edge_index.size(1))
            weights[self.nonself] = scaled
        self.last_weights = weights.detach()
        return self.core((source, target), edge_index, edge_attr=weights, size=size)


class PathBranch(nn.Module):
    def __init__(self, hidden, rna_type, assets, mode, width=32):
        super().__init__()
        self.rna_type, self.mode, self.width = rna_type, mode, width
        self.families = assets['families']
        self.query_r = nn.Linear(hidden, width, bias=False)
        self.query_d = nn.Linear(hidden, width, bias=False)
        self.keys = nn.ModuleList([nn.Linear(hidden, width, bias=False) for _ in self.families])
        self.values = nn.ModuleList([nn.Linear(hidden, width, bias=False) for _ in self.families])
        self.head = nn.Sequential(nn.Linear(len(self.families) * (width + 1), 32),
                                  nn.LeakyReLU(.1), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        for fold, entries in assets['packs'].items():
            for i, pack in enumerate(entries):
                for key, tensor in pack.items():
                    self.register_buffer(f'{fold}_{i}_{key}', tensor.clone())
                rows, slots = pack['valid'].nonzero(as_tuple=True)
                self.register_buffer(f'{fold}_{i}_rows', rows)
                self.register_buffer(f'{fold}_{i}_slots', slots)
                self.register_buffer(f'{fold}_{i}_flat_ids', pack['ids'][rows, slots])
        if mode != 'path_attention':
            self.query_r.requires_grad_(False)
            self.query_d.requires_grad_(False)
            self.keys.requires_grad_(False)
        if mode == 'path_count':
            self.values.requires_grad_(False)
        self.last_weights = {}

    def forward(self, z, pairs, fold, intervention=None):
        r, d = pairs
        if intervention == 'branch_off':
            return z[self.rna_type].new_zeros(r.size(0))
        normalised = {nt: norm(value) for nt, value in z.items()}
        query = self.query_r(normalised[self.rna_type][r]) + self.query_d(normalised['drug'][d])
        parts, self.last_weights = [], {}
        for i, family in enumerate(self.families):
            valid = getattr(self, f'{fold}_{i}_valid')
            counts = getattr(self, f'{fold}_{i}_counts')
            if self.mode == 'path_count':
                pooled = query.new_zeros((len(r), self.width))
            else:
                rows = getattr(self, f'{fold}_{i}_rows')
                slots = getattr(self, f'{fold}_{i}_slots')
                ids = getattr(self, f'{fold}_{i}_flat_ids')
                middle = normalised[family['middle']]
                if self.mode == 'path_uniform' or intervention == 'uniform':
                    weights = valid.to(query.dtype) / counts[:, None].clamp_min(1)
                else:
                    key = self.keys[i](middle)[ids]
                    scores = query.new_full(valid.shape, -1e4)
                    scores[rows, slots] = (query[rows] * key).sum(-1) / math.sqrt(self.width)
                    weights = masked_weights(scores, valid)
                values = self.values[i](middle)[ids]
                # Only real path instances enter the differentiable gather
                # and reduction. Padding never becomes a dummy middle node.
                pooled = query.new_zeros((len(r), self.width)).index_add(
                    0, rows, weights[rows, slots, None] * values)
                self.last_weights['path_' + str(i)] = weights.detach()
            parts.extend([pooled, counts.log1p()[:, None]])
        return self.head(torch.cat(parts, -1)).view(-1)


class EdgePathModel(nn.Module):
    def __init__(self, base, data, assets, config, arm, seed, width=32, edge_width=16):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.base, self.arm, self.branch = base, arm, None
        self.edge_layers = []
        for fold, pairs in assets['pairs'].items():
            self.register_buffer('pairs_' + fold, pairs.clone())
        if arm.startswith('edge_'):
            for layer in (0, 1):
                for i, (relation, attributes) in enumerate(assets['attributes'].items()):
                    path = 'convs.' + str(layer) + '.' + '__'.join(relation)
                    original = base.encoder.get_submodule(path)
                    module = AttributeConv(original, data[relation].edge_index, attributes,
                                           data[relation[2]].num_nodes, arm, seed + 700003 + i,
                                           edge_width)
                    parent, child = path.rsplit('.', 1)
                    setattr(base.encoder.get_submodule(parent), child, module)
                    self.edge_layers.append((path, module))
        elif arm.startswith('path_'):
            self.branch = PathBranch(config['hidden'], config['rna'], assets, arm, width)

    def forward(self, x, edges, fold, intervention=None):
        for _, module in self.edge_layers:
            module.intervention = intervention
        try:
            z = self.base.encoder(x, edges)
            pairs = getattr(self, 'pairs_' + fold)
            result = self.base.decoder(z, pairs)
            if self.branch is not None:
                result = result + self.branch(z, pairs, fold, intervention)
            return result
        finally:
            for _, module in self.edge_layers:
                module.intervention = None

    def attention(self):
        arrays = {path: module.last_weights.cpu().numpy()
                  for path, module in self.edge_layers if module.last_weights is not None}
        if self.branch is not None:
            arrays.update({name: weights.cpu().numpy() for name, weights in self.branch.last_weights.items()})
        return arrays
