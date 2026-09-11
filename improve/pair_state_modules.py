"""Joint node and RNA--drug pair-state encoding on the existing graph.

The extra state lives on (RNA, drug) pairs. RNA relations move its first
coordinate; drug relations move its second coordinate. Known-association
boundary values are rebuilt from the supplied, query-masked graph on every
forward pass. They never come from an unmasked cached label matrix.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


PAIR_TYPES = ('lncRNA', 'miRNA')


class AxisOperator(nn.Module):
    """Sparse incoming-mean transport; indices/values remain ordinary buffers."""
    def __init__(self, edge_index, n_source, n_target, remove_self=False):
        super().__init__()
        index = edge_index.detach().cpu().long()
        if remove_self:
            index = index[:, index[0] != index[1]]
        adjacency = torch.sparse_coo_tensor(index.flip(0), torch.ones(index.size(1)),
                                            (n_target, n_source)).coalesce()
        index, values = adjacency.indices(), adjacency.values()
        degree = torch.zeros(n_target).index_add_(0, index[0], values)
        values = values / degree[index[0]].clamp_min(1)
        self.register_buffer('indices', index)
        self.register_buffer('values', values)
        self.shape = (n_target, n_source)
        self._cached = None

    def forward(self, field):
        if field.size(0) != self.shape[1]:
            raise ValueError('Source axis does not match graph operator')
        if self._cached is None or self._cached.device != field.device:
            self._cached = torch.sparse_coo_tensor(
                self.indices, self.values, self.shape, device=field.device).coalesce()
        flat = torch.sparse.mm(self._cached, field.reshape(field.size(0), -1))
        return flat.reshape(self.shape[0], *field.shape[1:])

    def _load_from_state_dict(self, *args, **kwargs):
        self._cached = None
        return super()._load_from_state_dict(*args, **kwargs)


class PairBlock(nn.Module):
    def __init__(self, types, relations, width):
        super().__init__()
        self.types, self.relations = tuple(types), tuple(relations)
        self.self_maps = nn.ModuleDict({t: nn.Linear(width, width) for t in types})
        self.rna_maps = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in relations])
        self.drug_maps = nn.ModuleDict({t: nn.Linear(width, width, bias=False) for t in types})
        self.norms = nn.ModuleDict({t: nn.LayerNorm(width) for t in types})
        self.step = nn.ParameterDict({t: nn.Parameter(torch.tensor(-1.)) for t in types})

    def forward(self, fields, rna_operators, drug_operator, transport=True):
        messages = {t: self.self_maps[t](fields[t]) for t in self.types}
        counts = {t: 2 for t in self.types}
        for rel, op, linear in zip(self.relations, rna_operators, self.rna_maps):
            source, _, target = rel
            value = op(fields[source]) if transport else fields[target]
            messages[target] = messages[target] + linear(value)
            counts[target] += 1
        result = {}
        for t in self.types:
            # Transpose only the two pair coordinates, never the channel axis.
            value = drug_operator(fields[t].transpose(0, 1)).transpose(0, 1) if transport else fields[t]
            mixed = (messages[t] + self.drug_maps[t](value)) / math.sqrt(counts[t])
            update = F.gelu(self.norms[t](mixed))
            result[t] = fields[t] + self.step[t].sigmoid() * update
        return result


class PairStateBranch(nn.Module):
    def __init__(self, graph, hidden, rna_type, mode='pair_state', width=16, depth=2):
        super().__init__()
        if mode not in ('pair_local', 'pair_features', 'pair_state'):
            raise ValueError(mode)
        if width < 2 or depth < 1:
            raise ValueError('A nontrivial width and at least one layer are required')
        self.mode, self.width, self.rna_type = mode, width, rna_type
        self.types = tuple(t for t in PAIR_TYPES if t in graph.node_types)
        if rna_type not in self.types:
            raise ValueError('Missing target RNA type')
        self.sizes = {t: graph[t].num_nodes for t in self.types}
        self.n_drug = graph['drug'].num_nodes
        self.relations = tuple(k for k in graph.edge_types if k[0] in self.types and k[2] in self.types)
        self.associations = tuple(k for k in graph.edge_types if k[0] in self.types and k[2] == 'drug')
        self.rna_operators = nn.ModuleList([
            AxisOperator(graph[k].edge_index, graph[k[0]].num_nodes,
                         graph[k[2]].num_nodes, remove_self=k[0] == k[2]) for k in self.relations])
        drug_relations = [k for k in graph.edge_types if k[0] == k[2] == 'drug']
        drug_edges = torch.cat([graph[k].edge_index for k in drug_relations], dim=1)
        self.drug_operator = AxisOperator(drug_edges, self.n_drug, self.n_drug, remove_self=True)
        self.rna_projections = nn.ModuleDict({
            t: nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * width)) for t in self.types})
        self.drug_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * width))
        self.boundary_vectors = nn.ParameterDict({t: nn.Parameter(torch.randn(width) * .1) for t in self.types})
        self.blocks = nn.ModuleList([PairBlock(self.types, self.relations, width) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear((depth + 1) * width, 2 * width),
                                  nn.LeakyReLU(.1), nn.Linear(2 * width, 1))
        # Preserve the original model's function at initialization while all
        # components remain jointly trainable after the first head update.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def boundary(self, edge_index_dict, reference):
        result = {t: reference.new_zeros(self.sizes[t], self.n_drug) for t in self.types}
        for rel in self.associations:
            r, d = edge_index_dict[rel]
            result[rel[0]][r, d] = 1.
        return result

    def forward(self, z, edge_index_dict, pairs, intervention=None):
        if intervention not in (None, 'no_transport', 'no_seed'):
            raise ValueError(intervention)
        drug_main, drug_product = self.drug_projection(z['drug']).chunk(2, dim=-1)
        boundary = self.boundary(edge_index_dict, drug_main)
        use_seed = self.mode != 'pair_features' and intervention != 'no_seed'
        fields = {}
        for t in self.types:
            main, product = self.rna_projections[t](z[t]).chunk(2, dim=-1)
            field = main[:, None, :] + drug_main[None, :, :] + product[:, None, :] * drug_product[None, :, :]
            if use_seed:
                field = field + boundary[t][..., None] * self.boundary_vectors[t]
            fields[t] = field.tanh()
        r, d = pairs
        selected = [fields[self.rna_type][r, d]]
        transport = self.mode != 'pair_local' and intervention != 'no_transport'
        for block in self.blocks:
            fields = block(fields, self.rna_operators, self.drug_operator, transport=transport)
            selected.append(fields[self.rna_type][r, d])
        return self.head(torch.cat(selected, dim=-1)).flatten()


class PairStateModel(nn.Module):
    def __init__(self, base, graph, hidden, rna_type, arm, width=16, depth=2):
        super().__init__()
        self.base = base
        self.branch = (None if arm in ('pca_query', 'pca_default') else
                       PairStateBranch(graph, hidden, rna_type, arm, width, depth))

    def forward(self, x_dict, edge_index_dict, pairs, weights, return_components=False, intervention=None):
        z = self.base.encoder(x_dict, edge_index_dict, weights)
        original = self.base.decoder(z, pairs)
        correction = (torch.zeros_like(original) if self.branch is None else
                      self.branch(z, edge_index_dict, pairs, intervention=intervention))
        if return_components:
            return original + correction, original, correction
        return original + correction
