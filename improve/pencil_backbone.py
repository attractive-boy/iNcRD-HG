"""A compact, typed adaptation of PENCIL (arXiv:2602.01553).

Each prediction uses local identity/adjacency tokens, two endpoint readout
tokens, a plain Transformer and adjacency-multiplication residuals. There
are no learned global node IDs and no original GNN inside this model.

Changes from the homogeneous paper: relation-specific adjacency slots,
type-specific feature projections, and log counts from the current support
graph. Counts retain information that a bounded neighborhood sample loses.
This is an adaptation, not a reproduction of the paper's published numbers.
"""
import math

import numpy as np
from scipy import sparse
import torch
from torch import nn


class SupportGraph:
    """Rebuilt from the exact graph visible to an optimization step.

    Query edges must already have been removed in both directions by the
    caller. Sampling reads topology only, never a label array. Adjacency is
    stored as a relation bit mask; the present PCA protocol has unit weights.
    """

    def __init__(self, x_dict, edges, weights=None):
        self.node_types = tuple(x_dict)
        self.edge_types = tuple(edges)
        if len(edges) > 30:
            raise ValueError('At most 30 relation types fit the adjacency bit mask')
        self.offsets = {}
        count = 0
        ids = []
        for i, (name, values) in enumerate(x_dict.items()):
            self.offsets[name] = count
            count += len(values)
            ids.append(np.full(len(values), i, dtype=np.int64))
        self.n_nodes = count
        self.type_ids_np = np.concatenate(ids)
        device = next(iter(x_dict.values())).device
        self.type_ids = torch.from_numpy(self.type_ids_np).to(device)
        self.bits = torch.zeros((count, count), dtype=torch.int32, device=device)
        self.degrees = torch.zeros((count, len(edges)), device=device)
        rows, cols = [], []
        for i, (key, edge) in enumerate(edges.items()):
            if weights is not None and not bool(torch.all(weights[key] == 1)):
                raise ValueError('PENCIL tokenizer currently requires unit edge weights')
            src = edge[0] + self.offsets[key[0]]
            dst = edge[1] + self.offsets[key[2]]
            # Rows are recipients, columns are message sources.
            self.bits[dst, src] = self.bits[dst, src] | (1 << i)
            self.degrees[:, i] = torch.bincount(dst, minlength=count).float()
            rows.append(src.detach().cpu().numpy())
            cols.append(dst.detach().cpu().numpy())
        row, col = np.concatenate(rows), np.concatenate(cols)
        adj = sparse.coo_matrix((np.ones(len(row), dtype=np.float32),
                                 (row, col)), shape=(count, count)).tocsr()
        adj = adj.maximum(adj.T)
        adj.setdiag(0)
        adj.eliminate_zeros()
        self.neighbors = [adj.indices[adj.indptr[i]:adj.indptr[i+1]]
                          for i in range(count)]

    def sample(self, pairs, rna_type, budget, seed):
        """Type-balanced one-hop context, with two-hop fill when space remains.

        Endpoints occupy slots 0 and 1. Other slots are randomly permuted.
        A pair's deterministic seed makes inference independent of batch size.
        This does not claim exact invariance to a different context ordering.
        """
        if budget < 4:
            raise ValueError('Context budget must be at least four nodes')
        pairs = np.asarray(pairs, dtype=np.int64)
        result = np.full((pairs.shape[1], budget), -1, dtype=np.int64)
        for i, (r, d) in enumerate(pairs.T):
            src = int(r) + self.offsets[rna_type]
            dst = int(d) + self.offsets['drug']
            rng = np.random.default_rng((int(seed) + src * 1000003 + dst * 9176) % (2**63-1))
            seen = {src, dst}
            chosen = []
            # Round robin over endpoint/type groups prevents the many circRNA
            # neighbors of a drug from consuming the entire sampling budget.
            groups = []
            for endpoint in (src, dst):
                neighbors = self.neighbors[endpoint]
                for nt in range(len(self.node_types)):
                    group = neighbors[self.type_ids_np[neighbors] == nt]
                    if len(group):
                        groups.append(list(rng.permutation(group)))
            while len(chosen) < budget-2 and any(groups):
                for j in rng.permutation(len(groups)):
                    while groups[j] and len(chosen) < budget-2:
                        node = int(groups[j].pop())
                        if node not in seen:
                            seen.add(node)
                            chosen.append(node)
                            break
            if len(chosen) < budget-2 and chosen:
                frontier = np.unique(np.concatenate([self.neighbors[v] for v in chosen]))
                for node in rng.permutation(frontier):
                    node = int(node)
                    if node not in seen:
                        seen.add(node)
                        chosen.append(node)
                        if len(chosen) == budget-2:
                            break
            chosen = list(rng.permutation(chosen))
            selected = [src, dst] + chosen
            result[i, :len(selected)] = selected
        return result

    def local(self, indices):
        """Typed incoming adjacency, token validity and support degrees."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.bits.device)
        valid = indices >= 0
        safe = indices.clamp(min=0)
        bits = self.bits[safe[:, :, None], safe[:, None, :]]
        relations = torch.arange(len(self.edge_types), device=bits.device)
        adjacency = ((bits[:, None] >> relations[None, :, None, None]) & 1).float()
        adjacency = adjacency * valid[:, None, :, None] * valid[:, None, None, :]
        return safe, valid, adjacency, self.degrees[safe] * valid[:, :, None]


class PencilLayer(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2*width, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.propagation = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.propagation.weight, std=.01)

    def forward(self, h, adjacency, padding, budget, use_graph=True):
        z = self.transformer(h, src_key_padding_mask=padding)
        if use_graph:
            # A common, fixed scale preserves different neighborhood totals.
            # It is deliberately not a per-node degree normalization.
            z = z + self.propagation(torch.bmm(adjacency, z)) / math.sqrt(budget)
        return z.masked_fill(padding[:, :, None], 0)


class HeteroPencil(nn.Module):
    def __init__(self, node_dims, edge_types, rna_type, width=64, depth=2,
                 heads=4, budget=34, dropout=.1, use_counts=True):
        super().__init__()
        if width % heads or depth < 1 or budget < 4:
            raise ValueError('Invalid PENCIL dimensions')
        self.node_types = tuple(node_dims)
        self.edge_types = tuple(edge_types)
        self.rna_type = rna_type
        self.width, self.budget = width, budget
        self.use_counts = bool(use_counts)
        self.features = nn.ModuleDict({nt: nn.Sequential(
            nn.Linear(dim, width), nn.LeakyReLU(.1), nn.Linear(width, width))
            for nt, dim in node_dims.items()})
        self.node_type = nn.Embedding(len(node_dims), width)
        # Local identity, relation-specific adjacency rows, and task role.
        input_dim = (1 + len(edge_types))*budget + 2
        self.structural = nn.Linear(input_dim, width, bias=False)
        nn.init.orthogonal_(self.structural.weight)
        self.structural.weight.requires_grad_(False)
        self.counts = nn.Sequential(nn.Linear(len(edge_types), width),
                                    nn.LeakyReLU(.1), nn.Linear(width, width))
        self.layers = nn.ModuleList([PencilLayer(width, heads, dropout)
                                     for _ in range(depth)])
        self.input_norm = nn.LayerNorm(width)
        self.output_norm = nn.LayerNorm(width)
        self.decoder = nn.Sequential(nn.Linear(2*width, 2*width),
            nn.LeakyReLU(.1), nn.Dropout(dropout), nn.Linear(2*width, width),
            nn.LeakyReLU(.1), nn.Linear(width, 1))

    def forward(self, x_dict, support, indices, intervention=None):
        if tuple(x_dict) != self.node_types or support.edge_types != self.edge_types:
            raise ValueError('Node/relation order differs from the trained model')
        if indices.shape[1] != self.budget:
            raise ValueError('Context budget differs from the trained model')
        ids, valid, typed_adj, degrees = support.local(indices)
        batch, count = ids.shape
        projected = torch.cat([self.features[nt](x_dict[nt]) for nt in self.node_types])
        attributes = projected[ids] + self.node_type(support.type_ids[ids])
        if self.use_counts and intervention != 'no_counts':
            attributes = attributes + self.counts(torch.log1p(degrees))
        identity = torch.eye(count, device=ids.device).expand(batch, -1, -1)
        adjacency_rows = typed_adj.permute(0, 2, 1, 3).reshape(batch, count, -1)
        structural = torch.cat([identity, adjacency_rows], dim=-1)
        # Both task tokens copy their endpoint's local identity and adjacency.
        structural = torch.cat([structural, structural[:, :2]], dim=1)
        role = structural.new_zeros((batch, count+2, 2))
        role[:, :count, 0] = 1
        role[:, count:, 1] = 1
        attributes = torch.cat([attributes, attributes[:, :2]], dim=1)
        h = self.input_norm(self.structural(torch.cat([structural, role], dim=-1)) + attributes)
        padding = torch.cat([~valid, torch.zeros((batch, 2), dtype=torch.bool, device=ids.device)], 1)
        # A union adjacency residual follows PENCIL; relation identity remains
        # available in the typed adjacency slots and type-specific features.
        adj = (typed_adj.sum(1) > 0).to(h.dtype)
        adj = torch.cat([adj, adj[:, :2]], dim=1)
        adj = torch.cat([adj, adj.new_zeros((batch, count+2, 2))], dim=2)
        h = h.masked_fill(padding[:, :, None], 0)
        for layer in self.layers:
            h = layer(h, adj, padding, count, use_graph=intervention != 'no_graph')
        z = self.output_norm(h[:, -2:]).reshape(batch, 2*self.width)
        return self.decoder(z).squeeze(-1)
