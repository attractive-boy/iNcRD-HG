"""PULL-inspired auxiliary expected-graph self-training for a residual model.

Only support fitting negatives may be proposed as latent edges. A separate
absolute-probability head avoids interpreting a residual logit as a probability.
Inference uses the original support graph, as in the author's released code.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from improve import overlap_block_models as physical
from improve import sharpness_graph_v1 as shared
from improve import graph_prediction_precision as precision

KINDS = ('full', 'no_correction', 'observed_only')
identity_task = shared.identity_task


def parameters():
    return dict(width=32, depth=2, dropout=.1, folds=5, stage_steps=50,
        edge_increment=.05, top_degree_nodes=100, auxiliary_weight=.1,
        lnc_policy='identity_strong_reference')


def boolean_graph(edges, nodes):
    rows, cols = np.concatenate([edges, edges[::-1]], axis=1)
    keep = rows != cols
    graph = sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32),
                              (rows[keep], cols[keep])), shape=(nodes, nodes))
    graph.sum_duplicates()
    graph.data[:] = 1.
    graph.sort_indices()
    return graph


@dataclass
class WeightedGraph:
    edges: torch.Tensor
    weights: torch.Tensor
    nodes: int


def normalize_graph(adjacency, device):
    adjacency = adjacency + sparse.eye(adjacency.shape[0], dtype=np.float32, format='csr')
    adjacency.sort_indices()
    coo = adjacency.tocoo()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    weight = coo.data / np.sqrt(degree[coo.row] * degree[coo.col])
    return WeightedGraph(torch.as_tensor(np.stack((coo.row, coo.col)), device=device, dtype=torch.long),
                         torch.as_tensor(weight, device=device), adjacency.shape[0])


def confidence_selection(probabilities, budget):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError('Invalid latent edge probabilities')
    if np.any((probabilities < 0) | (probabilities > 1)) or budget < 0:
        raise ValueError('Invalid confidence or budget')
    budget = min(int(budget), len(probabilities))
    if not budget:
        return np.zeros(len(probabilities), dtype=bool), None
    cutoff = float(np.partition(probabilities, len(probabilities) - budget)[len(probabilities) - budget])
    return probabilities >= cutoff, cutoff


class ExpectedContext:
    def __init__(self, a, omitted, p, device):
        omitted = np.asarray(omitted, dtype=np.int64)
        edges, _ = physical.masked_physical(a, omitted)
        nodes = int(np.sum(a['node_counts']))
        self.adjacency = boolean_graph(edges, nodes)
        self.original = normalize_graph(self.adjacency, device)
        self.expected = self.original
        self.query_pairs = a['pairs_train'][:, omitted] + a['pair_offset'][:, None]
        support = np.ones(len(a['y_train']), dtype=bool)
        support[omitted] = False
        self.support_positive_count = int(np.sum(a['y_train'][support] == 1))
        candidate_ids = np.flatnonzero(support & (a['y_train'] == 0))
        candidates = a['pairs_train'][:, candidate_ids] + a['pair_offset'][:, None]
        target_type = shared.target_type(a)
        starts = np.r_[0, np.cumsum(a['node_counts'])[:-1]]
        target_nodes = np.r_[np.arange(starts[target_type], starts[target_type] + a['node_counts'][target_type]),
                             np.arange(starts[2], starts[2] + a['node_counts'][2])]
        # Degree comes solely from the query-deleted physical graph.
        degree = np.asarray(self.adjacency.sum(axis=1)).reshape(-1)
        top_count = min(p['top_degree_nodes'], len(target_nodes))
        cutoff = np.partition(degree[target_nodes], len(target_nodes) - top_count)[len(target_nodes) - top_count]
        eligible_nodes = np.zeros(nodes, dtype=bool)
        eligible_nodes[target_nodes] = degree[target_nodes] >= cutoff
        eligible = eligible_nodes[candidates[0]] | eligible_nodes[candidates[1]]
        self.candidate_ids, self.candidates = candidate_ids[eligible], candidates[:, eligible]
        if self.candidates.shape[1] and np.any(np.asarray(self.adjacency[self.candidates[0], self.candidates[1]])):
            raise ValueError('An unlabeled proposal is already a physical edge')
        if np.any(np.asarray(self.adjacency[self.query_pairs[0], self.query_pairs[1]])):
            raise ValueError('A fitting query is present in the original support')
        if set(self.candidate_ids) & set(omitted):
            raise ValueError('A query entered latent-edge candidates')
        self.query_teacher = np.zeros(len(omitted), dtype=np.float64)
        self.cutoff, self.stage, self.added = None, 0, self.candidates[:, :0]
        self.added_weights = np.empty(0, dtype=np.float64)
        self.audit = dict(support_positive_count=self.support_positive_count,
            candidate_count=self.candidates.shape[1], query_count=len(omitted),
            target_degree_cutoff=float(cutoff), forbidden_query_overlap=0,
            virtual_edges_are_observed_facts=False, virtual_candidates='Support fitting rows labelled zero only',
            no_reference_offsets_used_for_proposals=True,
            candidate_sha256=hashlib.sha256(np.ascontiguousarray(self.candidates).tobytes()).hexdigest())

    @torch.no_grad()
    def refresh(self, model, xs, stage, p, kind):
        if kind not in KINDS or stage < 0:
            raise ValueError('Invalid expected-graph stage')
        self.stage = int(stage)
        self.expected = self.original
        self.added, self.added_weights = self.candidates[:, :0], np.empty(0, dtype=np.float64)
        self.query_teacher = np.zeros(self.query_pairs.shape[1], dtype=np.float64)
        self.cutoff = None
        budget = int(math.floor(p['edge_increment'] * stage * self.support_positive_count))
        if kind != 'observed_only' and budget and self.candidates.shape[1]:
            training = model.training
            with precision.deterministic_evaluation():
                model.eval()
                hidden = model.encode(xs, self.original)
                device = hidden.device
                q = torch.as_tensor(self.candidates, device=device)
                confidence = model.decode(hidden, q)[1].double().sigmoid().cpu().numpy()
                mask, self.cutoff = confidence_selection(confidence, budget)
                self.added, self.added_weights = self.candidates[:, mask], confidence[mask]
                query = torch.as_tensor(self.query_pairs, device=device)
                self.query_teacher = model.decode(hidden, query)[1].double().sigmoid().cpu().numpy()
            model.train(training)
            rows, cols = np.concatenate((self.added, self.added[::-1]), axis=1)
            weights = np.r_[self.added_weights, self.added_weights].astype(np.float32)
            additions = sparse.csr_matrix((weights, (rows, cols)), shape=self.adjacency.shape)
            if np.any(np.asarray(additions[self.query_pairs[0], self.query_pairs[1]])):
                raise ValueError('A fitting query was inserted as a virtual edge')
            self.expected = normalize_graph(self.adjacency + additions, self.original.edges.device)
        return dict(stage=int(stage), nominal_virtual_budget=budget,
            actual_virtual_edges=self.added.shape[1], cutoff=self.cutoff,
            teacher_query_count=len(self.query_teacher),
            mean_virtual_weight=float(self.added_weights.mean()) if len(self.added_weights) else 0.,
            forbidden_query_overlap=0, correction='Original graph and observed fitting outcomes')

    def soft_targets(self, labels):
        if len(labels) != len(self.query_teacher):
            raise ValueError('Pseudo target/query length mismatch')
        teacher = torch.as_tensor(self.query_teacher, device=labels.device, dtype=labels.dtype)
        negative_target = torch.zeros_like(teacher) if self.cutoff is None else torch.where(
            teacher >= self.cutoff, teacher, torch.zeros_like(teacher))
        # The observed label is used only to define a training target, never to construct a graph.
        return torch.where(labels == 1, torch.ones_like(labels), negative_target)

    def evidence(self):
        return dict(stage=self.stage, query_pairs=self.query_pairs.copy(),
            added_pairs=self.added.copy(), added_weights=self.added_weights.copy(),
            query_teacher=self.query_teacher.copy(), cutoff=self.cutoff, audit=dict(self.audit))


class ExpectedGraphModel(nn.Module):
    def __init__(self, dimensions, p):
        super().__init__()
        width = p['width']
        self.p = dict(p)
        self.project = nn.ModuleList(nn.Linear(size, width) for size in dimensions)
        self.input_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(p['depth']))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(p['depth']))
        self.pair = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Dropout(p['dropout']))
        self.residual = nn.Linear(width, 1)
        self.absolute = nn.Linear(width, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def encode(self, xs, graph):
        h = torch.cat([layer(x / x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt())
                       for layer, x in zip(self.project, xs)])
        h = F.gelu(self.input_norm(h))
        source, target = graph.edges
        for layer, norm in zip(self.layers, self.norms):
            message = h.new_zeros(h.shape).index_add(0, target, h[source] * graph.weights[:, None])
            h = F.gelu(norm(layer(torch.cat((h, message), dim=-1))))
            h = F.dropout(h, self.p['dropout'], self.training)
        return h

    def decode(self, hidden, pairs):
        left, right = hidden[pairs[0]], hidden[pairs[1]]
        representation = self.pair(torch.cat((left, right, left * right, (left - right).abs()), dim=-1))
        return self.residual(representation).squeeze(-1), self.absolute(representation).squeeze(-1)

    def forward_both(self, xs, graph, pairs):
        return self.decode(self.encode(xs, graph), pairs)

    def forward(self, xs, context, pairs):
        return self.forward_both(xs, context.original, pairs)[0]


def setup(a, p, kind, device):
    if kind not in KINDS:
        raise ValueError('Unregistered expected-graph arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == fold), device=device) for fold in range(p['folds'])]
    if identity_task(a, p):
        return shared.IdentityResidual().to(device), [], [None] * p['folds'], folds
    graphs = [ExpectedContext(a, ids.cpu().numpy(), p, device) for ids in folds]
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    return ExpectedGraphModel([x.shape[1] for x in xs], p).to(device), xs, graphs, folds


def auxiliary_objective(expected_logits, original_logits, soft_targets, observed_labels, kind):
    if kind not in KINDS:
        raise ValueError('Unknown auxiliary self-training arm')
    expected_loss = F.binary_cross_entropy_with_logits(expected_logits.double(), soft_targets.double())
    correction = F.binary_cross_entropy_with_logits(original_logits.double(), observed_labels.double())
    # Replacement preserves total auxiliary weight; it is stricter than simply halving the loss.
    auxiliary = expected_loss if kind == 'no_correction' else .5 * (expected_loss + correction)
    return auxiliary, expected_loss, correction
