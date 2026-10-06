"""Degree-preserving gated fusion of a type-specific channel and a shared channel.

Why this shape
--------------
The repo measured, on the released protocol, that replacing the relation sum
aggregation with a mean (weight-sum-1, i.e. softmax-style normalisation) costs
0.0394 / 0.0404 test AUPR on MiDrug / LncDrug, and that a gate which restores
each target's in-degree total weight recovers +0.0245 / +0.0340 AUPR relative to
that weight-sum-1 gate.  Every attention that normalises the incoming mass away
(HAN-style relation attention, GATv2) is negative on both tasks
(`results/propagation_attention_20260907/REPORT.md`).  The failure is therefore
not "attention" in general but *discarding the degree scale the metric reads*.

This module keeps that scale in two ways:

1. `degree_preserving_weights` renormalises learned edge scores so that every
   target node's incoming weight sums to its in-degree, never to 1.
2. `SharedSpecificGate` adds the shared channel as a *rejectable residual* on
   top of the specific channel instead of renormalising a mixture between them:
   `h = h_specific + g * h_shared`, with a per-node/per-channel sigmoid gate
   initialised near zero.  The shared channel can be switched off, so the fused
   model starts at the specific baseline and cannot lose the degree mass.

Nothing here reads pair labels; the gate covariates are label-free.
"""
import torch
from torch import nn


def degree_preserving_weights(scores, edge_index, num_nodes):
    """a_ij = d_j * u_ij / sum_{k->j} u_kj, so that sum_j a_ij == in-degree(j).

    scores: (E,) non-negative edge scores u_ij (e.g. a sigmoid output).
    edge_index: (2, E) source -> target.
    num_nodes: target count (int) or (num_src, num_tgt) for a heterogeneous graph.
    """
    if isinstance(num_nodes, int):
        num_tgt = num_nodes
    else:
        num_tgt = num_nodes[1]
    src, dst = edge_index
    eps = torch.finfo(scores.dtype).eps
    denom = scores.new_zeros(num_tgt).index_add_(0, dst, scores).clamp_min(eps)
    degree = scores.new_zeros(num_tgt).index_add_(
        0, dst, torch.ones_like(scores)).clamp_min(1.0)
    return scores * (degree[dst] / denom[dst])


class SharedSpecificGate(nn.Module):
    """h_out = h_specific + gate * h_shared, gate = sigmoid(MLP([h_specific, h_shared, evidence])).

    `init_bias` is negative so the gate starts almost closed: the fused model is
    initially the specific baseline and only opens the shared channel if the
    training signal asks for it.  No softmax over channels, no renormalisation
    of `h_specific`.
    """

    def __init__(self, hidden, evidence_dim=0, init_bias=-2.0):
        super().__init__()
        self.evidence = (nn.Linear(evidence_dim, 2 * hidden, bias=False)
                         if evidence_dim else None)
        self.gate = nn.Linear(2 * hidden, hidden)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, init_bias)
        self.last_gate = None

    def forward(self, h_specific, h_shared, evidence=None):
        x = torch.cat([h_specific, h_shared], dim=-1)
        if self.evidence is not None and evidence is not None:
            x = x + self.evidence(evidence)
        g = torch.sigmoid(self.gate(x))
        self.last_gate = g.detach()
        return h_specific + g * h_shared
