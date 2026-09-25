"""LoMETab network and a per-drug group ranking loss.

LoMETab: Choi, Park, Kwon, Jeong, arXiv:2605.14365 (2026). It is a rank-r
generalization of the TabM/BatchEnsemble multiplicative implicit ensemble.
Each member's weight is W_k = W * (1 + A_k B_k^T) with a shared W and
member-specific low-rank factors (A_k, B_k).  For r >= 2 the hypothesis class
strictly enlarges the rank-1 TabM family; r and sigma_init control member
diversity.  The network below keeps TabM's shared-weight structure, per-member
biases and independent output heads, and simply replaces the rank-1 input/output
scaling layers with rank-r identity-residual Hadamard layers.

The group ranking loss is a differentiable surrogate for per-drug macro AP: for
every drug that has both positive and negative rows in the batch, it averages
softplus(logit_neg - logit_pos) over all within-drug positive/negative pairs and
all ensemble members.  Drugs with a single label class contribute nothing,
matching the macro_drug_ap definition.  The two metrics still come from one
shared checkpoint and one network; the loss is BCE + rank_lambda * group_loss.

Pair features, crossfitting and quantile preprocessing are reused unchanged
from the verified graph_tabm_modules pipeline.
"""
import math

import numpy as np
import torch
from torch import nn

from improve.graph_tabm_modules import PairFeatureEncoder, crossfit_features, fit_transform


class LoMETabLinear(nn.Module):
    """Weight = W * (1 + A_k B_k^T) per member; bias is member-specific.

    Inputs x are (N, M, input_dim) and outputs (N, M, output_dim), where M is
    the number of ensemble members.  The effective member weight matrix is
    assembled once per forward pass; A_k B_k^T is a rank-r (M, out, in) matrix.
    """

    def __init__(self, input_dim, output_dim, members, rank, sigma_init=0.1):
        super().__init__()
        assert rank >= 1
        self.members, self.rank = members, rank
        bound = 1 / math.sqrt(input_dim)
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim).uniform_(-bound, bound))
        self.A = nn.Parameter(torch.randn(members, output_dim, rank) * sigma_init)
        self.B = nn.Parameter(torch.randn(members, input_dim, rank) * sigma_init)
        self.bias = nn.Parameter(torch.empty(members, output_dim).uniform_(-bound, bound))

    def forward(self, x):
        ab = torch.einsum('mor,mir->moi', self.A, self.B)  # (M, out, in)
        effective = self.weight[None, :, :] * (1.0 + ab)  # (M, out, in)
        return torch.einsum('nmi,moi->nmo', x, effective) + self.bias


class LoMETab(nn.Module):
    def __init__(self, input_dim, width=128, depth=3, members=16, rank=2,
                 sigma_init=0.1, dropout=.1):
        super().__init__()
        self.members = members
        dims = [input_dim] + [width] * depth
        self.blocks = nn.ModuleList([
            LoMETabLinear(dims[i], dims[i + 1], members, rank, sigma_init)
            for i in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.output_weight = nn.Parameter(torch.empty(members, width).uniform_(-1/math.sqrt(width), 1/math.sqrt(width)))
        self.output_bias = nn.Parameter(torch.zeros(members))

    def forward(self, x):
        h = x[:, None].expand(-1, self.members, -1)
        for layer in self.blocks:
            h = self.dropout(torch.relu(layer(h)))
        return (h * self.output_weight).sum(-1) + self.output_bias


def group_ranking_loss(logits, targets, pairs, margin=0.0):
    """Per-drug pairwise softplus ranking loss over (N, M) logits.

    For each drug in the batch that has both positive and negative rows,
    average softplus(logit_neg - logit_pos) over all within-drug pos/neg pairs
    and all members, then average over drugs.  Returns a zero-valued leaf with
    gradients when no drug has both classes.
    """
    logits = torch.as_tensor(logits)
    targets = torch.as_tensor(targets).detach().cpu().numpy()
    pairs = np.asarray(pairs, dtype=np.int64)
    values = []
    for drug in np.unique(pairs[1]):
        mask = pairs[1] == drug
        pos = mask & (targets == 1)
        neg = mask & (targets == 0)
        npos, nneg = int(pos.sum()), int(neg.sum())
        if npos == 0 or nneg == 0:
            continue
        diff = logits[neg][:, None, :] - logits[pos][None, :, :]  # (nneg, npos, M)
        values.append(torch.nn.functional.softplus(diff - margin).mean())
    if not values:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return torch.stack(values).mean()