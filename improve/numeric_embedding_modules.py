"""Feature-wise numerical embeddings for the existing graph-feature TabM.

Piecewise-linear encoding follows Gorishniy et al., NeurIPS 2022,
"On Embeddings for Numerical Features in Tabular Deep Learning". This is a
small local implementation, with either quantile or supervised tree knots.
All knots are fitted on the actual fitting rows, never evaluation rows.
"""
import math

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import torch
from torch import nn

from improve.graph_tabm_modules import GraphTabM


def fit_bin_edges(x_train, y_train=None, n_bins=16, method='quantile', min_leaf=16):
    """Return strictly increasing feature-wise knots; constants have one knot.

    A tree uses only fitting labels to choose informative scalar thresholds.
    Quantile knots do not depend on labels. Duplicate quantiles are removed.
    """
    x = np.asarray(x_train, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite, nonempty fitting features')
    if n_bins < 2 or method not in ('quantile', 'tree'):
        raise ValueError('Invalid binning method or bin count')
    if method == 'tree':
        y = np.asarray(y_train)
        if y.shape != (len(x),) or not np.isin(y, (0, 1)).all():
            raise ValueError('Tree binning requires fitting binary labels')
    edges = []
    for j in range(x.shape[1]):
        col = x[:, j]
        lo, hi = float(col.min()), float(col.max())
        if lo == hi:
            edge = np.array([lo], dtype=np.float32)
        elif method == 'quantile':
            edge = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)).astype(np.float32))
        else:
            tree = DecisionTreeClassifier(max_leaf_nodes=n_bins, min_samples_leaf=min_leaf,
                                          random_state=20260912)
            tree.fit(col[:, None], y)
            threshold = tree.tree_.threshold[tree.tree_.children_left != -1]
            edge = np.unique(np.r_[lo, threshold, hi].astype(np.float32))
        edges.append(edge)
    return edges


class PiecewiseLinearEncoding(nn.Module):
    """Cumulative ramp basis, padded per feature and clamped at the endpoints.

    For interval [b_i,b_{i+1}], coordinate i is clip((x-b_i)/(b_{i+1}-b_i),0,1).
    Constant features map to zero. Invalid padded intervals never contribute.
    """
    def __init__(self, edges, n_bins=16):
        super().__init__()
        left = torch.zeros(len(edges), n_bins)
        width = torch.ones(len(edges), n_bins)
        valid = torch.zeros(len(edges), n_bins)
        counts = []
        for j, edge in enumerate(edges):
            edge = np.asarray(edge, dtype=np.float32)
            if edge.ndim != 1 or not len(edge) or not np.isfinite(edge).all():
                raise ValueError('Invalid knots')
            if np.any(np.diff(edge) <= 0) or len(edge) > n_bins + 1:
                raise ValueError('Knots must increase strictly within the bin budget')
            count = len(edge) - 1
            counts.append(count)
            if count:
                left[j, :count] = torch.from_numpy(edge[:-1].copy())
                width[j, :count] = torch.from_numpy(np.diff(edge))
                valid[j, :count] = 1.
        self.register_buffer('left', left)
        self.register_buffer('width', width)
        self.register_buffer('valid', valid)
        self.register_buffer('counts', torch.tensor(counts, dtype=torch.long))

    def forward(self, x):
        return ((x[..., None] - self.left) / self.width).clamp(0., 1.) * self.valid


class NumericalEmbedding(nn.Module):
    """Separate trainable projection and ReLU for every numerical feature.

    The linear control expands the same scalars to the same output dimension.
    Its downstream TabM architecture is identical to the PLE arms.
    """
    def __init__(self, input_dim, embedding_dim=16, edges=None, n_bins=16):
        super().__init__()
        if edges is not None and len(edges) != input_dim:
            raise ValueError('One knot array is required per input feature')
        self.basis = PiecewiseLinearEncoding(edges, n_bins) if edges is not None else None
        basis_dim = n_bins if edges is not None else 1
        bound = 1. / math.sqrt(basis_dim)
        self.weight = nn.Parameter(torch.empty(input_dim, basis_dim, embedding_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(input_dim, embedding_dim).uniform_(-bound, bound))

    def forward(self, x):
        values = self.basis(x) if self.basis is not None else x[..., None]
        return torch.relu(torch.einsum('nfb,fbd->nfd', values, self.weight) + self.bias).flatten(1)


class EmbeddedTabM(nn.Module):
    def __init__(self, input_dim, embedding_dim=16, edges=None, n_bins=16,
                 width=128, depth=3, members=16, dropout=.1):
        super().__init__()
        self.embedding = NumericalEmbedding(input_dim, embedding_dim, edges, n_bins)
        self.backbone = GraphTabM(input_dim * embedding_dim, width, depth, members, dropout)

    def forward(self, x):
        return self.backbone(self.embedding(x))
