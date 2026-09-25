"""Learned neighbourhood Gaussian transport for typed link prediction.

This is an independent adaptation of covariance pooling and Gaussian
Bures--Wasserstein geometry, not an implementation of GBWBN or CCP-GNN.
Covariances describe learned message observations, not biological uncertainty.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import nested_anchor_inputs_v2 as nested
from improve import overlap_block_models as graph_lib
from improve.sharpness_graph_v1 import IdentityResidual, identity_task

KINDS = ('full', 'diagonal', 'root_euclidean')
TRAIN_KEYS = nested.TRAIN_KEYS | {'anchor_oof'}


def parameters():
    return dict(width=64, heads=4, depth=2, dropout=.1, folds=5,
        gaussian_heads=4, gaussian_dimension=8, covariance_ridge=.01,
        covariance_accumulation='float64 population covariance of incoming typed messages including loops',
        distance_dtype='float64', negative_distance_tolerance=1e-8,
        distribution_scope='all target-type endpoints, built without inspecting query list',
        lnc_policy='identity_strong_reference')


def sym(x):
    return (x+x.transpose(-1, -2))*.5


class _PositiveSqrt(torch.autograd.Function):
    """SPD square root with a Sylvester derivative, including repeated spectra.

    Autodifferentiating eigenvectors introduces eigenvalue-gap singularities.
    The derivative of the matrix function itself only uses sums of positive
    square roots; no stop-gradient or approximate inverse is used.
    """
    @staticmethod
    def forward(ctx, x):
        if x.dtype != torch.float64 or not torch.isfinite(x).all():
            raise ValueError('SPD square root requires finite float64 matrices')
        values, vectors = torch.linalg.eigh(sym(x))
        if not torch.isfinite(values).all() or torch.any(values <= 0):
            raise FloatingPointError('A nonpositive matrix reached the SPD square root')
        roots = values.sqrt()
        ctx.save_for_backward(roots, vectors)
        return sym((vectors*roots.unsqueeze(-2))@vectors.transpose(-1, -2))

    @staticmethod
    def backward(ctx, grad):
        roots, vectors = ctx.saved_tensors
        local = vectors.transpose(-1, -2)@sym(grad)@vectors
        result = local/(roots.unsqueeze(-1)+roots.unsqueeze(-2))
        return sym(vectors@result@vectors.transpose(-1, -2))


def positive_sqrt(x):
    return _PositiveSqrt.apply(x)


def population_moments(values, receiver, count):
    """Uniform empirical moments, retaining multiedge observations explicitly."""
    if values.dtype != torch.float64 or values.ndim != 3:
        raise ValueError('Expected float64 edge/head/coordinate message values')
    if len(receiver) != len(values) or torch.any(receiver < 0) or torch.any(receiver >= count):
        raise ValueError('Invalid receiver indices')
    degree = torch.bincount(receiver, minlength=count).to(values.dtype)
    if torch.any(degree == 0):
        raise ValueError('Every represented endpoint needs a support message (self loop)')
    mean = values.new_zeros((count, *values.shape[1:])).index_add(0, receiver, values)
    mean = mean/degree[:, None, None]
    # Centre observations before summation to avoid E[xx']-E[x]E[x]' cancellation.
    centered = values-mean[receiver]
    product = centered.unsqueeze(-1)*centered.unsqueeze(-2)
    cov = values.new_zeros((count, values.shape[1], values.shape[2], values.shape[2]))
    cov = cov.index_add(0, receiver, product)/degree[:, None, None, None]
    return mean, sym(cov)


def gaussian_cost(mean_left, mean_right, cov_left, cov_right, kind, tolerance=1e-8):
    """Per-head squared mean and covariance distances in the learned common space."""
    if kind not in KINDS:
        raise ValueError('Unregistered distributional distance')
    mean_cost = (mean_left-mean_right).square().sum(-1)
    if kind == 'diagonal':
        covariance_cost = (cov_left.diagonal(dim1=-2, dim2=-1).sqrt()
                           -cov_right.diagonal(dim1=-2, dim2=-1).sqrt()).square().sum(-1)
    else:
        left_root = positive_sqrt(cov_left)
        if kind == 'root_euclidean':
            covariance_cost = (left_root-positive_sqrt(cov_right)).square().sum((-2, -1))
        else:
            middle = sym(left_root@cov_right@left_root)
            cross = positive_sqrt(middle).diagonal(dim1=-2, dim2=-1).sum(-1)
            trace = (cov_left.diagonal(dim1=-2, dim2=-1).sum(-1)
                     +cov_right.diagonal(dim1=-2, dim2=-1).sum(-1))
            covariance_cost = trace-2*cross
            if torch.any(covariance_cost.detach() < -tolerance*(1+trace.detach())):
                raise FloatingPointError('Negative Bures distance beyond rounding tolerance')
            covariance_cost = covariance_cost.clamp_min(0)
    if not torch.isfinite(covariance_cost).all() or not torch.isfinite(mean_cost).all():
        raise FloatingPointError('Nonfinite Gaussian distances')
    return mean_cost, covariance_cost


class GaussianTransportGraph(nn.Module):
    def __init__(self, dimensions, node_counts, relations, p, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unregistered Gaussian transport arm')
        self.p, self.kind = dict(p), kind
        width, gh, gd = p['width'], p['gaussian_heads'], p['gaussian_dimension']
        self.project = nn.ModuleList([nn.Linear(d+relations, width) for d in dimensions])
        self.norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([graph_lib.SparseTypedAttention(width, p['heads'],
            2*relations+2, p['dropout']) for _ in range(p['depth'])])
        self.value = nn.Linear(width, gh*gd, bias=False)
        self.message_relation = nn.Embedding(2*relations+2, gh*gd)
        nn.init.normal_(self.message_relation.weight, std=.02)
        self.left_transform = nn.Parameter(torch.eye(gd).expand(gh, gd, gd).clone())
        self.right_transform = nn.Parameter(torch.eye(gd).expand(gh, gd, gd).clone())
        # Both the mean and covariance distances remain available to the same
        # non-monotone compatibility decoder. Drug response is not assumed to
        # increase monotonically as distributions become closer.
        self.readout = nn.Sequential(nn.Linear(4*width+4*gh, width), nn.GELU(),
            nn.Dropout(p['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()
        offsets = np.r_[0, np.cumsum(node_counts)]
        start, stop = int(offsets[1]), int(offsets[3])
        indices = torch.arange(start, stop, dtype=torch.long)
        mapping = torch.full((int(offsets[-1]),), -1, dtype=torch.long)
        mapping[indices] = torch.arange(len(indices))
        self.register_buffer('endpoint_indices', indices)
        self.register_buffer('endpoint_mapping', mapping)
        self.geometry_diagnostics = dict(forwards=0, message_observations=0,
            pair_head_distances=0, maximum_covariance_trace=0., maximum_distance=0.)

    def encode(self, xs, graph):
        edges, relation, counts = graph
        parts, start = [], 0
        for x, projection in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            parts.append(projection(torch.cat((x/rms, counts[start:start+len(x)]), dim=-1)))
            start += len(x)
        h = F.gelu(self.norm(torch.cat(parts)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        keep = self.endpoint_mapping[edges[1]] >= 0
        src, dst = edges[:, keep]
        gh, gd = self.p['gaussian_heads'], self.p['gaussian_dimension']
        values = self.value(h)[src]+self.message_relation(relation[keep])
        values = values.reshape(-1, gh, gd).double()
        means, covariances = population_moments(values, self.endpoint_mapping[dst],
                                                len(self.endpoint_indices))
        self.geometry_diagnostics['message_observations'] += len(src)
        return h, means, covariances

    def distribution(self, means, covariance, transform):
        transform = transform.double()
        mean = torch.einsum('hij,nhj->nhi', transform, means)
        cov = transform@covariance@transform.transpose(-1, -2)
        ridge = self.p['covariance_ridge']*torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        return mean, sym(cov)+ridge

    def forward(self, xs, graph, pairs):
        if torch.any(self.endpoint_mapping[pairs] < 0):
            raise ValueError('Decoder only accepts registered MiDrug endpoint types')
        h, means, cov = self.encode(xs, graph)
        r, d = pairs
        ri, di = self.endpoint_mapping[r], self.endpoint_mapping[d]
        left_mean, left_cov = self.distribution(means[ri], cov[ri], self.left_transform)
        right_mean, right_cov = self.distribution(means[di], cov[di], self.right_transform)
        mean_cost, cov_cost = gaussian_cost(left_mean, right_mean, left_cov, right_cov,
                                           self.kind, self.p['negative_distance_tolerance'])
        gd = self.p['gaussian_dimension']
        lt = left_cov.diagonal(dim1=-2, dim2=-1).sum(-1)
        rt = right_cov.diagonal(dim1=-2, dim2=-1).sum(-1)
        geometry = torch.cat((mean_cost, cov_cost, lt, rt), -1)
        geometry = torch.log1p(geometry/gd).to(h.dtype)
        left, right = h[r], h[d]
        feature = torch.cat((left, right, left*right, (left-right).abs(), geometry), dim=-1)
        diag = self.geometry_diagnostics
        diag['forwards'] += 1
        diag['pair_head_distances'] += mean_cost.numel()
        diag['maximum_covariance_trace'] = max(diag['maximum_covariance_trace'], float(torch.maximum(lt, rt).detach().max()))
        diag['maximum_distance'] = max(diag['maximum_distance'], float((mean_cost+cov_cost).detach().max()))
        return self.readout(feature).squeeze(-1)


def setup(a, p, kind, device):
    if set(a)-TRAIN_KEYS:
        raise ValueError('Gaussian transport setup accepts training arrays only')
    if kind not in KINDS:
        raise ValueError('Unregistered distributional arm')
    folds = [torch.as_tensor(np.flatnonzero(a['folds'] == f), device=device) for f in range(p['folds'])]
    if identity_task(a, p):
        model = IdentityResidual().to(device)
        model.geometry_diagnostics = {}
        return model, [], [None]*p['folds'], folds
    n, relations = int(a['node_counts'].sum()), int(a['relation'].max())+1
    graphs = []
    for fold in range(p['folds']):
        e, labels = graph_lib.masked_physical(a, np.flatnonzero(a['folds'] == fold))
        graphs.append(graph_lib.graph_inputs(e, labels, np.empty((2, 0), dtype=np.int64),
                                             n, relations, device))
    xs = [torch.as_tensor(a['x_'+str(i)], device=device) for i in range(len(a['node_counts']))]
    model = GaussianTransportGraph([x.shape[1] for x in xs], a['node_counts'], relations, p, kind).to(device)
    return model, xs, graphs, folds
