"""Exact linear-attention GP covariance with typed attributes and link conditioning.

The spatial covariance is Corollary 8 of arXiv:2603.17569v1. Symmetric graph
normalization, type-block input covariance, layer variance normalization, and
tensor link regression are explicit local adaptations. Binary outcomes are
regression targets here; this is not an exact Bernoulli GP posterior.
"""
import numpy as np
from scipy import sparse
import torch

from improve import overlap_block_models as physical

TRAIN_KEYS = {'pairs_train', 'y_train', 'folds', 'edges', 'relation', 'target_codes',
              'node_counts', 'pair_offset', 'nd', 'x_0', 'x_1', 'x_2', 'x_3'}
KINDS = ('full', 'gcn_gp')


def support_operator(a, heldout):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Only explicitly listed training arrays are accepted')
    edges, _ = physical.masked_physical(a, np.asarray(heldout, dtype=np.int64))
    n = int(a['node_counts'].sum())
    graph = sparse.csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n, n))
    graph = graph.maximum(graph.T)
    graph.data[:] = 1.
    graph.setdiag(1.)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    norm = sparse.diags(degree ** -.5)
    return (norm @ graph @ norm).tocsr()


def initial_covariance(a, device='cpu'):
    blocks = []
    for i, n in enumerate(a['node_counts']):
        x = torch.as_tensor(a['x_'+str(i)], dtype=torch.float64, device=device)
        if len(x) != n or not torch.isfinite(x).all():
            raise ValueError('Invalid frozen entity features')
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        # A shared, fixed unit feature within each type also handles zero rows.
        blocks.append(.5 * (x @ x.T + 1.))
    return normalize(torch.block_diag(*blocks))


def normalize(covariance):
    diagonal = torch.diag(covariance)
    if not torch.isfinite(covariance).all() or torch.any(diagonal <= 0):
        raise ValueError('Require a finite covariance with positive marginal variances')
    scale = diagonal.sqrt()
    return .5 * (covariance + covariance.T) / (scale[:, None] * scale[None, :])


def relu_covariance(covariance):
    scale = torch.diag(covariance).sqrt()
    product = scale[:, None] * scale[None, :]
    cosine = (covariance / product).clamp(-1., 1.)
    angle = torch.acos(cosine)
    return product * (torch.sqrt((1.-cosine.square()).clamp_min(0.)) + (torch.pi-angle)*cosine) / (2.*torch.pi)


def torch_operator(operator, device):
    operator = sparse.coo_matrix(operator)
    indices = torch.as_tensor(np.stack((operator.row, operator.col)), device=device, dtype=torch.int64)
    data = torch.as_tensor(operator.data, dtype=torch.float64, device=device)
    return torch.sparse_coo_tensor(indices, data, operator.shape, device=device).coalesce()


def aggregate(operator, covariance):
    left = torch.sparse.mm(operator, covariance)
    return torch.sparse.mm(operator, left.T).T


def spatial_covariance(operator, covariance, kind):
    propagated = aggregate(operator, covariance)
    if kind == 'full':
        return covariance * propagated + aggregate(operator, covariance.square())
    if kind == 'gcn_gp':
        return propagated
    raise ValueError('Unregistered graph GP kind')


@torch.no_grad()
def entity_kernels(a, heldout, kind, depth=2, device='cpu'):
    operator = torch_operator(support_operator(a, heldout), device)
    covariance = initial_covariance(a, device)
    for _ in range(depth):
        covariance = normalize(relu_covariance(spatial_covariance(operator, covariance, kind)))
    offsets = np.r_[0, np.cumsum(a['node_counts'])]
    if list(a['pair_offset']) != [int(offsets[1]), int(offsets[2])]:
        raise ValueError('This frozen candidate applies only to MiDrug')
    rna = covariance[offsets[1]:offsets[2], offsets[1]:offsets[2]].clone()
    drug = covariance[offsets[2]:offsets[3], offsets[2]:offsets[3]].clone()
    return rna, drug


@torch.no_grad()
def condition(rna_kernel, drug_kernel, pairs, y, ridge=10.):
    """Exact tensor-product GP regression, with a saved full-universe mean."""
    device = rna_kernel.device
    pairs = torch.as_tensor(pairs, dtype=torch.int64, device=device)
    y = torch.as_tensor(y, dtype=torch.float64, device=device)
    if pairs.shape != (2, len(y)) or ridge <= 0 or not torch.all((y == 0) | (y == 1)):
        raise ValueError('Invalid binary support labels or kernel ridge precision')
    r, d = pairs
    kernel = rna_kernel[r[:, None], r[None, :]] * drug_kernel[d[:, None], d[None, :]]
    kernel.diagonal().add_(ridge)
    factor = torch.linalg.cholesky(kernel)
    target = 2.*y - 1.
    alpha = torch.cholesky_solve(target[:, None], factor).flatten()
    error = float((kernel @ alpha - target).abs().max())
    if error > 1e-7:
        raise ValueError('Exact kernel solve residual exceeds tolerance')
    weights = torch.zeros((len(rna_kernel), len(drug_kernel)), device=device, dtype=torch.float64)
    weights.index_put_((r, d), alpha, accumulate=True)
    mean = rna_kernel @ weights @ drug_kernel.T
    return mean, alpha, error
