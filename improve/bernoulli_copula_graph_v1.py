"""Learned inter-edge Bernoulli copula conditioning, with Laplace inference.

Independent adaptation of the low-rank correlation idea in CopulaLSP (ICLR
2026). This is not its relaxed-Bernoulli likelihood or its signed encoder.
Only recorded support outcomes enter a differentiable probit-factor posterior.
Query marginals use externally supplied, properly nested reference logits.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import graph_observation_offset as nuisance
from improve import overlap_block_models as graph_lib
from improve import overlap_block_experiment as common

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/bernoulli_copula_core_v1_20260916'
KINDS = ('full', 'independent')


def correlation_factors(edge_embedding, epsilon=1.):
    """R = P P.T + diag(K), with exactly unit diagonal.

    Bounded embeddings and positive epsilon supply an explicit diagonal floor.
    No dense edge-by-edge matrix is materialized in the implementation.
    """
    if epsilon <= 0:
        raise ValueError('The covariance diagonal must be positive')
    q = edge_embedding.double()
    diagonal = q.square().sum(-1) + epsilon
    return q / diagonal.sqrt()[:, None], epsilon / diagonal


def latent_threshold(logits):
    # The clip protects the normal inverse CDF. Residual differences use the
    # same clipped threshold on both sides, preserving the identity null.
    return torch.special.ndtri(logits.double().sigmoid().clamp(1e-7, 1 - 1e-7))


def posterior_terms(u, factor, diagonal, threshold, labels):
    """Negative log probit posterior, exact gradient, and exact Hessian."""
    inverse_sd = diagonal.rsqrt()
    design = factor * inverse_sd[:, None]
    sign = 2 * labels.double() - 1
    t = sign * (threshold * inverse_sd + design @ u)
    log_cdf = torch.special.log_ndtr(t)
    mills = torch.exp(-.5 * t.square() - .5 * math.log(2 * math.pi) - log_cdf)
    curvature = mills * (mills + t)
    if not torch.isfinite(curvature).all() or (curvature < -1e-10).any():
        raise FloatingPointError('Invalid probit curvature')
    curvature = curvature.clamp_min(0.)
    identity = torch.eye(len(u), dtype=u.dtype, device=u.device)
    objective = .5 * u.square().sum() - log_cdf.sum()
    gradient = u - design.T @ (sign * mills)
    hessian = identity + design.T @ (curvature[:, None] * design)
    return objective, gradient, hessian


def laplace_posterior(factor, diagonal, threshold, labels, iterations=8):
    """Fixed differentiable Newton updates and covariance at the final state.

    Step clipping caps the Euclidean update at 3; no claim of an exact posterior
    or universal convergence follows from this finite Laplace approximation.
    All matrices to factorize have size equal to the embedding rank.
    """
    if iterations < 1:
        raise ValueError('A positive inference iteration count is required')
    u = factor.new_zeros(factor.shape[1])
    for _ in range(iterations):
        _, gradient, hessian = posterior_terms(u, factor, diagonal, threshold, labels)
        cholesky = torch.linalg.cholesky(hessian)
        direction = torch.cholesky_solve(gradient[:, None], cholesky).squeeze(-1)
        multiplier = (3. / direction.norm().clamp_min(3.)).clamp_max(1.)
        u = u - multiplier * direction
    objective, gradient, hessian = posterior_terms(u, factor, diagonal, threshold, labels)
    cholesky = torch.linalg.cholesky(hessian)
    covariance = torch.cholesky_inverse(cholesky)
    return u, covariance, dict(objective=objective, gradient_max=gradient.abs().max())


def gaussian_probit_logit(threshold, mean, variance):
    """Analytic Bernoulli probability after integrating a Gaussian latent."""
    if (variance <= 0).any():
        raise FloatingPointError('Nonpositive conditional latent variance')
    standardized = (threshold + mean) / variance.sqrt()
    # log(Phi(x)) - log(Phi(-x)) is stable in both tails.
    return torch.special.log_ndtr(standardized) - torch.special.log_ndtr(-standardized)


def posterior_logit_shift(query_factor, query_diagonal, query_threshold, mean, covariance):
    conditional_mean = query_factor @ mean
    conditional_variance = query_diagonal + ((query_factor @ covariance) * query_factor).sum(-1)
    posterior = gaussian_probit_logit(query_threshold, conditional_mean, conditional_variance)
    prior = gaussian_probit_logit(query_threshold, torch.zeros_like(query_threshold),
                                 query_diagonal + query_factor.square().sum(-1))
    return posterior - prior


@dataclass
class CopulaContext:
    edges: torch.Tensor
    relation: torch.Tensor
    support_pairs: torch.Tensor
    support_labels: torch.Tensor
    support_logits: torch.Tensor
    support_indices: np.ndarray
    omitted_indices: np.ndarray
    nuisance_audit: dict


def context(a, omitted, parameters, device):
    omitted = np.asarray(omitted, dtype=np.int64)
    n = int(a['node_counts'].sum())
    nr = int(np.max(a['relation'])) + 1
    edges, relation = graph_lib.masked_physical(a, omitted)
    arcs, typed, _ = graph_lib.graph_inputs(edges, relation, np.empty((2, 0), dtype=np.int64),
                                           n, nr, device)
    support = np.ones(len(a['y_train']), dtype=bool)
    support[omitted] = False
    ids = np.flatnonzero(support)
    pairs = a['pairs_train'][:, ids]
    labels = a['y_train'][ids]
    offsets = np.r_[0, np.cumsum(a['node_counts'])[:-1]]
    rna_type = int(np.flatnonzero(offsets == a['pair_offset'][0])[0])
    shape = (int(a['node_counts'][rna_type]), int(a['nd']))
    r, d, intercept, diagnostic = nuisance.fit_effects(pairs, labels, shape,
                                                    penalty=parameters['support_penalty'])
    support_logits = r[pairs[0]] + d[pairs[1]] + intercept
    return CopulaContext(arcs, typed,
        torch.as_tensor(pairs + a['pair_offset'][:, None], device=device),
        torch.as_tensor(labels, device=device),
        torch.as_tensor(support_logits, device=device), ids, omitted.copy(), diagnostic)


class BernoulliCopulaGraph(nn.Module):
    def __init__(self, dimensions, relations, parameters, kind='full'):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown copula arm')
        self.kind, self.config = kind, parameters
        width, rank = parameters['width'], parameters['rank']
        self.project = nn.ModuleList([nn.Linear(d, width) for d in dimensions])
        self.input_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([graph_lib.SparseTypedAttention(width, parameters['heads'],
            2 * relations + 2, parameters['dropout']) for _ in range(parameters['depth'])])
        self.embedding = nn.Linear(width, rank)
        self.margin = nn.Sequential(nn.Linear(4 * rank, width), nn.GELU(),
                                   nn.Dropout(parameters['dropout']), nn.Linear(width, 1))
        nn.init.zeros_(self.margin[-1].weight)
        nn.init.zeros_(self.margin[-1].bias)
        self.rna_degree, self.drug_degree = nn.Identity(), nn.Identity()

    def encode(self, xs, graph):
        values = []
        for x, project in zip(xs, self.project):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            values.append(project(x / rms))
        h = F.gelu(self.input_norm(torch.cat(values)))
        for layer in self.layers:
            h = layer(h, graph.edges, graph.relation)
        return torch.tanh(self.embedding(h))

    def pair_parameters(self, h, pairs):
        r, d = h[pairs[0]], h[pairs[1]]
        product = r * d
        residual = self.margin(torch.cat([r, d, product, (r - d).abs()], dim=1)).squeeze(-1).double()
        factor, diagonal = correlation_factors(product, self.config['epsilon'])
        return residual, factor, diagonal

    def forward(self, xs, graph, pairs, query_logits):
        h = self.encode(xs, graph)
        residual, query_factor, query_diagonal = self.pair_parameters(h, pairs)
        # Execute the same marginal branches in both arms so their dropout
        # streams stay paired; posterior inference itself uses no random draws.
        support_residual, support_factor, support_diagonal = self.pair_parameters(h, graph.support_pairs)
        if self.kind == 'independent':
            return residual
        support_threshold = latent_threshold(graph.support_logits + support_residual)
        mean, covariance, _ = laplace_posterior(support_factor, support_diagonal,
            support_threshold, graph.support_labels, self.config['posterior_steps'])
        query_threshold = latent_threshold(query_logits + residual)
        return residual + posterior_logit_shift(query_factor, query_diagonal, query_threshold,
                                                mean, covariance)


def setup(a, p, kind, device):
    graphs, folds = [], []
    for fold in range(p['folds']):
        omitted = np.flatnonzero(a['folds'] == fold)
        graphs.append(context(a, omitted, p, device))
        folds.append(torch.as_tensor(omitted, device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = BernoulliCopulaGraph([x.shape[1] for x in xs], int(a['relation'].max()) + 1, p, kind).to(device)
    return model, xs, graphs, folds


def parameters():
    return dict(width=64, rank=16, depth=2, heads=4, dropout=.1, folds=5,
                epsilon=1., posterior_steps=8, support_penalty=1.)


def prepare(out=OUT):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new copula core directory')
    names = ('bernoulli_copula_graph_v1.py', 'test_bernoulli_copula_graph_v1.py',
             'graph_observation_offset.py', 'overlap_block_models.py',
             'overlap_block_experiment.py', 'non_tabular_experiment.py')
    literature = ROOT / 'research/residual_next_20260916'
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family='bernoulli_copula',
        implementation_module='bernoulli_copula_graph_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(KINDS), primary_kind='full', primary_controls=['independent'],
        **parameters(), source_sha256={n: common.base.digest(ROOT / n) for n in names},
        inputs={str(literature / n): common.base.digest(literature / n) for n in
                ('copula_gnn_2026.pdf', 'COPULA_AUTHOR_FILES_V1.json')},
        literature=dict(title='A Scalable Inter-edge Correlation Modeling in CopulaGNN for Link Sign Prediction',
            venue='ICLR 2026 Poster', openreview='U7tR3lCRr5',
            official_pdf_sha256=common.base.digest(literature / 'copula_gnn_2026.pdf')),
        selection='Two fixed learning rates selected at seed 666 jointly across tasks; unchanged primary full.',
        mechanism='End-to-end learned edge Gramian couples binary labels through a latent probit factor. Differentiable support-only Newton/Laplace inference yields a conditional logit correction and covariance adjustment.',
        controls='independent has the same encoder, embedding and marginal decoder. The same embedding feeds its decoder, but the inter-edge conditional operator is removed.',
        adaptations=['Bernoulli/probit factor marginals instead of continuous relaxed Bernoulli labels.',
            'Finite Newton/Laplace posterior approximation, not exact binary likelihood or exact posterior.',
            'Conditional discriminative BCE, not the paper joint surrogate density objective.',
            'Ordinary typed attention over positive physical facts; no signed social-balance encoder.',
            'Bounded tanh node embeddings and epsilon=1 bound every diagonal below 1/(rank+1).',
            'Support marginals use a classical Rasch fit of that support only plus the learned decoder.',
            'Query marginals use the properly nested strong reference plus the same learned decoder.',
            'Different support/query base predictors are explicit fixed marginal inputs.',
            'Shared graph and covariance parameters are learned, not frozen pretrained representations.',
            'All inference uses deterministic Gaussian integration; no query-dependent Monte Carlo stream.'],
        boundary='Omit every positive and zero query-fold outcome before graph construction, support selection, and nuisance fitting. Global anchor_oof is never used for support marginals. Query probabilities and pair batches do not enter the support posterior.',
        limitations=['The graph depends on observed support labels, so this is a predictive conditional operator; no globally label-independent generative model is asserted.',
            'The original paper does not validate bipartite association prediction and does not show universal AUC gains.',
            'The paper correlation-space convergence result is not a guarantee for this neural optimization or finite Laplace inference.'],
        novelty_scope='One conditional edge-label dependence family. Rasch offsets, nested references, the encoder, and fixed final scaling are common infrastructure, not separate innovations.',
        original_test_source_access=False, confirmed_innovations=0, smoke=False)
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(ROOT / name, out / 'source_snapshot' / name)
    common.base.write_json(out / 'protocol.json', p)
    print('FROZEN_BERNOULLI_COPULA_CORE', out, flush=True)
