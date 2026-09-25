"""Query-masked typed variational overlapping-block link prediction.

Inspired by TGSBM (WWW 2026, arXiv:2601.20646), not its official implementation.
The bipartite adaptation uses separate endpoint readouts and an IBP-style
cumulative-product Beta stick prior. This differs from the exclusive
stick-breaking formula printed in TGSBM. Binary-Concrete noise is logistic,
not a single Gumbel variable. Held-out outcomes never enter the encoder.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def expander_edges(n, degree, seed):
    """Label-independent permutation overlay, with explicit inverse arcs."""
    if degree % 2:
        raise ValueError('Expander degree must be even')
    rng = np.random.default_rng(seed)
    edges = []
    for _ in range(degree // 2):
        src, dst = np.arange(n), rng.permutation(n)
        keep = src != dst
        edges.extend((np.stack((src[keep], dst[keep])),
                      np.stack((dst[keep], src[keep]))))
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.unique(np.concatenate(edges, axis=1), axis=1)


def masked_physical(a, query_indices):
    """Remove queried association facts before creating reverse arcs/counts."""
    q = a['pairs_train'][:, np.asarray(query_indices)]
    codes = q[0] * int(a['nd']) + q[1]
    keep = ~np.isin(a['target_codes'], codes)
    if np.isin(a['target_codes'][keep], codes).any():
        raise AssertionError('Query fact remains visible')
    return a['edges'][:, keep], a['relation'][keep]


def graph_inputs(edges, relations, overlay, n, nr, device):
    """Only physical facts enter typed degree features; overlay is a new type."""
    counts = np.zeros((n, nr), dtype=np.float32)
    np.add.at(counts, (edges[0], relations), 1)
    np.add.at(counts, (edges[1], relations), 1)
    loops = np.stack((np.arange(n), np.arange(n)))
    arcs = np.concatenate((edges, edges[::-1], overlay, loops), axis=1)
    kinds = np.concatenate((relations, relations + nr,
                            np.full(overlay.shape[1], 2 * nr, dtype=np.int64),
                            np.full(n, 2 * nr + 1, dtype=np.int64)))
    return (torch.as_tensor(arcs, device=device), torch.as_tensor(kinds, device=device),
            torch.as_tensor(np.log1p(counts), device=device))


def edge_softmax(values, destination, n):
    """Stable sparse softmax independently for each receiving node/head."""
    ids = destination[:, None].expand_as(values)
    maxima = values.new_full((n, values.shape[1]), -torch.inf)
    maxima.scatter_reduce_(0, ids, values, reduce='amax', include_self=True)
    weights = (values - maxima[destination]).exp()
    denominator = values.new_zeros((n, values.shape[1]))
    denominator.index_add_(0, destination, weights)
    return weights / denominator[destination].clamp_min(1e-20)


class SparseTypedAttention(nn.Module):
    def __init__(self, width, heads, edge_types, dropout):
        super().__init__()
        if width % heads:
            raise ValueError('Width must be divisible by head count')
        self.heads, self.dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.edge = nn.Embedding(edge_types, width)
        nn.init.normal_(self.edge.weight, std=.02)
        self.output = nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edges, relation):
        n = len(h)
        q, k, v = self.qkv(self.norm1(h)).reshape(n, 3, self.heads, self.dim).unbind(1)
        src, dst = edges
        e = self.edge(relation).reshape(-1, self.heads, self.dim)
        score = (q[dst] * (k[src] + e)).sum(-1) / math.sqrt(self.dim)
        attention = edge_softmax(score, dst, n)
        aggregate = h.new_zeros((n, self.heads, self.dim))
        aggregate.index_add_(0, dst, attention[..., None] * v[src])
        h = h + self.dropout(self.output(aggregate.flatten(1)))
        return h + self.dropout(self.ff(self.norm2(h)))


def bernoulli_kl(q_logits, prior):
    prior = prior.clamp(1e-6, 1 - 1e-6)
    q = q_logits.sigmoid()
    return q * (F.logsigmoid(q_logits) - prior.log()) + (1-q) * (
        F.logsigmoid(-q_logits) - torch.log1p(-prior))


def concrete_sample(logits, temperature, generator):
    u = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype,
                   generator=generator).clamp(1e-6, 1-1e-6)
    return ((logits + u.log() - torch.log1p(-u)) / temperature).sigmoid()


class OverlapBlock(nn.Module):
    def __init__(self, dims, node_counts, relations, width=64, depth=3, heads=4,
                 communities=32, dropout=.1, mode='full'):
        super().__init__()
        if mode not in ('full', 'deterministic', 'no_stick', 'no_expander'):
            raise ValueError('Unknown mechanism control')
        self.mode, self.node_counts, self.communities = mode, node_counts, communities
        self.project = nn.ModuleList([nn.Linear(d+relations, width) for d in dims])
        self.input_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([SparseTypedAttention(width, heads, 2*relations+2, dropout)
                                     for _ in range(depth)])
        self.membership = nn.Linear(width, communities)
        self.strength = nn.Linear(width, communities)
        self.log_std = nn.Linear(width, communities)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -2.)
        # Positive Beta parameters; cumulative products provide sparse,
        # overlapping Bernoulli features rather than exclusive categories.
        self.stick_a = nn.Parameter(torch.full((communities,), math.log(math.expm1(10.))))
        self.stick_b = nn.Parameter(torch.full((communities,), math.log(math.expm1(1.))))
        self.rna_readout = nn.Sequential(nn.Linear(communities, 2*communities), nn.GELU(),
                                         nn.Linear(2*communities, communities))
        self.drug_readout = nn.Sequential(nn.Linear(communities, 2*communities), nn.GELU(),
                                          nn.Linear(2*communities, communities))
        self.rna_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))
        self.drug_degree = nn.Sequential(nn.Linear(relations, 32), nn.GELU(), nn.Linear(32, 1))

    def encode(self, xs, graph):
        edges, relation, counts = graph
        projected, offset = [], 0
        for x, project, size in zip(xs, self.project, self.node_counts):
            rms = x.square().mean(1, keepdim=True).clamp_min(1e-8).sqrt()
            projected.append(project(torch.cat((x / rms, counts[offset:offset+size]), 1)))
            offset += size
        h = F.gelu(self.input_norm(torch.cat(projected)))
        for layer in self.layers:
            h = layer(h, edges, relation)
        mu = self.strength(h)
        log_std = self.log_std(h).clamp(-5., 1.)
        a, b = F.softplus(self.stick_a)+1e-4, F.softplus(self.stick_b)+1e-4
        prior = (a/(a+b)).cumprod(0).clamp(1e-6, 1-1e-6)
        if self.mode == 'no_stick':
            prior = torch.full_like(prior, .5)
        evidence_logits = self.membership(h)
        logits = evidence_logits + torch.logit(prior)
        if self.mode == 'deterministic':
            kl = mu.sum() * 0.
        else:
            kl_r = .5 * (mu.square() + (2*log_std).exp() - 1 - 2*log_std).mean()
            beta_q = torch.distributions.Beta(a, b)
            beta_p = torch.distributions.Beta(torch.full_like(a, 10.), torch.ones_like(b))
            kl_v = torch.distributions.kl_divergence(beta_q, beta_p).mean()
            kl = kl_r + (0. if self.mode == 'no_stick' else kl_v / len(h))
        return dict(mu=mu, log_std=log_std, logits=logits, counts=counts, kl=kl,
                    prior=prior, a=a, b=b, evidence_logits=evidence_logits)

    def sample(self, posterior, generator, temperature=1., discrete=False):
        if self.mode == 'deterministic':
            return posterior['mu'], posterior['kl']
        mu = posterior['mu']
        if self.mode == 'no_stick':
            prior = torch.full_like(posterior['a'], .5)
        else:
            # Gamma-ratio reparameterization of Beta with a private RNG;
            # this does not disturb the matched encoder's dropout stream.
            ga = torch._standard_gamma(posterior['a'], generator=generator)
            gb = torch._standard_gamma(posterior['b'], generator=generator)
            prior = (ga/(ga+gb).clamp_min(1e-12)).cumprod(0).clamp(1e-6, 1-1e-6)
        logits = posterior['evidence_logits'] + torch.logit(prior)
        if discrete:
            u = torch.rand(logits.shape, device=logits.device, generator=generator)
            membership = (u < logits.sigmoid()).to(logits.dtype)
        else:
            membership = concrete_sample(logits, temperature, generator)
        noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        z = membership * (mu + posterior['log_std'].exp() * noise)
        return z, posterior['kl'] + bernoulli_kl(logits, prior).mean()

    def decode(self, z, counts, pairs):
        r, d = pairs
        interaction = (self.rna_readout(z[r]) * self.drug_readout(z[d])).sum(-1)
        return interaction / math.sqrt(self.communities) + self.rna_degree(counts[r]).squeeze(-1) + \
            self.drug_degree(counts[d]).squeeze(-1)

    def forward(self, xs, graph, pairs, generator, temperature=1.):
        posterior = self.encode(xs, graph)
        z, kl = self.sample(posterior, generator, temperature)
        return self.decode(z, posterior['counts'], pairs), kl


@torch.no_grad()
def posterior_prediction(model, xs, graphs, pairs, samples=16, seed=17031):
    """Fixed common random numbers make evaluation deterministic and query-free."""
    model.eval()
    probabilities = []
    for fold, graph in enumerate(graphs):
        posterior = model.encode(xs, graph)
        generator = torch.Generator(device=pairs.device).manual_seed(seed + fold)
        draws = 1 if model.mode == 'deterministic' else samples
        estimates = []
        for _ in range(draws):
            z, _ = model.sample(posterior, generator, discrete=True)
            estimates.append(model.decode(z, posterior['counts'], pairs).sigmoid())
        probabilities.append(torch.stack(estimates).mean(0))
    p = torch.stack(probabilities).mean(0).clamp(1e-7, 1-1e-7)
    return torch.logit(p)
