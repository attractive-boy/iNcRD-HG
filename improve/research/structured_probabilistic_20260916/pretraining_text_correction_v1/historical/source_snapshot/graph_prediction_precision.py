"""Deterministic graph inference and stable posterior probability averaging.

CUDA atomic aggregation can vary at the last few float32 bits. Bernoulli
posterior draws can turn those changes into a discrete membership change.
The deterministic evaluation context addresses that source; float64 sigmoid
and accumulation independently protect the inverse logit near probability
tails. Training and checkpoint selection are not changed by this module.

CUDA callers must start with CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8).
"""
from contextlib import contextmanager

import torch


@contextmanager
def deterministic_evaluation():
    previous = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


@torch.no_grad()
def block_prediction(model, xs, graphs, pairs, samples=16, seed=17031):
    with deterministic_evaluation():
        return _block_prediction(model, xs, graphs, pairs, samples, seed)


def _block_prediction(model, xs, graphs, pairs, samples, seed):
    model.eval()
    probabilities = []
    for fold, graph in enumerate(graphs):
        posterior = model.encode(xs, graph)
        generator = torch.Generator(device=pairs.device).manual_seed(seed+fold)
        draws = 1 if model.mode == 'deterministic' else samples
        estimates = []
        for _ in range(draws):
            z, _ = model.sample(posterior, generator, discrete=True)
            # float32 sigmoid saturates before the inverse-logit; averaging in
            # float64 only afterwards cannot recover the discarded precision.
            estimates.append(model.decode(z, posterior['counts'], pairs).double().sigmoid())
        probabilities.append(torch.stack(estimates).mean(0))
    p = torch.stack(probabilities).mean(0).clamp(1e-7, 1-1e-7)
    return torch.logit(p)


@torch.no_grad()
def geometry_prediction(model, xs, graphs, pairs):
    with deterministic_evaluation():
        model.eval()
        p = torch.stack([model(xs, graph, pairs).double().sigmoid() for graph in graphs]).mean(0)
        return torch.logit(p.clamp(1e-7, 1-1e-7))
