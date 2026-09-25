"""Binary copula calculus, conditional boundaries, nulls, and inference replay."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
import torch

from improve import bernoulli_copula_graph_v1 as copula
from improve.test_nested_anchor_v2 import fixture


class BernoulliCopulaTests(unittest.TestCase):
    def sample(self):
        generator = torch.Generator().manual_seed(512)
        q = torch.randn(21, 3, generator=generator, dtype=torch.float64).tanh()
        p, k = copula.correlation_factors(q)
        threshold = torch.randn(21, generator=generator, dtype=torch.float64)
        y = torch.arange(21).remainder(3).eq(0).double()
        return p, k, threshold, y

    def test_probit_gradient_hessian_against_autodiff(self):
        p, k, b, y = self.sample()
        u = torch.tensor([.3, -.2, .7], dtype=torch.float64, requires_grad=True)
        value, gradient, hessian = copula.posterior_terms(u, p, k, b, y)
        actual, = torch.autograd.grad(value, u, create_graph=True)
        dense = torch.stack([torch.autograd.grad(v, u, retain_graph=True)[0] for v in actual])
        torch.testing.assert_close(gradient, actual, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(hessian, dense, atol=1e-12, rtol=1e-12)
        self.assertGreaterEqual(torch.linalg.eigvalsh(hessian).min().item(), 1 - 1e-12)

    def test_laplace_mode_against_scipy_and_differentiability(self):
        p, k, b, y = self.sample()
        p = p.clone().requires_grad_()
        mean, covariance, detail = copula.laplace_posterior(p, k, b, y, iterations=8)
        def objective(x):
            value, derivative, _ = copula.posterior_terms(torch.tensor(x), p.detach(), k, b, y)
            return value.item(), derivative.numpy()
        optimized = minimize(objective, np.zeros(3), jac=True, method='BFGS', options={'gtol': 1e-9})
        np.testing.assert_allclose(mean.detach().numpy(), optimized.x, atol=1e-7)
        self.assertLess(detail['gradient_max'].item(), 1e-8)
        self.assertGreater(torch.linalg.eigvalsh(covariance).min().item(), 0)
        loss = mean.square().sum() + covariance.trace()
        derivative, = torch.autograd.grad(loss, p)
        self.assertTrue(torch.isfinite(derivative).all())
        self.assertGreater(derivative.abs().sum().item(), 1e-8)
        direction = torch.zeros_like(p); direction[2, 1] = 1
        def perturbed(e):
            m, s, _ = copula.laplace_posterior(p.detach() + e * direction, k, b, y, iterations=8)
            return (m.square().sum() + s.trace()).item()
        finite_difference = (perturbed(1e-5) - perturbed(-1e-5)) / 2e-5
        self.assertAlmostEqual(derivative[2, 1].item(), finite_difference, places=7)

    def test_correlation_and_gaussian_probit_moment(self):
        p, k, b, y = self.sample()
        correlation = p @ p.T + torch.diag(k)
        torch.testing.assert_close(correlation.diag(), torch.ones(21, dtype=torch.float64))
        self.assertGreaterEqual(k.min().item(), .25)
        self.assertGreater(torch.linalg.eigvalsh(correlation).min().item(), 0)
        mean, covariance, _ = copula.laplace_posterior(p, k, b, y)
        shift = copula.posterior_logit_shift(p, k, b, mean, covariance)
        points, weights = np.polynomial.hermite.hermgauss(100)
        # Integrate Phi((b + P u)/sqrt(K)) under the one-dimensional marginal
        # of the Gaussian Laplace factor; this is a separate quadrature check.
        for i in (0, 4, 12):
            m = (p[i] @ mean).item()
            v = (p[i] @ covariance @ p[i]).item()
            values = ndtr((b[i].item() + m + np.sqrt(2 * v) * points) / np.sqrt(k[i].item()))
            expected = float(values @ weights / np.sqrt(np.pi))
            actual = copula.gaussian_probit_logit(b[i], torch.tensor(m), k[i] + v).sigmoid().item()
            self.assertAlmostEqual(actual, expected, places=7)
        self.assertTrue(torch.isfinite(shift).all())
        null = copula.posterior_logit_shift(p, k, b, torch.zeros(3, dtype=torch.float64),
                                            torch.eye(3, dtype=torch.float64))
        torch.testing.assert_close(null, torch.zeros_like(null), atol=1e-12, rtol=1e-12)

    def test_query_facts_offsets_and_labels_do_not_enter_support_posterior(self):
        a, _ = fixture()
        parameters = copula.parameters(); parameters.update(width=8, rank=3, depth=1, dropout=0.)
        omitted = np.flatnonzero(a['folds'] == 0)
        graph = copula.context(a, omitted, parameters, 'cpu')
        changed = {k: v.copy() for k, v in a.items()}
        changed['y_train'][omitted] = 1 - changed['y_train'][omitted]
        changed['anchor_oof'] = np.full(len(a['y_train']), 999.)
        changed['y_val'] = np.arange(111)
        other = copula.context(changed, omitted, parameters, 'cpu')
        for name in ('edges', 'relation', 'support_pairs', 'support_labels', 'support_logits'):
            torch.testing.assert_close(getattr(graph, name), getattr(other, name), atol=0, rtol=0)
        forbidden = a['pairs_train'][:, omitted] + a['pair_offset'][:, None]
        edges = set(map(tuple, graph.edges.numpy().T))
        support = set(map(tuple, graph.support_pairs.numpy().T))
        self.assertFalse(any(tuple(q) in edges or tuple(q[::-1]) in edges or tuple(q) in support
                             for q in forbidden.T))

    def test_control_parameter_equality_gradients_and_query_replay(self):
        a, _ = fixture()
        parameters = copula.parameters(); parameters.update(width=8, rank=3, depth=1, dropout=0.)
        torch.manual_seed(51)
        model, xs, graphs, folds = copula.setup(a, parameters, 'full', 'cpu')
        independent = deepcopy(model); independent.kind = 'independent'
        self.assertEqual(set(model.state_dict()), set(independent.state_dict()))
        model.eval(); independent.eval()
        pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])[:, folds[0]]
        logits = torch.linspace(-3., 3., pairs.shape[1], dtype=torch.float64)
        actual = model(xs, graphs[0], pairs, logits)
        baseline = independent(xs, graphs[0], pairs, logits)
        torch.testing.assert_close(baseline, torch.zeros_like(baseline))
        self.assertGreater((actual - baseline).abs().max().item(), 1e-6)
        actual.square().mean().backward()
        self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0.)
        reversed_value = model(xs, graphs[0], pairs.flip(1), logits.flip(0)).flip(0)
        chunks = torch.cat([model(xs, graphs[0], pairs[:, i:i + 3], logits[i:i + 3])
                            for i in range(0, pairs.shape[1], 3)])
        torch.testing.assert_close(actual, reversed_value, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual, chunks, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
