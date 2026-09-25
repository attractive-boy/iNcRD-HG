"""Covariance, variational-bound, quadrature and initialization contracts."""
import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from improve import variational_lastlayer_graph_v1 as core
from improve.test_nested_anchor_v2 import fixture


class VariationalLastLayerTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator(device='cpu').manual_seed(518)

    def rand(self, *shape):
        return torch.randn(*shape, generator=self.rng, dtype=torch.float64)

    def test_dense_covariance_and_gaussian_kl_match_torch_distribution(self):
        head = core.GaussianClassifier(4, 'full').double()
        with torch.no_grad():
            head.mean.copy_(self.rand(2, 5)*.2)
            head.log_scale.copy_(self.rand(2, 5)*.1-1.)
            head.lower.copy_(self.rand(*head.lower.shape)*.05)
        x = self.rand(7, 4)
        phi = torch.cat((x, torch.ones(7, 1, dtype=x.dtype)), -1)
        mu, variance = head(x)
        factor = head.factor()
        covariance = factor@factor.transpose(-1, -2)
        expected_variance = torch.einsum('nd,kde,ne->nk', phi, covariance, phi)
        torch.testing.assert_close(variance, expected_variance, atol=1e-12, rtol=1e-12)
        q = torch.distributions.MultivariateNormal(head.mean, scale_tril=factor)
        p = torch.distributions.MultivariateNormal(torch.zeros_like(head.mean),
            covariance_matrix=torch.eye(5, dtype=x.dtype)*head.prior_variance)
        expected = torch.distributions.kl_divergence(q, p).sum()
        torch.testing.assert_close(head.regularizer(), expected, atol=1e-12, rtol=1e-12)
        (mu.sum()+variance.sum()+head.regularizer()).backward()
        self.assertGreater(float(head.lower.grad.norm()), 0.)
        self.assertTrue(torch.isfinite(head.lower.grad).all())

    def test_zero_offdiagonal_equals_diagonal_and_map_has_only_mean_prior(self):
        full, diagonal, point = [core.GaussianClassifier(4, mode).double() for mode in core.KINDS]
        x = self.rand(6, 4)
        mean = self.rand(2, 5)*.2
        for head in (full, diagonal, point):
            with torch.no_grad():
                head.mean.copy_(mean)
        for x1, x2 in zip(full(x), diagonal(x)):
            torch.testing.assert_close(x1, x2, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(full.regularizer(), diagonal.regularizer(), atol=0, rtol=0)
        self.assertFalse(point.log_scale.requires_grad)
        self.assertFalse(point.lower.requires_grad)
        torch.testing.assert_close(point(x)[1], torch.zeros(6, 2, dtype=x.dtype), atol=0, rtol=0)
        torch.testing.assert_close(point.regularizer(), .5*mean.square().sum()/point.prior_variance)

    def test_jensen_bound_dominates_expected_bce_and_recovers_map(self):
        mean = self.rand(5, 2)
        variance = self.rand(5, 2).square()*.2
        offset = torch.tensor([-7., -1., 0., 2., 9.], dtype=mean.dtype)
        labels = torch.tensor([0, 1, 0, 1, 1])
        loss, bound, regularizer = core.objective(mean, variance, offset, labels, torch.tensor(9.), 30)
        self.assertAlmostEqual(float(regularizer), .3)
        points, log_weights = core.quadrature(128)
        nodes, weights = torch.as_tensor(points), torch.as_tensor(np.exp(log_weights))
        latent = offset+mean[:, 1]-mean[:, 0]
        sample = latent[:, None]+torch.sqrt(2*variance.sum(1))[:, None]*nodes
        exact_expectation = (F.binary_cross_entropy_with_logits(sample,
            labels.double()[:, None].expand_as(sample), reduction='none')*weights).sum(1).mean()
        self.assertGreaterEqual(float(bound-exact_expectation), -1e-12)
        _, map_nll, _ = core.objective(mean, torch.zeros_like(variance), offset, labels, torch.tensor(0.), 30)
        torch.testing.assert_close(map_nll, F.binary_cross_entropy_with_logits(latent, labels.double()), atol=1e-12, rtol=1e-12)

    def test_predictive_quadrature_symmetry_limit_and_reference_accuracy(self):
        m = torch.tensor([-30., -9., -3., 0., 3., 9., 30.], dtype=torch.float64)
        for v in (0., .01, .5, 2.):
            var = torch.full_like(m, v)
            approximate = core.logistic_normal_logit(m, var, 64)
            reference = core.logistic_normal_logit(m, var, 128)
            torch.testing.assert_close(approximate, reference, atol=2e-9, rtol=2e-9)
            torch.testing.assert_close(approximate, -approximate.flip(0), atol=2e-13, rtol=2e-13)
            if v == 0.:
                torch.testing.assert_close(approximate, m, atol=0, rtol=0)
        mean = torch.zeros(7, 2, dtype=m.dtype)
        zero, _, _ = core.residual_prediction(mean, torch.zeros_like(mean), m)
        torch.testing.assert_close(zero, torch.zeros_like(m), atol=0, rtol=0)

    def test_encoder_initialization_and_rng_match_across_posterior_arms(self):
        a, _ = fixture()
        p = core.parameters(); p.update(width=8, heads=2)
        initial, rng = None, None
        for mode in core.KINDS:
            torch.random.default_generator.manual_seed(29)
            model, _, _, _ = core.setup(a, p, mode, 'cpu')
            state = {k:v for k,v in model.state_dict().items() if not k.startswith('last.')}
            if initial is None:
                initial, rng = state, torch.get_rng_state()
            for key in state:
                torch.testing.assert_close(state[key], initial[key], atol=0, rtol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            core.setup(dict(a, y_val=np.ones(3)), p, 'full', 'cpu')


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
