"""Check B-splines against scipy and verify paired initialization/symmetry."""
import unittest

import numpy as np
from scipy.interpolate import BSpline
import torch

from improve import spline_attention_graph_v1 as core


class SplineChecks(unittest.TestCase):
    def test_basis_and_input_derivative_match_scipy(self):
        scorer = core.SplineScore(3, 2).double()
        x = torch.tensor([[0., .17, 1.], [.11, .49, .82]], dtype=torch.float64, requires_grad=True)
        actual = core.cubic_basis(x, scorer.knots)
        knots = scorer.knots.numpy()
        reference = BSpline(knots, np.eye(7), 3)
        np.testing.assert_allclose(actual.detach(), reference(x.detach().numpy()), atol=2e-15, rtol=2e-15)
        torch.testing.assert_close(actual.sum(-1), torch.ones_like(x), atol=1e-14, rtol=1e-14)
        weights = torch.arange(7, dtype=torch.float64).square()
        derivative = torch.autograd.grad((actual * weights).sum(), x)[0]
        expected = reference.derivative()(x.detach().numpy()) @ weights.numpy()
        np.testing.assert_allclose(derivative, expected, atol=2e-13, rtol=2e-13)

    def test_scalar_map_values_and_gradient(self):
        torch.manual_seed(8)
        scorer = core.SplineScore(3, 2).double()
        x = torch.tensor([[.14, .32, .71], [.12, .42, .93]], dtype=torch.float64, requires_grad=True)
        reference = BSpline(scorer.knots.numpy(), np.eye(7), 3)(x.detach().numpy())
        spline = np.einsum('edb,hdb->eh', reference, scorer.coefficients.detach().numpy())
        base = torch.nn.functional.linear(torch.nn.functional.silu(x), scorer.base, scorer.bias)
        np.testing.assert_allclose(scorer(x).detach(), base.detach().numpy() + spline, atol=1e-14)
        self.assertTrue(torch.autograd.gradcheck(scorer, (x,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_shared_parameters_rng_and_capacity(self):
        p = core.parameters()
        torch.manual_seed(61)
        full = core.SplineGraph([3], 2, p, 'full')
        full_rng = torch.get_rng_state()
        torch.manual_seed(61)
        control = core.SplineGraph([3], 2, p, 'mlp_score')
        torch.testing.assert_close(full_rng, torch.get_rng_state(), atol=0, rtol=0)
        for key, value in full.state_dict().items():
            if '.score.' not in key:
                torch.testing.assert_close(value, control.state_dict()[key], atol=0, rtol=0)
        a = sum(v.numel() for v in full.parameters())
        b = sum(v.numel() for v in control.parameters())
        self.assertGreater(b, a)
        self.assertLess((b - a) / a, .005)

    def test_node_query_equivariance_and_attention_gradients(self):
        p = dict(core.parameters(), width=8, heads=2, depth=1, dropout=0.)
        torch.manual_seed(18)
        model = core.SplineGraph([3], 2, p, 'full').double().eval()
        torch.nn.init.normal_(model.readout[-1].weight, std=.1)
        xs = [torch.randn(5, 3, dtype=torch.float64)]
        edges = torch.tensor([[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 2, 0]])
        types, counts = torch.arange(7) % 4, torch.rand(5, 2, dtype=torch.float64)
        pairs = torch.tensor([[0, 2, 3], [3, 4, 1]])
        permutation = torch.tensor([3, 1, 4, 0, 2])
        inverse = permutation.argsort()
        actual = model(xs, (edges, types, counts), pairs)
        renamed = model([xs[0][permutation]], (inverse[edges], types, counts[permutation]), inverse[pairs])
        torch.testing.assert_close(actual, renamed, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual, model(xs, (edges, types, counts), pairs.flip(1)).flip(0), atol=0, rtol=0)
        actual.square().sum().backward()
        grad = model.layers[0].score.coefficients.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
