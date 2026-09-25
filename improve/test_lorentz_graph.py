import unittest

import numpy as np
import torch

from improve import lorentz_graph_models as m
from improve import overlap_block_models as graph


class TestLorentzGraph(unittest.TestCase):
    def test_lift_and_midpoint_on_manifold(self):
        torch.manual_seed(3)
        x = m.lift(torch.randn(7, 5, dtype=torch.float64), 2.)
        torch.testing.assert_close(m.lorentz_inner(x, x), torch.full((7,), -2., dtype=x.dtype))
        midpoint = m.normalize_midpoint(x.mean(0), 2.)
        torch.testing.assert_close(m.lorentz_inner(midpoint, midpoint), torch.tensor(-2., dtype=x.dtype))

    def test_lorentz_boost_distance_invariance(self):
        torch.manual_seed(5)
        x = m.lift(torch.randn(5, 3, dtype=torch.float64))
        y = m.lift(torch.randn(5, 3, dtype=torch.float64))
        eta = torch.tensor(.7, dtype=x.dtype)
        matrix = torch.eye(4, dtype=x.dtype)
        matrix[0, 0] = matrix[1, 1] = eta.cosh()
        matrix[0, 1] = matrix[1, 0] = eta.sinh()
        torch.testing.assert_close(m.chordal_sq(x, y), m.chordal_sq(x@matrix.T, y@matrix.T))

    def test_flat_limit_and_gradient(self):
        torch.manual_seed(2)
        x = (torch.randn(5, 4, dtype=torch.float64)*.01).requires_grad_()
        y = torch.randn(5, 4, dtype=torch.float64)*.01
        curved = m.chordal_sq(m.lift(x, 100.), m.lift(y, 100.), 100.)
        flat = (x-y).square().sum(-1)
        torch.testing.assert_close(curved, flat, atol=1e-8, rtol=1e-4)
        curved.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_masked_geometry_full_gradient_and_query_invariance(self):
        torch.manual_seed(11)
        model = m.LorentzGraph([5, 4], [4, 3], 1, width=8, depth=2, heads=2, dropout=0.)
        g = graph.graph_inputs(np.array([[0, 1, 2], [4, 5, 6]]), np.zeros(3, dtype=int),
                               np.empty((2, 0), dtype=int), 7, 1, 'cpu')
        xs = [torch.randn(4, 5), torch.randn(3, 4)]
        pairs = torch.tensor([[0, 2, 3], [5, 4, 6]])
        loss = model(xs, g, pairs).square().mean()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        full = m.predict(model, xs, [g], pairs)
        reverse = m.predict(model, xs, [g], pairs.flip(1)).flip(0)
        chunks = torch.cat([m.predict(model, xs, [g], q) for q in torch.tensor_split(pairs, 2, dim=1)])
        torch.testing.assert_close(full, reverse, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(full, chunks, atol=2e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
