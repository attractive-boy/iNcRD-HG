import unittest

import numpy as np
import torch

from improve.antisymmetric_graph_dynamics import spatial_operators, AntisymmetricDynamics, setup, KINDS


class AntisymmetricGraphTests(unittest.TestCase):
    def test_operators_match_dense_definition_with_duplicates_and_isolate(self):
        edges = np.array([[0, 0, 1, 1, 2, 0], [1, 2, 2, 3, 2, 1]])
        sym, skew = spatial_operators(edges, 5, 'cpu')
        adjacency = np.zeros((5, 5))
        for u, v in edges.T:
            if u != v:
                adjacency[u, v] = adjacency[v, u] = 1
        degree = adjacency.sum(-1)
        inv = np.divide(1., degree, out=np.zeros(5), where=degree > 0)
        walk = inv[:, None] * adjacency
        np.testing.assert_allclose(sym.to_dense().numpy(), 2 * np.sqrt(inv[:, None] * inv[None, :]) * adjacency)
        np.testing.assert_allclose(skew.to_dense().numpy(), walk - walk.T, atol=1e-8)
        torch.testing.assert_close(sym.to_dense(), sym.to_dense().T, atol=0, rtol=0)
        torch.testing.assert_close(skew.to_dense(), -skew.to_dense().T, atol=0, rtol=0)
        self.assertEqual(sym.to_dense()[4].abs().sum().item(), 0)
        self.assertGreater(skew.values().abs().sum().item(), 0)
        _, regular = spatial_operators(np.array([[0, 1, 2], [1, 2, 0]]), 3, 'cpu')
        self.assertEqual(regular.values().abs().sum().item(), 0)

    def test_full_global_jacobian_skew_and_unconstrained_control_breaks_it(self):
        torch.manual_seed(7)
        # An irregular graph is needed to test the spatial branch.
        sym, skew = [x.double() for x in spatial_operators(np.array([[0, 0, 1], [1, 2, 3]]), 4, 'cpu')]
        layer = AntisymmetricDynamics(3, 4, .1, 0., .5, 'full').double()
        h = torch.randn(4, 3, dtype=torch.float64)
        jac = torch.autograd.functional.jacobian(lambda x: layer.linear_field(x, sym, skew), h).reshape(12, 12)
        torch.testing.assert_close(jac, -jac.T, atol=1e-12, rtol=1e-12)
        nonlinear = torch.autograd.functional.jacobian(lambda x: layer.linear_field(x, sym, skew).tanh(), h).reshape(12, 12)
        self.assertLess(torch.linalg.eigvals(nonlinear).real.abs().max().item(), 1e-10)
        layer.kind = 'free_channels'
        free = torch.autograd.functional.jacobian(lambda x: layer.linear_field(x, sym, skew), h).reshape(12, 12)
        self.assertGreater((free + free.T).abs().max().item(), .1)

    def test_dynamics_equivariant_and_spatial_parameters_receive_gradients(self):
        torch.manual_seed(4)
        edges = np.array([[0, 0, 1, 3], [1, 2, 3, 4]])
        sym, skew = [x.double() for x in spatial_operators(edges, 5, 'cpu')]
        layer = AntisymmetricDynamics(4, 6, .1, .1, .5, 'full').double()
        h = torch.randn(5, 4, dtype=torch.float64)
        permutation = torch.tensor([2, 0, 4, 1, 3])
        permuted = [x.to_dense()[permutation][:, permutation].to_sparse() for x in (sym, skew)]
        prediction = layer(h, sym, skew)
        torch.testing.assert_close(prediction[permutation], layer(h[permutation], *permuted), atol=1e-12, rtol=1e-12)
        prediction.square().sum().backward()
        for p in (layer.W, layer.V, layer.Z):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 1e-9)

    def test_query_removed_and_initial_parameters_identical_across_arms(self):
        rng = np.random.default_rng(13)
        a = dict(node_counts=np.array([3, 2]), nd=np.array(2), pair_offset=np.array([0, 3]),
            pairs_train=np.array([[0, 1, 2], [0, 1, 1]]), y_train=np.array([1, 1, 0]),
            folds=np.array([0, 1, 0]), edges=np.array([[0, 1, 0, 1], [3, 4, 1, 2]]),
            relation=np.array([0, 0, 1, 1]), target_codes=np.array([0, 3, -1, -1]),
            x_0=rng.normal(size=(3, 2)).astype(np.float32),
            x_1=rng.normal(size=(2, 4)).astype(np.float32))
        p = dict(width=8, depth=4, epsilon=.1, gamma=.1, beta=.5, dropout=0., folds=2)
        states = []
        for kind in KINDS:
            torch.manual_seed(11)
            model, xs, graphs, _ = setup(a, p, kind, 'cpu')
            states.append(model.state_dict())
            for operator in graphs[0][:2]:
                self.assertEqual(operator.to_dense()[0, 3].item(), 0)
                self.assertEqual(operator.to_dense()[3, 0].item(), 0)
            self.assertEqual(graphs[0][2][0, 0].item(), 0)
            pairs = torch.tensor([[0, 1], [3, 4]])
            prediction = model(xs, graphs[0], pairs)
            torch.testing.assert_close(prediction, model(xs, graphs[0], pairs.flip(1)).flip(0))
            prediction.sum().backward()
            self.assertTrue(torch.isfinite(model.project[0].weight.grad).all())
        for state in states[1:]:
            for key in state:
                torch.testing.assert_close(state[key], states[0][key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
