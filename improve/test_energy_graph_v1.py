"""Energy-gradient identity, topology boundaries, and mechanism controls."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from improve import energy_graph_v1 as energy
from improve.test_nested_anchor_v2 import fixture


class EnergyGraphTests(unittest.TestCase):
    def test_normalized_laplacian_matches_dense_and_is_psd(self):
        edges = np.array([[0, 1, 1, 2, 3, 1], [1, 0, 2, 1, 3, 2]])
        graph = energy.normalized_graph(edges, 5, 'cpu')
        adjacency = torch.zeros(5, 5, dtype=torch.float64)
        adjacency[0, 1] = adjacency[1, 0] = 1
        adjacency[1, 2] = adjacency[2, 1] = 1
        degree = adjacency.sum(1)
        inverse = degree.clamp_min(1).rsqrt()
        dense = torch.diag((degree > 0).double()) - inverse[:, None] * adjacency * inverse[None]
        torch.testing.assert_close(graph.laplacian(torch.eye(5, dtype=torch.float64)), dense)
        self.assertGreaterEqual(torch.linalg.eigvalsh(dense).min().item(), -1e-12)
        self.assertLessEqual(torch.linalg.eigvalsh(dense).max().item(), 2 + 1e-12)
        torch.testing.assert_close(dense[3:], torch.zeros(2, 5, dtype=torch.float64))

    def test_analytic_gradient_is_derivative_of_bounded_energy(self):
        a, _ = fixture()
        graph = energy.context(a, np.flatnonzero(a['folds'] == 0), 'cpu')
        generator = torch.Generator().manual_seed(912)
        h = torch.randn(23, 8, generator=generator, dtype=torch.float64, requires_grad=True)
        initial = torch.randn(23, 8, generator=generator, dtype=torch.float64)
        for kind in energy.KINDS:
            value, derivative, gate = energy.energy_and_gradient(h, initial, graph, kind=kind)
            expected, = torch.autograd.grad(value, h, retain_graph=True)
            torch.testing.assert_close(derivative, expected, atol=1e-12, rtol=1e-12)
            next_value, _, _ = energy.energy_and_gradient(h - 1e-4 * derivative, initial, graph, kind=kind)
            self.assertLess(next_value.item(), value.item())
            self.assertTrue(0 < gate.item() < 1)
        empty = energy.normalized_graph(np.empty((2, 0), dtype=int), 23, 'cpu')
        no_zeros = energy.EnergyContext(graph.positive, empty, graph.fidelity, np.array([]), np.array([]))
        _, full, _ = energy.energy_and_gradient(h, initial, no_zeros, kind='full')
        _, null, _ = energy.energy_and_gradient(h, initial, no_zeros, kind='no_repulsion')
        torch.testing.assert_close(full, null)

    def test_both_query_labels_and_query_edges_are_absent(self):
        a, _ = fixture()
        omitted = np.flatnonzero(a['folds'] == 0)
        graph = energy.context(a, omitted, 'cpu')
        changed = {k: v.copy() for k, v in a.items()}
        changed['y_train'][omitted] = 1 - changed['y_train'][omitted]
        other = energy.context(changed, omitted, 'cpu')
        for name in ('positive', 'negative'):
            for field in ('arcs', 'degree', 'weights', 'active'):
                torch.testing.assert_close(getattr(getattr(graph, name), field),
                                           getattr(getattr(other, name), field))
        expected = np.flatnonzero((a['folds'] != 0) & (a['y_train'] == 0))
        np.testing.assert_array_equal(graph.negative_training_indices, expected)
        forbidden = a['pairs_train'][:, omitted] + a['pair_offset'][:, None]
        for name in ('positive', 'negative'):
            arcs = set(map(tuple, getattr(graph, name).arcs.numpy().T))
            self.assertFalse(any(tuple(pair) in arcs or tuple(pair[::-1]) in arcs for pair in forbidden.T))

    def test_controls_share_parameters_but_change_forward_force(self):
        a, _ = fixture()
        p = energy.parameters()
        p.update(width=8, depth=3, dropout=0.)
        torch.manual_seed(71)
        model, xs, graphs, _ = energy.setup(a, p, 'full', 'cpu')
        pairs = torch.tensor(a['pairs_train'] + a['pair_offset'][:, None])
        model.eval()
        torch.testing.assert_close(model(xs, graphs[0], pairs), torch.zeros(pairs.shape[1]))
        nn = torch.nn
        nn.init.normal_(model.output[-1].weight, std=.1)
        full = model(xs, graphs[0], pairs)
        values = []
        for kind in energy.KINDS[1:]:
            control = deepcopy(model)
            control.kind = kind
            values.append(control(xs, graphs[0], pairs))
            self.assertEqual(set(model.state_dict()), set(control.state_dict()))
        self.assertTrue(all((full - value).abs().max().item() > 1e-6 for value in values))
        full.square().mean().backward()
        self.assertGreater(model.project[1].weight.grad.abs().sum().item(), 0.)
        torch.testing.assert_close(full, model(xs, graphs[0], pairs.flip(1)).flip(0))
        chunks = torch.cat([model(xs, graphs[0], pairs[:, i:i + 7]) for i in range(0, pairs.shape[1], 7)])
        torch.testing.assert_close(full, chunks, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
