import unittest

import numpy as np
import torch

from improve.learned_graph_structure import distance_adjacency, StructureLayer, setup, KINDS


class LearnedStructureTests(unittest.TestCase):
    def test_affinity_decreases_with_distance_and_has_no_self_edges(self):
        field = torch.tensor([[1., 0.], [.8, .6], [-1., 0.]], dtype=torch.float64, requires_grad=True)
        temperature = torch.tensor(3., dtype=torch.float64, requires_grad=True)
        threshold = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        a = distance_adjacency(field, temperature, threshold)
        self.assertGreater(a[0, 1].item(), a[0, 2].item())
        torch.testing.assert_close(a.sum(-1), torch.ones(3, dtype=torch.float64))
        torch.testing.assert_close(a.diag(), torch.zeros(3, dtype=torch.float64))
        a[0, 1].backward()
        for value in (field, temperature, threshold):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        singleton = distance_adjacency(torch.ones(1, 3), 3., 1.)
        torch.testing.assert_close(singleton, torch.zeros(1, 1))

    def test_latent_operator_equivariant_to_type_preserving_node_relabeling(self):
        torch.manual_seed(9)
        layer = StructureLayer(8, 2, 4, 3, [4, 3], 0., 'full').double().eval()
        h = torch.randn(7, 8, dtype=torch.float64)
        static = (distance_adjacency(h[:4], 3., 1.), distance_adjacency(h[4:], 3., 1.))
        permutation = torch.tensor([2, 0, 3, 1, 6, 4, 5])
        inverse = torch.argsort(permutation)
        first = layer.latent(h, static)
        second = layer.latent(h[permutation], static)[inverse]
        torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-12)

    def test_detached_control_has_identical_forward_but_removes_structure_gradient(self):
        torch.manual_seed(4)
        full = StructureLayer(8, 2, 4, 3, [5], 0., 'full').double()
        detached = StructureLayer(8, 2, 4, 3, [5], 0., 'detached').double()
        detached.load_state_dict(full.state_dict())
        x = torch.randn(5, 8, dtype=torch.float64)
        a, b = full.latent(x, ()), detached.latent(x, ())
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        coefficients = torch.arange(40, dtype=torch.float64).reshape(5, 8)
        (a * coefficients).sum().backward()
        (b * coefficients).sum().backward()
        self.assertGreater(full.log_inverse_scale.grad.abs().sum().item(), 1e-10)
        self.assertIsNone(detached.log_inverse_scale.grad)
        self.assertGreater(detached.metric[0].weight.grad.abs().sum().item(), 1e-10)
        self.assertGreater((full.metric[0].weight.grad - detached.metric[0].weight.grad).abs().sum().item(), 1e-10)

    def test_physical_query_removed_and_all_arms_have_identical_initial_parameters(self):
        rng = np.random.default_rng(13)
        a = dict(node_counts=np.array([3, 2]), nd=np.array(2), pair_offset=np.array([0, 3]),
            pairs_train=np.array([[0, 1, 2], [0, 1, 1]]), y_train=np.array([1, 1, 0]),
            folds=np.array([0, 1, 0]), edges=np.array([[0, 1, 0, 1], [3, 4, 1, 2]]),
            relation=np.array([0, 0, 1, 1]), target_codes=np.array([0, 3, -1, -1]),
            x_0=rng.normal(size=(3, 2)).astype(np.float32),
            x_1=rng.normal(size=(2, 4)).astype(np.float32))
        p = dict(width=8, depth=2, heads=2, metric_width=3, dropout=0., folds=2)
        states = []
        for kind in KINDS:
            torch.manual_seed(7)
            model, xs, graphs, _ = setup(a, p, kind, 'cpu')
            states.append(model.state_dict())
            arcs, _, counts = graphs[0][0]
            self.assertEqual(counts[0, 0].item(), 0)
            self.assertFalse(((arcs[0] == 0) & (arcs[1] == 3)).any())
            self.assertFalse(((arcs[0] == 3) & (arcs[1] == 0)).any())
            self.assertEqual([tuple(g.shape) for g in graphs[0][1]], [(3, 3), (2, 2)])
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
