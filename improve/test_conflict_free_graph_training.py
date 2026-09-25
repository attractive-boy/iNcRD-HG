import unittest

import numpy as np
import torch

from improve.conflict_free_graph_training import (combine_gradients, add_secondary_training,
    context, SECONDARY_KEYS, cosine_rows, setup, parameters, KINDS)


def toy_tasks():
    rng = np.random.default_rng(3)
    shared = dict(node_counts=np.array([3, 2, 2]), nd=np.array(2),
        edges=np.array([[0, 1, 2, 3, 4, 0, 1], [5, 6, 6, 5, 6, 1, 2]]),
        relation=np.array([0, 0, 0, 1, 1, 2, 2]),
        offset_nodes=np.zeros((2, 7)), offset_intercepts=np.zeros(2),
        x_0=rng.normal(size=(3, 2)).astype(np.float32),
        x_1=rng.normal(size=(2, 2)).astype(np.float32),
        x_2=rng.normal(size=(2, 3)).astype(np.float32))
    main = dict(shared, pairs_train=np.array([[0, 1], [0, 1]]), y_train=np.array([1., 1.]),
                folds=np.array([0, 1]), pair_offset=np.array([3, 5]),
                target_codes=np.array([-1, -1, -1, 0, 3, -1, -1]))
    secondary = dict(shared, pairs_train=np.array([[0, 1], [0, 1]]), y_train=np.array([1., 1.]),
                     folds=np.array([0, 1]), pair_offset=np.array([0, 5]),
                     target_codes=np.array([0, 3, -1, -1, -1, -1, -1]))
    secondary = {k: v for k, v in secondary.items() if k in SECONDARY_KEYS}
    return main, secondary


class ConflictFreeTests(unittest.TestCase):
    def test_config_matches_pseudoinverse_and_equal_cosines(self):
        torch.manual_seed(12)
        gradients = torch.randn(2, 17, dtype=torch.float64)
        gradients[0] *= 4
        units = gradients / gradients.norm(dim=1, keepdim=True)
        direction = torch.linalg.pinv(units) @ torch.ones(2, dtype=torch.float64)
        direction /= direction.norm()
        expected = (gradients @ direction).sum() * direction
        actual = combine_gradients(gradients, 'full')
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        projections = cosine_rows(gradients, actual)
        self.assertGreater(projections.min().item(), 0)
        torch.testing.assert_close(projections[0], projections[1], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual, combine_gradients(gradients.flip(0), 'full'))
        rotation, _ = torch.linalg.qr(torch.randn(17, 17, dtype=torch.float64))
        torch.testing.assert_close(actual @ rotation, combine_gradients(gradients @ rotation, 'full'), atol=1e-12, rtol=1e-12)

    def test_conflicts_zeros_parallel_and_pcgrad_definition(self):
        g = torch.tensor([[1., 0.], [-1., 2.]], dtype=torch.float64)
        pcgrad = combine_gradients(g, 'pcgrad')
        torch.testing.assert_close(pcgrad, torch.tensor([.8, 2.4], dtype=torch.float64))
        self.assertGreater(cosine_rows(g, pcgrad).min().item(), 0)
        config = combine_gradients(g, 'full')
        self.assertGreater((config / config.norm() - pcgrad / pcgrad.norm()).abs().max().item(), .1)
        opposite = torch.tensor([[2., 0.], [-1., 0.]], dtype=torch.float64)
        torch.testing.assert_close(combine_gradients(opposite, 'full'), torch.zeros(2, dtype=torch.float64))
        for values in ([[0., 0.], [3., 1.]], [[2., 0.], [1., 0.]], [[0., 0.], [0., 0.]]):
            x = torch.tensor(values, dtype=torch.float64)
            torch.testing.assert_close(combine_gradients(x, 'full'), x.sum(0))
        with self.assertRaises(FloatingPointError):
            combine_gradients(torch.full((2, 3), float('nan')), 'full')

    def test_secondary_inputs_reject_outcomes_and_graph_masks_both_tasks(self):
        main, secondary = toy_tasks()
        original_edges = main['edges'].copy()
        with self.assertRaisesRegex(ValueError, 'training inputs only'):
            add_secondary_training(main, dict(secondary, y_val=np.array([1.])))
        a = add_secondary_training(main, secondary)
        np.testing.assert_array_equal(main['edges'], original_edges)
        np.testing.assert_array_equal(main['x_0'], a['x_0'])
        # The extra historical secondary positive (2,6) is discarded.
        self.assertFalse(((a['edges'][0] == 2) & (a['edges'][1] == 6)).any())
        arcs, _, counts = context(a, 0, 'cpu')
        for u, v in [(3, 5), (0, 5)]:
            self.assertFalse(((arcs[0] == u) & (arcs[1] == v)).any())
            self.assertFalse(((arcs[0] == v) & (arcs[1] == u)).any())
        self.assertEqual(counts[3, 1].item(), 0)
        self.assertEqual(counts[0, 0].item(), 0)
        self.assertTrue(((arcs[0] == 1) & (arcs[1] == 6)).any())

    def test_graph_models_identical_and_query_permutation_safe(self):
        a = add_secondary_training(*toy_tasks())
        p = dict(parameters(), width=8, depth=1, heads=2, dropout=0., folds=2)
        states = []
        prediction = None
        for kind in KINDS:
            torch.manual_seed(10)
            model, xs, graphs, _ = setup(a, p, kind, 'cpu')
            states.append(model.state_dict())
            pairs = torch.tensor([[3, 4, 0, 1], [5, 6, 5, 6]])
            current = model(xs, graphs[0], pairs)
            torch.testing.assert_close(current, model(xs, graphs[0], pairs.flip(1)).flip(0))
            if prediction is not None:
                torch.testing.assert_close(current, prediction, atol=0, rtol=0)
            prediction = current
        for state in states[1:]:
            for key in state:
                torch.testing.assert_close(state[key], states[0][key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
