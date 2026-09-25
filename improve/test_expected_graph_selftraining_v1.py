"""Support-only proposals, equal-weight correction control, and weighted graph checks."""
import unittest

import numpy as np
import torch

from improve import expected_graph_selftraining_v1 as core
from improve.test_nested_anchor_v2 import fixture


class ExpectedGraphTests(unittest.TestCase):
    def test_confidence_selection_retains_ties_and_rejects_bad_probabilities(self):
        selected, cutoff = core.confidence_selection(np.array([.1, .8, .8, .3]), 1)
        np.testing.assert_array_equal(selected, [False, True, True, False])
        self.assertEqual(cutoff, .8)
        selected, cutoff = core.confidence_selection(np.array([]), 0)
        self.assertEqual(len(selected), 0)
        self.assertIsNone(cutoff)
        with self.assertRaises(ValueError):
            core.confidence_selection(np.array([1.1]), 1)

    def test_query_labels_cannot_enter_candidates_or_expected_graph(self):
        a, _ = fixture()
        p = core.parameters()
        p.update(width=8, dropout=0., edge_increment=.5)
        omitted = np.flatnonzero(a['folds'] == 0)
        original = core.ExpectedContext(a, omitted, p, 'cpu')
        b = {k: v.copy() for k, v in a.items()}
        b['y_train'][omitted] = 1 - b['y_train'][omitted]
        other = core.ExpectedContext(b, omitted, p, 'cpu')
        np.testing.assert_array_equal(original.candidates, other.candidates)
        self.assertEqual((original.adjacency != other.adjacency).nnz, 0)
        xs = [torch.as_tensor(a['x_' + str(i)]) for i in range(4)]
        model = core.ExpectedGraphModel([4] * 4, p)
        report = original.refresh(model, xs, 1, p, 'full')
        other.refresh(model, xs, 1, p, 'full')
        self.assertGreater(report['actual_virtual_edges'], 0)
        np.testing.assert_array_equal(original.added, other.added)
        np.testing.assert_array_equal(original.added_weights, other.added_weights)
        np.testing.assert_array_equal(original.query_teacher, other.query_teacher)
        self.assertFalse(set(map(tuple, original.added.T)) & set(map(tuple, original.query_pairs.T)))
        edge_set = set(map(tuple, original.expected.edges.T.tolist()))
        for pair in original.query_pairs.T:
            self.assertNotIn(tuple(pair), edge_set)
            self.assertNotIn(tuple(pair[::-1]), edge_set)

    def test_corrective_replacement_keeps_weight_and_changes_target_gradient(self):
        expected = torch.tensor([.3, -.7], dtype=torch.float64, requires_grad=True)
        original = torch.tensor([-.2, .6], dtype=torch.float64, requires_grad=True)
        pseudo, labels = torch.tensor([1., .8]), torch.tensor([1., 0.])
        auxiliary, expected_loss, correction = core.auxiliary_objective(expected, original, pseudo, labels, 'full')
        torch.testing.assert_close(auxiliary, .5 * (expected_loss + correction), atol=0, rtol=0)
        auxiliary.backward(retain_graph=True)
        self.assertTrue(torch.isfinite(original.grad).all())
        self.assertGreater(float(original.grad.abs().sum()), 0.)
        original.grad = None
        auxiliary, _, _ = core.auxiliary_objective(expected, original, pseudo, labels, 'no_correction')
        auxiliary.backward()
        self.assertIsNone(original.grad)
        torch.testing.assert_close(auxiliary, expected_loss, atol=0, rtol=0)

    def test_original_graph_inference_and_zero_initial_residual(self):
        a, _ = fixture()
        p = core.parameters()
        p.update(width=8, dropout=0.)
        model, xs, graphs, folds = core.setup(a, p, 'full', 'cpu')
        query = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])[:, folds[0]]
        model.eval()
        scores = model(xs, graphs[0], query)
        torch.testing.assert_close(scores, torch.zeros_like(scores), atol=0, rtol=0)
        torch.nn.init.normal_(model.residual.weight, std=.1)
        before = model(xs, graphs[0], query)
        graphs[0].refresh(model, xs, 5, p, 'full')
        after = model(xs, graphs[0], query)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        reversed_scores = model(xs, graphs[0], query.flip(1)).flip(0)
        torch.testing.assert_close(after, reversed_scores, atol=1e-7, rtol=1e-5)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
