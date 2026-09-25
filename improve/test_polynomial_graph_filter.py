import unittest

import numpy as np
import torch

from improve.polynomial_graph_filter import normalized_operator, chebyshev_tokens, polynomial_attention, context, setup


class PolynomialFilterTests(unittest.TestCase):
    def test_operator_deduplicates_and_spectrum_is_bounded(self):
        edges = np.array([[0, 1, 0, 1, 2, 3], [1, 2, 1, 1, 3, 0]])
        operator = normalized_operator(edges, 5, 'cpu').to_dense()
        torch.testing.assert_close(operator, operator.T, atol=0, rtol=0)
        self.assertEqual(torch.diag(operator).abs().sum().item(), 0)
        self.assertLessEqual(torch.linalg.eigvalsh(operator).abs().max().item(), 1.000001)
        torch.testing.assert_close(operator[0, 1], torch.tensor(.5), atol=0, rtol=0)
        self.assertEqual(operator[4].sum().item(), 0)

    def test_chebyshev_response_and_gradient_match_an_eigenmode(self):
        op = normalized_operator(np.array([[0], [1]]), 2, 'cpu')
        signal = torch.tensor([[1.], [-1.]], requires_grad=True)
        tokens = chebyshev_tokens(op, signal, 4)
        for j in range(5):
            torch.testing.assert_close(tokens[:, j], signal * (-1)**j, atol=0, rtol=0)
        tokens[:, 4].sum().backward()
        torch.testing.assert_close(signal.grad, torch.ones_like(signal), atol=0, rtol=0)
        no_graph = chebyshev_tokens(op, signal, 4, True)
        self.assertEqual(torch.count_nonzero(no_graph[:, 1:]).item(), 0)

    def test_node_specific_signs_and_type_shared_null(self):
        query = torch.tensor([[[[1.], [1.]]], [[[1.], [1.]]]])
        key = torch.tensor([[[[1.], [-1.]]], [[[-1.], [1.]]]])
        value = torch.tensor([[[[1.], [2.]]], [[[1.], [2.]]]])
        bias = torch.ones(1, 2)
        _, full = polynomial_attention(query, key, value, bias, 'full', [2])
        _, shared = polynomial_attention(query, key, value, bias, 'type_shared', [2])
        _, softmax = polynomial_attention(query, key, value, bias, 'softmax', [2])
        self.assertLess(full[0, 0, 0, 1].item(), 0)
        self.assertGreater(full[1, 0, 0, 1].item(), 0)
        torch.testing.assert_close(shared[0], shared[1], atol=0, rtol=0)
        self.assertTrue((softmax >= 0).all())
        changed = polynomial_attention(query.flip(0), key.flip(0), value.flip(0), bias, 'full', [2])[0]
        torch.testing.assert_close(changed.flip(0), polynomial_attention(query, key, value, bias, 'full', [2])[0])

    def test_query_deletion_and_all_arms_have_matched_initialization(self):
        a = dict(node_counts=np.array([3, 2]), nd=np.array(2), pair_offset=np.array([0, 3]),
            pairs_train=np.array([[0, 1, 2], [0, 1, 1]]), y_train=np.array([1, 1, 0]),
            folds=np.array([0, 1, 0]), edges=np.array([[0, 1, 0, 1], [3, 4, 1, 2]]),
            relation=np.array([0, 0, 1, 1]), target_codes=np.array([0, 3, -1, -1]),
            x_0=np.ones((3, 2), np.float32), x_1=np.ones((2, 4), np.float32))
        op, counts = context(a, 0, 'cpu')
        self.assertEqual(op.to_dense()[0, 3].item(), 0)
        self.assertEqual(op.to_dense()[3, 0].item(), 0)
        self.assertEqual(counts[0, 0].item(), 0)
        p = dict(width=8, depth=1, order=2, heads=2, dropout=0., folds=2)
        states = []
        for kind in ('full', 'type_shared', 'softmax', 'no_graph'):
            torch.manual_seed(7)
            model, xs, graphs, _ = setup(a, p, kind, 'cpu')
            states.append(model.state_dict())
            pairs = torch.tensor([[0, 1], [3, 4]])
            score = model(xs, graphs[0], pairs)
            torch.testing.assert_close(score, model(xs, graphs[0], pairs.flip(1)).flip(0))
            score.sum().backward()
            self.assertTrue(torch.isfinite(model.project[0].weight.grad).all())
        for state in states[1:]:
            for name in state:
                torch.testing.assert_close(state[name], states[0][name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
