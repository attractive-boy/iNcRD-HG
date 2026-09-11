"""Numerical and information-boundary checks for the new decoder prototype."""
import unittest

import torch

from improve.support_operator_modules import (
    SupportOperatorDecoder, exclude_support_pairs, regularized_operator)


class SupportOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.z = {'lncRNA': torch.randn(8, 8, dtype=torch.double),
                  'drug': torch.randn(4, 8, dtype=torch.double)}
        self.query = torch.tensor([[6, 7], [0, 3]])
        self.support = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 3]])
        self.labels = torch.tensor([1., 0., 0., 1., 1., 0.], dtype=torch.double)
        self.model = SupportOperatorDecoder(8, 'lncRNA', rank=3).double()

    def test_excludes_positive_negative_and_duplicate_query_rows(self):
        pairs = torch.tensor([[0, 0, 1, 2, 3], [0, 0, 1, 1, 0]])
        labels = torch.tensor([1., 1., 0., 1., 0.])
        query = torch.tensor([[0, 1], [0, 1]])
        kept, y, mask = exclude_support_pairs(pairs, labels, query, 2)
        self.assertTrue(torch.equal(kept, pairs[:, 3:]))
        self.assertTrue(torch.equal(y, labels[3:]))
        self.assertEqual(mask.tolist(), [False, False, False, True, True])

    def test_empty_support_exactly_uses_prior(self):
        empty = torch.empty((2, 0), dtype=torch.long)
        labels = torch.empty(0, dtype=torch.double)
        adaptive, detail = self.model(self.z, self.query, empty, labels, return_details=True)
        prior = self.model(self.z, self.query, empty, labels, mode='prior')
        torch.testing.assert_close(adaptive, prior, rtol=0, atol=0)
        torch.testing.assert_close(detail['operator'], detail['prior'], rtol=0, atol=0)

    def test_unsupported_feature_directions_preserve_prior(self):
        gram = torch.diag(torch.tensor([3., 0., 0.], dtype=torch.double))[None]
        rhs = torch.tensor([[6., 0., 0.]], dtype=torch.double)
        prior = torch.tensor([[.4, -.7, 2.]], dtype=torch.double)
        w = regularized_operator(gram, rhs, prior, torch.tensor([.5], dtype=torch.double))
        torch.testing.assert_close(w[0, 1:], prior[0, 1:])
        self.assertAlmostEqual(w[0, 0].item(), (6 + .5 * .4) / 3.5)

    def test_solver_gradient_matches_finite_differences(self):
        x = torch.randn(2, 4, 3, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
        prior = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        raw = torch.zeros(2, dtype=torch.double, requires_grad=True)
        def solve(a, b, c, d):
            return regularized_operator(a.transpose(1, 2) @ a,
                                        (a.transpose(1, 2) @ b[:, :, None]).squeeze(-1),
                                        c, torch.nn.functional.softplus(d) + .01)
        self.assertTrue(torch.autograd.gradcheck(solve, (x, y, prior, raw)))

    def test_support_permutation_invariance(self):
        order = torch.tensor([3, 1, 5, 0, 2, 4])
        first = self.model(self.z, self.query, self.support, self.labels)
        second = self.model(self.z, self.query, self.support[:, order], self.labels[order])
        torch.testing.assert_close(first, second)

    def test_query_labels_cannot_change_prediction(self):
        extended = torch.cat((self.support, self.query), dim=1)
        first = torch.cat((self.labels, torch.tensor([1., 0.], dtype=torch.double)))
        second = torch.cat((self.labels, torch.tensor([0., 1.], dtype=torch.double)))
        a = self.model(self.z, self.query, extended, first)
        b = self.model(self.z, self.query, extended, second)
        c = self.model(self.z, self.query, self.support, self.labels)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_support_labels_matter_and_gradients_reach_all_mechanisms(self):
        prediction = self.model(self.z, self.query, self.support, self.labels)
        permuted = self.model(self.z, self.query, self.support, 1 - self.labels)
        self.assertGreater(float((prediction - permuted).abs().max()), 1e-4)
        prediction.square().sum().backward()
        for prefix in ('rna_basis', 'drug_prior', 'drug_key', 'ridge_head', 'raw_temperature'):
            gradients = [p.grad for n, p in self.model.named_parameters() if n.startswith(prefix)]
            self.assertTrue(gradients)
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 1e-9)


if __name__ == '__main__':
    unittest.main()
