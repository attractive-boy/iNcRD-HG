"""Information-boundary and numerical checks for local case adaptation."""
import unittest

import torch

from improve.local_support_modules import LocalSupportAdapter, local_newton_step


class LocalSupportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = LocalSupportAdapter(6, 5, 'lncRNA', rank=3).double()
        self.x = {'lncRNA': torch.randn(9, 6, dtype=torch.double),
                  'drug': torch.randn(4, 5, dtype=torch.double)}
        self.query = torch.tensor([[7, 8], [1, 2]])
        self.support = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 0, 2]])
        self.labels = torch.tensor([1., 0., 0., 1., 1., 0.], dtype=torch.double)
        self.prior = torch.tensor([-.2, .1, -.5, 1., -.4, .3], dtype=torch.double)

    def test_zero_residual_gives_zero_update(self):
        features = torch.randn(2, 5, 3, dtype=torch.double)
        probabilities = torch.rand(2, 5, dtype=torch.double)
        weights = torch.softmax(torch.randn(2, 5, dtype=torch.double), dim=-1)
        delta = local_newton_step(features, probabilities, probabilities, weights, .1)
        torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=0)

    def test_newton_gradient_matches_finite_differences(self):
        x = torch.randn(2, 4, 3, dtype=torch.double, requires_grad=True)
        raw = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([[0., 1., 0., 1.], [1., 0., 0., 1.]], dtype=torch.double)
        probabilities = torch.full((2, 4), .4, dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b: local_newton_step(a, labels, probabilities, b.softmax(-1), .2), (x, raw)))

    def test_queries_cannot_supply_their_own_labels_or_prior(self):
        support = torch.cat((self.support, self.query, self.query), dim=1)
        ya = torch.cat((self.labels, torch.tensor([0., 1., 0., 1.], dtype=torch.double)))
        yb = torch.cat((self.labels, torch.tensor([1., 0., 1., 0.], dtype=torch.double)))
        pa = torch.cat((self.prior, torch.ones(4, dtype=torch.double)))
        pb = torch.cat((self.prior, -torch.ones(4, dtype=torch.double)))
        first = self.model(self.x, self.query, support, ya, pa, topk=4)
        second = self.model(self.x, self.query, support, yb, pb, topk=4)
        clean = self.model(self.x, self.query, self.support, self.labels, self.prior, topk=4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, clean, rtol=0, atol=0)

    def test_empty_support_and_disabled_correction_are_exactly_zero(self):
        for intervention in (None, 'no_correction'):
            delta = self.model(self.x, self.query, self.support[:, :0], self.labels[:0],
                               self.prior[:0], intervention=intervention)
            torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=0)
        delta = self.model(self.x, self.query, self.support, self.labels, self.prior, ramp=0)
        torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=0)

    def test_support_permutation_invariance_and_logit_bound(self):
        order = torch.tensor([4, 2, 0, 3, 5, 1])
        a = self.model(self.x, self.query, self.support, self.labels, self.prior, topk=4,
                       max_correction=.2, ramp=.4)
        b = self.model(self.x, self.query, self.support[:, order], self.labels[order], self.prior[order],
                       topk=4, max_correction=.2, ramp=.4)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(bool((a.abs() <= .08).all()))

    def test_cases_change_correction_and_gradients_reach_keys_and_basis(self):
        a = self.model(self.x, self.query, self.support, self.labels, self.prior, topk=4)
        b = self.model(self.x, self.query, self.support, 1 - self.labels, self.prior, topk=4)
        self.assertGreater(float((a - b).abs().max()), 1e-4)
        a.square().sum().backward()
        for prefix in ('rna_key', 'drug_key', 'rna_basis', 'drug_basis'):
            gradients = [p.grad for n, p in self.model.named_parameters() if n.startswith(prefix)]
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 1e-10)

    def test_support_prior_is_stop_gradient(self):
        prior = self.prior.clone().requires_grad_(True)
        self.model(self.x, self.query, self.support, self.labels, prior, topk=4).sum().backward()
        self.assertIsNone(prior.grad)

    def test_same_drug_queries_can_retrieve_different_rna_cases(self):
        self.model.rna_key = torch.nn.Identity()
        self.model.drug_key = torch.nn.Identity()
        rna = torch.zeros(9, 6, dtype=torch.double)
        rna[:, 2] = 1
        rna[0] = rna[7] = torch.tensor([1., 0., 0., 0., 0., 0.], dtype=torch.double)
        rna[1] = rna[8] = torch.tensor([0., 1., 0., 0., 0., 0.], dtype=torch.double)
        x = {'lncRNA': rna, 'drug': torch.ones(4, 5, dtype=torch.double)}
        query = torch.tensor([[7, 8], [1, 1]])
        _, details = self.model(x, query, self.support, self.labels, self.prior,
                                topk=1, return_details=True)
        self.assertEqual(details['selected_pairs'][0, :, 0].tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
