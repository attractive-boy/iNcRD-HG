"""Checks specific to the separately registered bounded refinement."""
import unittest
import torch

from improve.tabicl_care_modules import ConditionalRelationAdapter
from improve.tabicl_care_refine import RefinedAdapter


class RefinedAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(36)
        self.original = ConditionalRelationAdapter()
        with torch.no_grad():
            self.original.controller[-1].weight.normal_()
        self.relation = torch.randn(7, 11, 5)
        self.evidence = torch.rand(7, 6)
        self.logits = torch.randn(3, 7, 2)

    def test_exact_first_trial_recovery(self):
        model = RefinedAdapter(self.original, 1.)
        expected = self.original(self.relation, self.evidence, self.logits, 1.)
        torch.testing.assert_close(model(self.relation, self.evidence, self.logits), expected, atol=0., rtol=0.)
        torch.testing.assert_close(model(self.relation, self.evidence, self.logits, 2.), 2. * expected, atol=0., rtol=0.)

    def test_soft_gate_bound_extreme_logits_and_query_independence(self):
        model = RefinedAdapter(self.original, .5)
        value = model(self.relation, self.evidence, self.logits, 4.)
        self.assertLessEqual(value.abs().max().item(), 8.)
        single = model(self.relation[[2]], self.evidence[[2]], self.logits[:, [2]], 4.)
        torch.testing.assert_close(value[:, [2]], single)
        torch.testing.assert_close(value, model(self.relation, self.evidence, self.logits.flip(-1), 4.))
        extreme = torch.tensor([10000., -10000.]).expand_as(self.logits)
        result = model(self.relation, self.evidence, extreme, 4.)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(torch.count_nonzero(result).item(), 0)

    def test_unregistered_values_rejected(self):
        with self.assertRaises(ValueError):
            RefinedAdapter(self.original, 0.)
        with self.assertRaises(ValueError):
            RefinedAdapter(self.original)(self.relation, self.evidence, self.logits, 5.)


if __name__ == '__main__':
    unittest.main()
