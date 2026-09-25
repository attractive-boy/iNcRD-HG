import unittest
import torch

from improve.tabicl_care_rank_modules import ScaleStableAdapter, teacher_order_loss


class StableControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(62)
        self.model = ScaleStableAdapter()
        with torch.no_grad():
            self.model.controller[-1].weight.normal_()
        self.relation = torch.randn(5, 17, 5)
        self.relation[..., 3:] = (self.relation[..., 3:] > 1.).float()
        self.evidence = torch.rand(5, 6)
        self.logits = torch.randn(3, 5, 2)

    def test_replicated_support_rates_and_class_symmetry(self):
        first = self.model(self.relation, self.evidence, self.logits, 2.)
        repeated = self.relation.repeat(1, 3, 1)
        changed_counts = self.evidence.clone()
        changed_counts[:, :2] = .99
        second = self.model(repeated, changed_counts, self.logits, 2.)
        torch.testing.assert_close(second, first.repeat(1, 1, 3), atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(first, self.model(self.relation, self.evidence, self.logits.flip(-1), 2.))
        self.assertLessEqual(first.abs().max().item(), 4.)

    def test_zero_start_and_query_independence(self):
        blank = ScaleStableAdapter()
        self.assertEqual(torch.count_nonzero(blank(self.relation, self.evidence, self.logits)).item(), 0)
        full = self.model(self.relation, self.evidence, self.logits)
        one = self.model(self.relation[[1]], self.evidence[[1]], self.logits[:, [1]])
        torch.testing.assert_close(one, full[:, [1]])

    def test_teacher_order_penalty_only_for_deterioration(self):
        labels, permutation = torch.tensor([1, 0, 1, 0]), torch.tensor([[0, 1]])
        teacher = torch.tensor([[[0., .6], [0., .2], [0., -.5], [0., .0]]])
        loss, count = teacher_order_loss(teacher, teacher, labels, permutation)
        self.assertEqual(float(loss), 0.)
        self.assertGreater(count, 0)
        changed = teacher.clone().requires_grad_()
        shift = torch.tensor([[[0., -.7], [0., .0], [0., .0], [0., .0]]])
        penalty, _ = teacher_order_loss(changed + shift, teacher, labels, permutation)
        self.assertGreater(float(penalty), 0.)
        penalty.backward()
        self.assertTrue(torch.isfinite(changed.grad).all())
        # Improving an already incorrect pair is not penalized by the teacher.
        better = teacher.clone()
        better[0, 2, 1] = .8
        self.assertEqual(float(teacher_order_loss(better, teacher, labels, permutation)[0]), 0.)


if __name__ == '__main__':
    unittest.main()
