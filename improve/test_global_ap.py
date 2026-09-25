import unittest

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from improve.global_ap_modules import positive_anchors, smooth_ap_loss


class GlobalAPTests(unittest.TestCase):
    def test_low_temperature_recovers_ap_without_ties(self):
        scores = torch.tensor([.9, .7, .5, .3, .1], dtype=torch.float64)
        labels = torch.tensor([1., 0., 1., 0., 1.], dtype=torch.float64)
        anchors = positive_anchors(labels, 20, 1)
        actual = 1. - float(smooth_ap_loss(scores, labels, anchors, .0001))
        self.assertAlmostEqual(actual, average_precision_score(labels.numpy(), scores.numpy()), places=12)

    def test_better_ordering_reduces_loss(self):
        labels = torch.tensor([1., 1., 0., 0.])
        anchors = positive_anchors(labels, 4, 1)
        good = torch.tensor([.8, .7, .2, .1])
        bad = torch.tensor([.2, .1, .8, .7])
        self.assertLess(float(smooth_ap_loss(good, labels, anchors)), float(smooth_ap_loss(bad, labels, anchors)))

    def test_ties_and_single_class_are_finite(self):
        scores = torch.zeros(6, requires_grad=True)
        labels = torch.tensor([1., 1., 0., 0., 0., 0.])
        loss = smooth_ap_loss(scores, labels, [0, 1])
        self.assertAlmostEqual(float(loss), 1.-3./7., places=6)
        for y in (torch.zeros(6), torch.ones(6)):
            value = smooth_ap_loss(scores, y, positive_anchors(y, 6, 0))
            self.assertAlmostEqual(float(value), 0., places=6)
            self.assertTrue(value.requires_grad)

    def test_all_references_contribute_and_gradients_exist(self):
        scores = torch.tensor([.6, .5, .4, .3], requires_grad=True)
        labels = torch.tensor([1., 0., 1., 0.])
        loss = smooth_ap_loss(scores, labels, [0], temperature=.2)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue((scores.grad[labels == 0] > 0).all())
        self.assertNotEqual(float(scores.grad[-1]), 0.)

    def test_translation_and_permutation_invariance(self):
        scores = torch.tensor([.8, .4, .1, .5], dtype=torch.float64)
        labels = torch.tensor([1., 0., 1., 0.], dtype=torch.float64)
        before = smooth_ap_loss(scores, labels, positive_anchors(labels, 4, 0))
        shifted = smooth_ap_loss(scores + 9., labels, positive_anchors(labels, 4, 0))
        order = torch.tensor([3, 0, 2, 1])
        after = smooth_ap_loss(scores[order], labels[order], positive_anchors(labels[order], 4, 0))
        torch.testing.assert_close(before, shifted)
        torch.testing.assert_close(before, after)

    def test_anchor_sampling_does_not_consume_global_rng(self):
        labels = np.array([0., 1., 1.]*40)
        np.random.seed(7)
        expected = np.random.rand(5)
        np.random.seed(7)
        a = positive_anchors(labels, 12, 100)
        actual = np.random.rand(5)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(a, positive_anchors(labels, 12, 100))
        self.assertEqual(len(a), 12)
        self.assertTrue((labels[a] == 1).all())
        self.assertEqual(len(np.unique(a)), 12)

    def test_invalid_or_nonpositive_anchors_are_rejected(self):
        scores, labels = torch.tensor([.6, .2]), torch.tensor([1., 0.])
        for anchors in ([1], [0, 0], [], [2], [-1]):
            with self.assertRaises(ValueError):
                smooth_ap_loss(scores, labels, anchors)
        with self.assertRaises(ValueError):
            smooth_ap_loss(scores, labels, [0], 0.)


if __name__ == '__main__':
    unittest.main()
