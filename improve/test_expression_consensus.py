"""Mechanism and protocol invariants for the cell-subset experiment."""
import unittest

import numpy as np
import torch

from improve.expression_consensus_modules import (
    complementary_partitions, consistency_loss, edge_reliability,
    expression_views, normalize_incoming, shuffle_edge_reliability,
    shuffle_node_reliability)
from improve.expression_consensus import check_no_test


def identity_parameters(n_features, n_components=3):
    rng = np.random.default_rng(1)
    return dict(input_mean=np.zeros(n_features), input_scale=np.ones(n_features),
                pca_mean=np.zeros(n_features), pca_components=rng.normal(size=(n_components, n_features)),
                component_mean=rng.normal(size=n_components), component_scale=np.arange(1, n_components + 1))


class ExpressionConsensusTests(unittest.TestCase):
    def test_complementary_views_recover_projection_even_and_odd(self):
        for n in (10, 11):
            raw = np.random.default_rng(0).normal(size=(7, n))
            partitions = complementary_partitions(n, repeats=4)
            full, views, confidence, sensitivity = expression_views(raw, identity_parameters(n), partitions)
            for i, (a, b) in enumerate(partitions):
                self.assertEqual(len(np.intersect1d(a, b)), 0)
                np.testing.assert_allclose((len(a) * views[2*i] + len(b) * views[2*i+1]) / n,
                                           full, rtol=2e-6, atol=2e-6)
            self.assertTrue(np.all((confidence > 0) & (confidence <= 1)))
            self.assertTrue(np.all(sensitivity >= 0))

    def test_disagreement_detects_cell_localized_expression(self):
        # Two nodes have the SAME full projected feature. One depends on one
        # cell line and the other has identical support in every cell line.
        raw = np.ones((2, 100))
        raw[1] = 0
        raw[1, 0] = 100
        params = identity_parameters(100, 1)
        params.update(pca_components=np.ones((1, 100)) / 10,
                      component_mean=np.zeros(1), component_scale=np.ones(1))
        full, _, confidence, _ = expression_views(raw, params, complementary_partitions(100))
        np.testing.assert_allclose(full[0], full[1])
        self.assertAlmostEqual(float(confidence[0]), 1.0)
        self.assertLess(float(confidence[1]), .02)

    def test_invalid_or_overlapping_views_are_rejected(self):
        with self.assertRaises(ValueError):
            expression_views(np.ones((3, 6)), identity_parameters(6), [(np.array([0, 1, 2]), np.array([2, 4, 5]))])

    def test_negative_correlation_is_supported_without_becoming_negative_weight(self):
        x = np.arange(12, dtype=float)
        raw = np.stack([x, -x, x + 5])
        edges = np.array([[0, 1, 0, 2, 0, 1, 2], [1, 0, 2, 0, 0, 1, 2]])
        similarity = np.corrcoef(raw)
        factors, details = edge_reliability(raw, edges, similarity, complementary_partitions(12, repeats=2))
        np.testing.assert_allclose(factors, 1., atol=1e-6)
        np.testing.assert_allclose(details['sign_agreement'], 1.)
        self.assertTrue(np.all(factors > 0))

    def test_incoming_mass_and_self_messages_are_preserved(self):
        edges = np.array([[0, 1, 2, 2, 0, 1], [1, 0, 0, 1, 0, 1]])
        strengths = np.array([.1, .2, .9, .8, .001, 1000.])
        weights = normalize_incoming(edges, strengths, n_nodes=4)
        active = edges[0] != edges[1]
        np.testing.assert_array_equal(weights[~active], 1.)
        np.testing.assert_allclose(np.bincount(edges[1, active], weights=weights[active], minlength=4),
                                   np.bincount(edges[1, active], minlength=4), atol=1e-6)
        np.testing.assert_allclose(normalize_incoming(edges, 10 * strengths, 4), weights)
        with self.assertRaises(ValueError):
            normalize_incoming(edges, np.zeros(len(strengths)), 4)

    def test_shuffles_preserve_reciprocity_and_distribution(self):
        pairs = np.array([(i, j) for i in range(10) for j in range(i + 1, 10)])
        edges = np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1)
        matrix = np.full((10, 10), .5)
        original = np.tile(np.linspace(.1, .9, len(pairs)), 2).astype(np.float32)
        shuffled = shuffle_edge_reliability(edges, matrix, original)
        np.testing.assert_array_equal(shuffled[:len(pairs)], shuffled[len(pairs):])
        np.testing.assert_array_equal(np.sort(original), np.sort(shuffled))
        self.assertFalse(np.array_equal(shuffled, original))
        q = np.arange(1, 11, dtype=float)
        node_shuffle = shuffle_node_reliability(q, np.arange(7))
        np.testing.assert_array_equal(np.sort(node_shuffle[:7]), q[:7])
        np.testing.assert_array_equal(np.sort(node_shuffle[7:]), q[7:])

    def test_consistency_uses_detached_teacher_and_equal_weight_budget(self):
        student = torch.tensor([-.4, .2, 1.], requires_grad=True)
        teacher = torch.tensor([.3, -.3, 1.2], requires_grad=True)
        q = torch.tensor([.1, .5, 1.])
        loss = consistency_loss(student, teacher, q)
        self.assertGreater(float(loss), 0.)
        torch.testing.assert_close(loss, consistency_loss(student, teacher, 7*q))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.)
        self.assertAlmostEqual(float(consistency_loss(student, student)), 0., places=7)
        torch.testing.assert_close(consistency_loss(student, teacher),
                                   consistency_loss(student, teacher, torch.ones_like(q)))

    def test_test_arrays_cannot_enter_training_runner(self):
        arrays = {k: np.zeros(2) for k in ('pairs_train', 'pairs_val', 'y_train', 'y_val')}
        check_no_test(arrays)
        with self.assertRaises(ValueError):
            check_no_test(dict(arrays, y_test=np.zeros(2)))


if __name__ == '__main__':
    unittest.main()
