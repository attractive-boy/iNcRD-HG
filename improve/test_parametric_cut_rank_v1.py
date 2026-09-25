"""Independent exhaustive-cut checks for the native solver and sparse adapter."""
import itertools
import unittest

import numpy as np
from scipy import sparse

from improve import parametric_cut_rank_v1 as core


def objective(graph, chosen, value):
    x = np.asarray(chosen, dtype=float)
    return float((graph * abs(x[:, None]-x[None, :])).sum()/2 -
                 value * np.dot(graph.sum(1), x))


class CutChecks(unittest.TestCase):
    def test_native_sequence_matches_exhaustive_optimization(self):
        rng = np.random.default_rng(723)
        w = rng.uniform(.1, 1., size=(7, 7))
        w = (w+w.T)/2
        np.fill_diagonal(w, 0)
        for negative, grid in [([], np.linspace(-1, 0, 19)), ([6], np.linspace(0, 1, 19))]:
            positive = [0]
            bp = core.solve_cuts(sparse.csr_matrix(w), positive, negative, grid)
            free = sorted(set(range(7))-set(positive)-set(negative))
            options = []
            for bits in itertools.product((False, True), repeat=len(free)):
                x = np.zeros(7, dtype=bool); x[positive] = True; x[free] = bits
                options.append(x)
            for i, value in enumerate(grid, 1):
                predicted = bp <= i
                observed = objective(w, predicted, value)
                optimum = min(objective(w, x, value) for x in options)
                self.assertAlmostEqual(observed, optimum, places=5)
            np.testing.assert_array_equal(bp, core.solve_cuts(sparse.csr_matrix(w), positive, negative, grid))

    def test_stage_two_keeps_positive_affinity_and_degree_unaries(self):
        w = np.array([[0., 4., .1, .1], [4., 0., .2, 1.], [.1, .2, 0., 4.], [.1, 1., 4., 0.]])
        bp = core.solve_cuts(sparse.csr_matrix(w), [0], [3], np.array([0., .1, .9]))
        self.assertTrue((bp <= 1)[1])
        self.assertFalse((bp <= 1)[2])
        self.assertTrue((bp <= 3)[1])
        self.assertTrue((bp <= 3)[2])

    def test_cartesian_graph_indexing_and_label_free_construction(self):
        r = np.array([[1., .7, .4], [.7, 1., .2], [.4, .2, 1.]])
        d = np.array([[1., .6], [.6, 1.]])
        graph = core.cartesian_graph(r, d, 1)
        self.assertEqual(graph.shape, (6, 6))
        self.assertEqual((graph-graph.T).nnz, 0)
        self.assertEqual(graph[0, 3], 0.)
        self.assertGreater(graph[0, 1], 0.)
        self.assertGreater(graph[0, 2], 0.)
        self.assertAlmostEqual(float(graph.sum())/6, 2.)

    def test_two_stage_seeds_and_dictionary_lookup_are_reproducible(self):
        x = np.arange(10)
        graph = sparse.csr_matrix(np.exp(-abs(x[:,None]-x[None,:])/2.)-np.eye(10))
        first = core.two_stage(graph, [0, 1], 3, grid_size=41)
        second = core.two_stage(graph, [1, 0], 3, grid_size=41)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertFalse(np.intersect1d(first['positive'], first['negative']).size)
        self.assertTrue((first['full_rank'][first['positive']]==1).all())
        self.assertTrue((first['full_rank'][first['negative']]==0).all())
        ids = np.array([2, 8, 4, 9])
        np.testing.assert_array_equal(first['full_rank'][ids], first['full_rank'][ids[::-1]][::-1])


if __name__ == '__main__':
    unittest.main()
