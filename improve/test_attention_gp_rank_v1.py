"""Independent covariance, conditioning, and query-isolation checks."""
import unittest

import numpy as np
from scipy import sparse
import torch

from improve import attention_gp_rank_v1 as core


class AttentionGPTests(unittest.TestCase):
    def test_covariance_against_edge_pair_definition(self):
        rng = np.random.default_rng(1148)
        x = rng.normal(size=(6, 4))
        k = x @ x.T + np.eye(6)
        a = rng.uniform(size=(6, 6)) * (rng.uniform(size=(6, 6)) > .5)
        oracle = np.zeros((6, 6))
        for u in range(6):
            for v in range(6):
                for i in range(6):
                    for j in range(6):
                        oracle[u, v] += a[u, i] * a[v, j] * k[i, j] * (k[u, v] + k[i, j])
        actual = core.spatial_covariance(core.torch_operator(sparse.csr_matrix(a), 'cpu'), torch.from_numpy(k), 'full')
        np.testing.assert_allclose(actual.numpy(), oracle, atol=1e-12, rtol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(actual).min(), -1e-9)

    def test_kernel_conditioning_matches_primal_features(self):
        rng = np.random.default_rng(2919)
        xr, xd = rng.normal(size=(7, 3)), rng.normal(size=(5, 2))
        pairs = np.array([[0, 0, 2, 3, 5, 6], [1, 3, 2, 0, 4, 2]])
        y = np.array([0., 1., 1., 0., 1., 0.])
        z = np.einsum('ij,ik->ijk', xr[pairs[0]], xd[pairs[1]]).reshape(6, -1)
        beta = np.linalg.solve(z.T @ z + 10.*np.eye(z.shape[1]), z.T @ (2*y-1))
        full = np.einsum('ri,dj->rdij', xr, xd).reshape(7, 5, -1) @ beta
        actual, _, error = core.condition(torch.from_numpy(xr @ xr.T), torch.from_numpy(xd @ xd.T), pairs, y)
        np.testing.assert_allclose(actual.numpy(), full, atol=1e-12, rtol=1e-12)
        self.assertLess(error, 1e-10)

    def test_relu_normalization_and_permutation(self):
        rng = np.random.default_rng(682)
        x = rng.normal(size=(8, 5))
        k = torch.from_numpy(x @ x.T + np.eye(8))
        a = sparse.csr_matrix(np.eye(8) + np.ones((8, 8))*.1)
        first = core.normalize(core.relu_covariance(core.spatial_covariance(core.torch_operator(a, 'cpu'), k, 'full')))
        order = rng.permutation(8)
        second = core.normalize(core.relu_covariance(core.spatial_covariance(core.torch_operator(a[order][:, order], 'cpu'), k[order][:, order], 'full')))
        np.testing.assert_allclose(first[order][:, order], second, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(torch.diag(first), 1., atol=1e-12)
        self.assertGreater(float(torch.linalg.eigvalsh(first).min()), -1e-9)

    def test_remove_queries_before_degrees_and_reverse_edges(self):
        a = dict(pairs_train=np.array([[0, 1], [0, 1]]), y_train=np.array([1, 1]),
                 nd=np.array(2), folds=np.array([0, 1]), node_counts=np.array([0, 2, 2, 0]),
                 pair_offset=np.array([0, 2]), edges=np.array([[0, 1], [2, 3]]),
                 relation=np.array([0, 0]), target_codes=np.array([0, 3]))
        graph = core.support_operator(a, [0]).toarray()
        expected = np.eye(4)
        expected[1, 1] = expected[3, 3] = expected[1, 3] = expected[3, 1] = .5
        np.testing.assert_allclose(graph, expected, atol=1e-12)
        with self.assertRaises(ValueError):
            core.support_operator(dict(a, y_val=np.ones(2)), [0])


if __name__ == '__main__':
    unittest.main()
