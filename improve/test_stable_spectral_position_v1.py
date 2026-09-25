"""Complete eigenspaces, basis symmetry, dense GIN semantics, and fold isolation."""
import unittest

import numpy as np
import torch

from improve import stable_spectral_position_v1 as spectral
from improve.test_nested_anchor_v2 import fixture


def ingredients():
    torch.manual_seed(962)
    n = 6
    vectors = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64)).Q
    values = torch.tensor([0., .4, .4, 1., 1.3, 2.], dtype=torch.float64)
    adjacency = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n-1):
        adjacency[i, i+1] = adjacency[i+1, i] = 1.
    p = dict(spectral.parameters(), pe_filters=3, psi_width=5, pe_width=5)
    model = spectral.SpectralPosition(p, 'full').double()
    return model, values, vectors, adjacency


class SpectralPositionChecks(unittest.TestCase):
    def test_repeated_eigenspace_rotation_and_eigenvector_order_preserve_values_and_gradients(self):
        model, values, vectors, adjacency = ingredients()
        rotated = vectors.clone()
        q = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
        rotated[:, 1:3] = vectors[:, 1:3] @ q
        actual = model(values, vectors, adjacency)
        equivalent = model(values, rotated, adjacency)
        torch.testing.assert_close(actual, equivalent, atol=1e-11, rtol=0)
        probe = torch.randn_like(actual)
        left = torch.autograd.grad((actual*probe).sum(), tuple(model.parameters()))
        right = torch.autograd.grad((equivalent*probe).sum(), tuple(model.parameters()))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, atol=2e-11, rtol=0)
        permutation = torch.tensor([5, 2, 0, 4, 1, 3])
        signs = torch.tensor([-1, 1, -1, -1, 1, -1], dtype=torch.float64)
        reordered = model(values[permutation], vectors[:, permutation]*signs, adjacency)
        torch.testing.assert_close(actual, reordered, atol=1e-11, rtol=0)

    def test_kernel_matches_explicit_spectral_sum_and_node_permutation(self):
        model, values, vectors, adjacency = ingredients()
        weights = model.coefficients(values)
        explicit = torch.stack([sum(weights[j, k]*torch.outer(vectors[:, j], vectors[:, j])
            for j in range(len(values))) for k in range(weights.shape[1])], dim=-1)
        torch.testing.assert_close(model.kernels(values, vectors), explicit, atol=2e-15, rtol=0)
        permutation = torch.tensor([4, 1, 3, 0, 5, 2])
        actual = model(values, vectors, adjacency)
        permuted = model(values, vectors[permutation], adjacency[permutation][:, permutation])
        torch.testing.assert_close(permuted, actual[permutation], atol=1e-11, rtol=0)

    def test_shared_spectrum_is_weighted_identity_with_matched_capacity_and_rng(self):
        p = dict(spectral.parameters(), pe_filters=3, psi_width=5, pe_width=5)
        torch.manual_seed(827)
        full = spectral.SpectralPosition(p, 'full').double()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(827)
        control = spectral.SpectralPosition(p, 'constant_spectrum').double()
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[key], atol=0, rtol=0)
        _, values, vectors, _ = ingredients()
        weights = full.coefficients(values).mean(0)
        expected = torch.eye(len(values), dtype=torch.float64)[..., None]*weights
        actual = control.kernels(values, vectors)
        torch.testing.assert_close(actual, expected, atol=2e-15, rtol=0)
        self.assertGreater(float((full.kernels(values, vectors)-actual).abs().max().detach()), 1e-3)
        gradients = torch.autograd.grad(actual.square().sum(), tuple(control.psi.parameters()))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 1e-6)

    def test_dense_gin_matches_neighbor_message_sum_and_gradients(self):
        _, _, _, adjacency = ingredients()
        layer = spectral.DenseGINLayer(3, 5).double()
        h = torch.randn(6, 6, 3, dtype=torch.float64, requires_grad=True)
        actual = layer(h, adjacency)
        messages = torch.stack([sum((h[j] for j in range(6) if adjacency[i, j] != 0),
                                    torch.zeros_like(h[0])) for i in range(6)])
        expected = layer.mlp((1+layer.eps)*h+messages)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)
        left = torch.autograd.grad(actual.square().sum(), (h,)+tuple(layer.parameters()), retain_graph=True)
        right = torch.autograd.grad(expected.square().sum(), (h,)+tuple(layer.parameters()))
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, atol=1e-12, rtol=0)

    def test_full_spectrum_includes_isolates_and_queries_cannot_affect_support(self):
        values, vectors, audit = spectral.eigensystem(np.zeros((7, 7)))
        np.testing.assert_array_equal(values, np.ones(7))
        np.testing.assert_allclose(vectors @ vectors.T, np.eye(7), atol=1e-14)
        self.assertEqual(audit['retained_eigenvectors'], 7)
        self.assertEqual(audit['isolated_nodes'], 7)
        a, _ = fixture()
        omitted = np.flatnonzero(a['folds'] == 0)
        first = spectral.context(a, omitted, 'cpu')
        changed = {k: v.copy() for k, v in a.items()}
        changed['y_train'][omitted] = 1-changed['y_train'][omitted]
        second = spectral.context(changed, omitted, 'cpu')
        for key in ('adjacency', 'eigenvalues', 'eigenvectors', 'target_nodes'):
            torch.testing.assert_close(getattr(first, key), getattr(second, key), atol=0, rtol=0)
        query = a['pairs_train'][:, omitted]
        self.assertTrue(torch.all(first.adjacency[query[0], a['node_counts'][1]+query[1]] == 0))
        self.assertEqual(first.eigenvectors.shape, (14, 14))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
