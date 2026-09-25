"""Semantic and leakage-boundary checks for the relational ICL probe."""
import unittest

import numpy as np
import torch

from improve.tabicl_reli_inputs import local_sites
from improve import tabicl_reli_modules as reli


class EvidenceTests(unittest.TestCase):
    def test_local_comparison_recovers_information_lost_by_pooling(self):
        local = np.array([[[1., 0.], [-1., 0.]], [[0., 1.], [0., -1.]],
                          [[1., 0.], [-1., 0.]]], np.float32)
        weights = np.full((3, 2), .5, np.float32)
        pooled = (local*weights[..., None]).sum(1)
        np.testing.assert_array_equal(pooled[0], pooled[1])
        kernel = reli.soft_local_kernel(local, weights)
        self.assertGreater(kernel[0, 2], kernel[0, 1]+.5)

    def test_repeated_evidence_with_split_mass_is_invariant(self):
        rng = np.random.default_rng(1)
        local = reli.normalize_rows(rng.normal(size=(3, 3, 5))).astype(np.float32)
        weights = np.full((3, 3), 1/3, np.float32)
        expected = reli.soft_local_kernel(local, weights)
        repeated = np.repeat(local, 2, axis=1)
        mass = np.repeat(weights/2, 2, axis=1)
        np.testing.assert_allclose(reli.soft_local_kernel(repeated, mass), expected, atol=1e-6)

    def test_missing_entities_cannot_receive_matches(self):
        local = np.ones((2, 2, 3), np.float32)/np.sqrt(3)
        weights = np.array([[.5, .5], [0., 0.]], np.float32)
        kernel = reli.soft_local_kernel(local, weights)
        np.testing.assert_array_equal(kernel[1], 0)
        np.testing.assert_array_equal(kernel[:, 1], 0)
        self.assertGreater(kernel[0, 0], .5)

    def test_query_permutation_cannot_change_relation_scores(self):
        rng = np.random.default_rng(7)
        kernels = {}
        for prefix, n in [('RNA', 5), ('drug', 4)]:
            values = rng.normal(size=(n, n)).astype(np.float32)
            kernels[prefix+'_local'] = (values+values.T)/2
            kernels[prefix+'_present'] = np.ones(n, bool)
        q = np.array([[1, 3, 2], [0, 1, 2]])
        s = np.array([[0, 2, 4, 1], [1, 3, 0, 2]])
        full = reli.pair_bias(kernels, 'RNA', q, s, 'local')
        np.testing.assert_array_equal(reli.pair_bias(kernels, 'RNA', q[:, [2, 0]], s, 'local'), full[[2, 0]])
        for key, value in list(kernels.items()):
            if key.startswith('RNA_'):
                kernels[key.replace('RNA_', 'OtherRNA_')] = value
        np.testing.assert_array_equal(full, reli.pair_bias(kernels, 'OtherRNA', q, s, 'local'))

    def test_full_span_sites_and_overlap_mass(self):
        for length in [0, 5, 22, 510, 1022, 10000, 347561]:
            sites, weights, windows = local_sites(length)
            self.assertLessEqual(len(sites), 64)
            if not length:
                self.assertEqual(sites, [])
                continue
            self.assertEqual(sites[0][0], 0)
            self.assertEqual(sites[-1][1], length)
            self.assertAlmostEqual(float(weights.sum()), 1., places=6)
            for (start, end), window in zip(sites, windows):
                self.assertTrue(window <= start < end <= min(window+510, length))


class AttentionTests(unittest.TestCase):
    def setUp(self):
        from tabicl._model.learning import ICLearning
        torch.manual_seed(13)
        self.model = ICLearning(max_classes=2, out_dim=2, d_model=32, num_blocks=2,
                               nhead=4, dim_feedforward=64, zero_init=False).eval()
        self.rows = torch.randn(2, 11, 32)
        self.labels = torch.tensor([[0., 1., 0., 1., 1., 0.], [1., 0., 1., 0., 0., 1.]])
        self.bias = torch.randn(5, 6)
        self.configs = [reli.recipes()[0], next(c for c in reli.recipes() if c['mode'] == 'local' and c['strength'] == .5)]

    @torch.no_grad()
    def test_zero_replays_native_icl_and_weights_remain_frozen(self):
        before = {key: val.clone() for key, val in self.model.state_dict().items()}
        expected = self.model._icl_predictions(self.rows.clone(), self.labels)[:, 6:, :2]
        outputs = reli.replay_last_layer(self.model, self.rows, self.labels, {'local': self.bias}, self.configs)
        torch.testing.assert_close(outputs['zero'], expected, atol=1e-6, rtol=1e-6)
        self.assertGreater(float((outputs['local_0p5']-outputs['zero']).abs().max()), 1e-4)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    @torch.no_grad()
    def test_queries_cannot_change_each_other_through_supports(self):
        whole = reli.replay_last_layer(self.model, self.rows, self.labels, {'local': self.bias}, self.configs)
        keep = [4, 0, 2]
        subset = torch.cat([self.rows[:, :6], self.rows[:, 6:][:, keep]], 1)
        partial = reli.replay_last_layer(self.model, subset, self.labels, {'local': self.bias[keep]}, self.configs)
        for key in whole:
            torch.testing.assert_close(partial[key], whole[key][:, keep], atol=1e-6, rtol=1e-6)

    @torch.no_grad()
    def test_support_order_and_bias_columns_must_agree(self):
        whole = reli.replay_last_layer(self.model, self.rows, self.labels, {'local': self.bias}, self.configs)
        order = [4, 0, 2, 5, 1, 3]
        rows = torch.cat([self.rows[:, :6][:, order], self.rows[:, 6:]], 1)
        reordered = reli.replay_last_layer(self.model, rows, self.labels[:, order],
                                          {'local': self.bias[:, order]}, self.configs)
        for key in whole:
            torch.testing.assert_close(reordered[key], whole[key], atol=1e-6, rtol=1e-6)

    def test_misaligned_query_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'alignment'):
            reli.replay_last_layer(self.model, self.rows, self.labels, {'local': self.bias[:2]}, self.configs)


if __name__ == '__main__':
    unittest.main()
