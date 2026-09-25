"""Mechanism and leakage-boundary checks for shared conditional ICL attention."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from improve.tabicl_care_modules import (ConditionalRelationAdapter, RelationEvidence,
    adapted_logits, adaptation_loss, prefix_cache)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(71)
        r, d = rng.normal(size=(9, 6)), rng.normal(size=(7, 6))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.evidence = RelationEvidence(r @ r.T, d @ d.T, np.ones(9, bool), np.ones(7, bool))
        self.support = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
        self.query = np.array([[0, 5, 7], [2, 3, 6]])

    def test_query_batch_and_support_order(self):
        relation, evidence = self.evidence.features(self.query, self.support)
        one, one_evidence = self.evidence.features(self.query[:, [1]], self.support)
        np.testing.assert_array_equal(one, relation[[1]])
        np.testing.assert_array_equal(one_evidence, evidence[[1]])
        order = np.array([5, 2, 0, 4, 1, 3])
        changed, changed_evidence = self.evidence.features(self.query, self.support[:, order])
        np.testing.assert_allclose(changed, relation[:, order], atol=2e-6)
        np.testing.assert_allclose(changed_evidence, evidence, atol=2e-6)

    def test_identifier_renaming_and_missing_evidence(self):
        original, query = self.evidence.features(self.query, self.support)
        order = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])
        renamed = RelationEvidence(self.evidence.rna_kernel[np.ix_(order, order)],
            self.evidence.drug_kernel, self.evidence.rna_present[order], self.evidence.drug_present)
        q, s = self.query.copy(), self.support.copy()
        q[0], s[0] = 8 - q[0], 8 - s[0]
        relation, evidence = renamed.features(q, s)
        np.testing.assert_array_equal(original, relation)
        np.testing.assert_array_equal(query, evidence)
        renamed.rna_present[:] = False
        relation, evidence = renamed.features(q, s)
        np.testing.assert_array_equal(relation[..., 0], 0.)
        np.testing.assert_array_equal(relation[..., 2], 0.)
        self.assertTrue(np.isfinite(relation).all())

    def test_support_label_leakage_boundary(self):
        with self.assertRaises(ValueError):
            self.evidence.features(self.support[:, :1], self.support)
        with self.assertRaises(ValueError):
            self.evidence.features(self.query, np.tile(self.support, (1, 2)))
        with self.assertRaises(ValueError):
            self.evidence.features(self.query.astype(float), self.support)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.relation = torch.randn(7, 13, 5)
        self.evidence = torch.rand(7, 6)
        self.native = torch.randn(2, 7, 2)
        self.adapter = ConditionalRelationAdapter()

    def test_zero_start_and_bounded_nonzero_modification(self):
        self.assertEqual(torch.count_nonzero(self.adapter(self.relation, self.evidence, self.native)).item(), 0)
        with torch.no_grad():
            self.adapter.global_coefficients.fill_(20.)
        bias = self.adapter(self.relation, self.evidence, self.native, .25)
        self.assertGreater(torch.count_nonzero(bias).item(), 0)
        self.assertLessEqual(bias.abs().max().item(), .5)
        self.assertEqual(torch.count_nonzero(self.adapter(self.relation, self.evidence, self.native, 0.)).item(), 0)
        with self.assertRaises(ValueError):
            self.adapter(self.relation, self.evidence, self.native, 1.1)

    def test_class_permutation_and_query_independence(self):
        with torch.no_grad():
            self.adapter.controller[-1].weight.normal_()
        expected = self.adapter(self.relation, self.evidence, self.native)
        swapped = self.adapter(self.relation, self.evidence, self.native.flip(-1))
        torch.testing.assert_close(expected, swapped, atol=0., rtol=0.)
        one = self.adapter(self.relation[[3]], self.evidence[[3]], self.native[:, [3]])
        torch.testing.assert_close(one, expected[:, [3]])

    def test_actual_frozen_layer_recovery_and_learning(self):
        from tabicl._model.learning import ICLearning
        predictor = ICLearning(max_classes=2, out_dim=2, d_model=16, num_blocks=2,
            nhead=2, dim_feedforward=32, dropout=0., zero_init=False).eval().requires_grad_(False)
        rows = torch.randn(2, 20, 16)
        labels = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]).expand(2, -1)
        cache = prefix_cache(predictor, rows, labels)
        original = {name: value.clone() for name, value in predictor.state_dict().items()}
        value = adapted_logits(predictor, cache, self.relation, self.evidence, self.adapter)
        torch.testing.assert_close(value, cache['native_logits'], atol=1e-6, rtol=1e-6)
        targets, permutations = torch.tensor([0, 1, 0, 1, 0, 1, 0]), torch.tensor([[0, 1], [1, 0]])
        loss, terms = adaptation_loss(value, cache['native_logits'], targets, permutations)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(self.adapter.global_coefficients.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in predictor.parameters()))
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=.1)
        optimizer.step()
        changed = adapted_logits(predictor, cache, self.relation, self.evidence, self.adapter)
        self.assertGreater((changed - value).abs().max().item(), 1e-6)
        for name, param in predictor.state_dict().items():
            torch.testing.assert_close(param, original[name], atol=0., rtol=0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_short_cuda_queries_backpropagate(self):
        from tabicl._model.learning import ICLearning
        predictor = ICLearning(max_classes=2, out_dim=2, d_model=16, num_blocks=2,
            nhead=2, dim_feedforward=32, dropout=0., zero_init=False).cuda().eval().requires_grad_(False)
        rows = torch.randn(2, 20, 16, device='cuda')
        labels = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]], device='cuda').expand(2, -1)
        cache = prefix_cache(predictor, rows, labels)
        adapter = self.adapter.cuda()
        changed = adapted_logits(predictor, cache, self.relation.cuda(), self.evidence.cuda(), adapter)
        changed.square().mean().backward()
        self.assertTrue(torch.isfinite(adapter.global_coefficients.grad).all())
        self.assertGreater(adapter.global_coefficients.grad.abs().sum().item(), 0.)


class EvaluationBoundaryTests(unittest.TestCase):
    def test_rejected_selection_blocks_outer_before_any_feature_loader(self):
        from improve import tabicl_care as experiment
        from improve import tabular_foundation as base
        with tempfile.TemporaryDirectory() as tmp:
            p = dict(sha256='example_protocol', smoke=False)
            base.write_json(Path(tmp) / 'SELECTION.json', base.seal(dict(
                protocol_sha256=p['sha256'], eligible=False, recipe=dict(id='zero'))))
            with patch.object(experiment.graph, 'protocol') as graph_loader, patch.object(np, 'load') as array_loader:
                with self.assertRaisesRegex(ValueError, 'outer loading is prohibited'):
                    experiment.stage_arrays(tmp, p, 'MiDrug', 'outer')
                graph_loader.assert_not_called()
                array_loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
