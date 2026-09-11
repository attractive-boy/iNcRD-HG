"""Information-boundary and mechanism tests for the denoising experiment."""
import unittest

import numpy as np
import torch
from torch_geometric.data import HeteroData

from improve.graph_tabm_modules import PairFeatureEncoder, crossfit_features
from improve.set_denoising_modules import (
    SetDenoiser, choose_checkpoint, corrupt_positive_set, drug_balanced_loss,
    fold_assignments, full_candidate_recall, support_and_queries, unknown_grid_features,
)


def toy_graph():
    g = HeteroData()
    rng = np.random.RandomState(52)
    for name, n, f in [('lncRNA', 8, 5), ('miRNA', 4, 5), ('drug', 6, 7)]:
        g[name].x = torch.from_numpy(rng.normal(size=(n, f)).astype(np.float32))
    d, r = np.meshgrid(np.arange(6), np.arange(8), indexing='ij')
    pairs = np.stack([r.ravel(), d.ravel()])
    labels = ((pairs[0] + pairs[1]) % 2 == 0).astype(np.float32)
    pos = pairs[:, labels == 1]
    edges = {
        ('lncRNA', 'LncDrug', 'drug'): pos,
        ('drug', 'rev_LncDrug', 'lncRNA'): pos[::-1].copy(),
        ('lncRNA', 'LncLnc', 'lncRNA'): np.stack([np.arange(8), np.roll(np.arange(8), 1)]),
        ('drug', 'DrugDrug', 'drug'): np.stack([np.arange(6), np.roll(np.arange(6), 1)]),
        ('lncRNA', 'LncMi', 'miRNA'): np.stack([np.arange(8), np.arange(8) % 4]),
        ('miRNA', 'MiDrug', 'drug'): np.array([[0, 1, 2, 3], [0, 2, 4, 5]]),
    }
    for key, edge in edges.items():
        g[key].edge_index = torch.from_numpy(edge).long()
    return g, pairs, labels


class SetDenoisingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(117)
        self.model = SetDenoiser(5, 3, 4, width=16, depth=2, slots=4, dropout=0.)
        self.x = torch.randn(2, 9, 5)
        self.base = torch.randn(2, 9)
        self.rna = torch.randn(9, 3)
        self.drug = torch.randn(2, 4)
        self.observed = torch.zeros(2, 9, dtype=torch.bool)
        self.observed[:, 0] = True
        self.t = torch.tensor([.5, 1.])

    def activate(self):
        with torch.no_grad():
            self.model.head.weight.normal_(std=.2)

    def output(self, interaction=True, **kwargs):
        inputs = dict(features=self.x, base_logits=self.base, rna=self.rna, drug=self.drug,
                      observed=self.observed, time=self.t, interaction=interaction)
        inputs.update(kwargs)
        return self.model(**inputs)

    def test_identity_initialization_bound_and_multiple_positives(self):
        for interaction in (False, True):
            torch.testing.assert_close(self.output(interaction), self.base, atol=0, rtol=0)
        self.activate()
        self.assertLessEqual(float((self.output()-self.base).abs().max()), 2.000001)
        high = torch.full_like(self.base, 6.)
        self.assertTrue(bool((torch.sigmoid(self.output(base_logits=high)) > .98).all()))

    def test_candidate_permutation_equivariance(self):
        self.activate()
        perm = torch.tensor([5, 1, 3, 6, 0, 8, 2, 7, 4])
        for interaction in (False, True):
            expected = self.output(interaction)[:, perm]
            actual = self.output(interaction, features=self.x[:, perm], base_logits=self.base[:, perm],
                                 rna=self.rna[perm], observed=self.observed[:, perm])
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)

    def test_local_control_cannot_read_other_candidates(self):
        self.activate()
        changed = self.rna.clone()
        changed[0] += torch.tensor([20., -10., 5.])
        before_local, before_set = self.output(False), self.output(True)
        after_local, after_set = self.output(False, rna=changed), self.output(True, rna=changed)
        torch.testing.assert_close(before_local[:, 1:], after_local[:, 1:], atol=0, rtol=0)
        self.assertGreater(float((before_set[:, 1:]-after_set[:, 1:]).abs().max()), 1e-5)

    def test_observed_context_features_are_neutralized(self):
        self.activate()
        changed_x, changed_base = self.x.clone(), self.base.clone()
        changed_x[:, 0] = 1000.
        changed_base[:, 0] = -100.
        before = self.output()
        after = self.output(features=changed_x, base_logits=changed_base)
        torch.testing.assert_close(before[:, 1:], after[:, 1:], atol=0, rtol=0)

    def test_terminal_prediction_cannot_read_query_labels(self):
        self.activate()
        query = ~self.observed
        a = torch.zeros_like(self.base)
        b = torch.ones_like(self.base)
        random = torch.rand_like(a)
        for target in (a, b):
            obs, _, shown = corrupt_positive_set(self.observed, query, target, torch.ones(2), random)
            torch.testing.assert_close(obs, self.observed, atol=0, rtol=0)
            self.assertFalse(bool(shown.any()))
            torch.testing.assert_close(self.output(observed=obs), self.output(), atol=0, rtol=0)

    def test_only_positive_context_and_no_loss_on_revealed_rows(self):
        query = ~self.observed
        target = torch.zeros_like(self.base)
        target[:, 2] = 1.
        obs, weight, shown = corrupt_positive_set(self.observed, query, target,
            torch.full((2,), .5), torch.full_like(target, .9))
        self.assertTrue(bool(obs[:, 2].all()))
        self.assertTrue(bool((weight[:, 2] == 0).all()))
        self.assertFalse(bool((shown & (target == 0)).any()))
        self.assertTrue(bool((weight[query & (target == 0)] == 1).all()))
        with self.assertRaises(ValueError):
            corrupt_positive_set(self.observed, torch.ones_like(query), target, self.t, torch.rand_like(target))

    def test_inverse_mask_weights_preserve_fixed_logit_risk(self):
        support = torch.zeros(1, 3, dtype=torch.bool)
        query = torch.ones_like(support)
        target = torch.tensor([[1., 1., 0.]])
        logits = torch.tensor([[.7, -1.2, .4]])
        base = torch.zeros_like(logits)
        values = []
        for a in (.1, .9):
            for b in (.1, .9):
                _, weight, _ = corrupt_positive_set(support, query, target, torch.tensor([.5]), torch.tensor([[a, b, .3]]))
                values.append(drug_balanced_loss(logits, base, target, query, weight, .0025))
        _, weight, _ = corrupt_positive_set(support, query, target, torch.ones(1), torch.zeros_like(target))
        expected = drug_balanced_loss(logits, base, target, query, weight, .0025)
        torch.testing.assert_close(torch.stack(values).mean(), expected)

    def test_both_paths_have_finite_useful_gradients(self):
        self.activate()
        for interaction in (False, True):
            self.model.zero_grad(set_to_none=True)
            self.output(interaction)[:, 1:].square().mean().backward()
            for name in ('feature', 'rna', 'drug', 'observed', 'time', 'head'):
                gradients = [p.grad for p in getattr(self.model, name).parameters() if p.grad is not None]
                self.assertTrue(gradients, name)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients), name)
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0., name)

    def test_fold_and_teacher_feature_information_boundary(self):
        graph, pairs, y = toy_graph()
        ids = fold_assignments(pairs, y, 3, 61)
        support, query, target, fit = support_and_queries(pairs, y, ids, 0, 8, 6)
        self.assertFalse(np.any(support & query))
        np.testing.assert_array_equal(support[pairs[1, fit], pairs[0, fit]], y[fit] == 1)
        queries = pairs[:, ids == 0]
        a = crossfit_features(graph, 'LncDrug', pairs[:, fit], y[fit], queries, folds=3, seed=73)
        altered = y.copy()
        altered[ids == 0] = 1 - altered[ids == 0]
        b = crossfit_features(graph, 'LncDrug', pairs[:, fit], altered[fit], queries, folds=3, seed=73)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        with self.assertRaises(ValueError):
            fold_assignments(np.repeat(pairs[:, :1], 12, axis=1), np.tile([0., 1.], 6), 3, 1)

    def test_full_candidate_coverage_and_visible_query_rejection(self):
        graph, pairs, y = toy_graph()
        encoder = PairFeatureEncoder(graph, 'LncDrug')
        support = pairs[:, np.flatnonzero(y == 1)[:7]]
        query, raw, names, matrix = unknown_grid_features(encoder, support)
        self.assertEqual(len(query.T) + int(matrix.sum()), 48)
        self.assertEqual(len(raw), len(query.T))
        self.assertFalse(np.any(matrix[query[1], query[0]]))
        with self.assertRaises(ValueError):
            encoder.transform(support, support)

    def test_full_candidate_recall_excludes_support(self):
        y, pairs = np.array([1., 0.]), np.array([[1, 3], [0, 0]])
        support = np.array([[True, False, False, False]])
        grid = np.array([[100., 5., 4., 3.]])
        self.assertEqual(full_candidate_recall(y, pairs, grid, support, k=1)['full_candidate_macro_recall1'], 1.)
        support[0, 1] = True
        with self.assertRaises(ValueError):
            full_candidate_recall(y, pairs, grid, support, k=1)

    def test_checkpoint_fallback_and_actual_step_selection(self):
        trace = [{'step': s, 'criterion': v} for s, v in [(0,.9), (20,.8), (40,.85), (60,.86)]]
        self.assertEqual(choose_checkpoint(trace)[0], 0)
        trace += [{'step': s, 'criterion': v} for s, v in [(80,.94), (100,.96), (120,.96)]]
        self.assertEqual(choose_checkpoint(trace)[0], 120)


if __name__ == '__main__':
    unittest.main()
