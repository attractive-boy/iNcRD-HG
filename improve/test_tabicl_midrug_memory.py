"""Information boundaries, fixed protection, routing and selection checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from improve import tabular_foundation as base
from improve import tabicl_midrug_memory as experiment
from improve import tabicl_midrug_memory_modules as kernels


def small_memory():
    rng = np.random.default_rng(41)
    drug, rna = rng.normal(size=(16, 12)), rng.normal(size=(24, 10))
    pairs = np.array([(r, d) for r in range(24) for d in range(16)], dtype=np.int64).T
    train, query = pairs[:, :240], pairs[:, 240:]
    state = kernels.fit_memory(drug, rna, train, clusters=3, context_size=80,
                               reservoir_size=8, side_dim=8)
    return drug, rna, train, query, state


class MemoryTests(unittest.TestCase):
    def test_retrieved_rows_are_unique_fitting_rows_with_shared_reservoir(self):
        _, _, train, query, state = small_memory()
        for context in state['contexts']:
            self.assertEqual(len(context), 80)
            self.assertEqual(len(np.unique(context)), 80)
            self.assertTrue(set(state['reservoir']) <= set(context))
            self.assertFalse(set(map(tuple, train[:, context].T)) & set(map(tuple, query.T)))

    def test_routing_is_per_query_and_invariant_to_other_queries(self):
        _, _, _, query, state = small_memory()
        w, rel = kernels.route(state, query)
        np.testing.assert_allclose(w.sum(1), 1., atol=1e-14)
        np.testing.assert_array_equal((w > 0).sum(1), np.full(len(w), 2))
        reverse, rel_reverse = kernels.route(state, query[:, ::-1])
        np.testing.assert_array_equal(reverse[::-1], w)
        np.testing.assert_array_equal(rel_reverse[::-1], rel)
        split = np.concatenate([kernels.route(state, part)[0] for part in np.array_split(query, 5, axis=1)])
        np.testing.assert_array_equal(split, w)

    def test_context_and_query_features_do_not_use_query_labels(self):
        _, _, train, query, state = small_memory()
        rng = np.random.default_rng(7)
        a = dict(raw_train=rng.normal(size=(240, 36)).astype(np.float32),
                 raw_query_views=rng.normal(size=(5, query.shape[1], 36)).astype(np.float32),
                 pairs_train=train, pairs_val=query, y_train=np.arange(240) % 2,
                 y_val=np.zeros(query.shape[1]))
        weights, _ = kernels.route(state, query)
        before = experiment.context_and_queries(a, state, weights, 'drug_side', 0)
        a['y_val'][:] = 1
        after = experiment.context_and_queries(a, state, weights, 'drug_side', 0)
        for x, y in zip(before, after):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(before[0].shape, (80, 52))
        self.assertEqual(before[2].shape[0], 5)

    def test_side_projection_and_memory_serialization(self):
        drug, _, train, query, state = small_memory()
        expected = (drug-state['drug_pca_mean']) @ state['drug_pca_components'].T
        np.testing.assert_allclose(state['drug_embedding'], expected, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.npz'
            np.savez_compressed(path, **state)
            with np.load(path, allow_pickle=False) as archive:
                loaded = {k:archive[k].copy() for k in archive.files}
            np.testing.assert_array_equal(kernels.route(state, query)[0], kernels.route(loaded, query)[0])
            np.testing.assert_array_equal(kernels.append_side(np.zeros((len(train.T), 36)), train, state),
                                          kernels.append_side(np.zeros((len(train.T), 36)), train, loaded))

    def test_bad_sizes_and_entity_indices_are_rejected(self):
        drug, rna, train, query, state = small_memory()
        with self.assertRaises(ValueError):
            kernels.fit_memory(drug, rna, train, context_size=8, reservoir_size=8)
        with self.assertRaises(ValueError):
            kernels.fit_memory(drug, rna, np.repeat(train[:, :1], 4, axis=1))
        bad = query.copy()
        bad[1, 0] = len(drug)
        with self.assertRaises(ValueError):
            kernels.route(state, bad)


class ProtectedResidualTests(unittest.TestCase):
    def test_lncdrug_identity_ignores_specialist_and_preserves_dtype_bits(self):
        p = np.array([0., 1., .123456789, .87654321], dtype=np.float32)
        for cfg in kernels.recipes():
            result = kernels.protected_residual('LncDrug', p, np.nan, None, cfg)
            self.assertEqual(result.dtype, p.dtype)
            self.assertEqual(result.tobytes(), p.tobytes())

    def test_zero_residual_and_zero_reliability_are_exact(self):
        p = np.array([0., 1., .123456789])
        np.testing.assert_array_equal(kernels.protected_residual('MiDrug', p, None, None, kernels.recipes()[0]), p)
        cfg = kernels.recipes()[1]
        result = kernels.protected_residual('MiDrug', p, np.full((5, 3), .6), np.zeros(3), cfg)
        np.testing.assert_array_equal(result, p)

    def test_logit_correction_is_bounded_and_disagreement_reduces_it(self):
        p = np.array([.2, .4, .7])
        cfg = next(c for c in kernels.recipes() if c['component'] == 'drug_side'
                   and c['alpha'] == .2 and c['gate'] == 'consensus')
        stable = np.tile([.8, .8, .9], (5, 1))
        varied = stable.copy()
        varied[:, 0] = [.5, .95, .95, .9, .7]  # Identical mean, larger variance.
        first = kernels.protected_residual('MiDrug', p, stable, np.ones(3), cfg)
        second = kernels.protected_residual('MiDrug', p, varied, np.ones(3), cfg)
        self.assertTrue(np.all(np.abs(kernels.logit(first)-kernels.logit(p)) <= .4+1e-12))
        self.assertLess(second[0]-p[0], first[0]-p[0])

    def test_public_predictor_protected_path_needs_no_expert_or_query_inputs(self):
        predictor = object.__new__(experiment.MidrugMemoryPredictor)
        predictor.selection = dict(chosen=dict(MiDrug=dict(recipe=kernels.recipes()[1])))
        p = np.array([.1, .9], dtype=np.float32)
        with patch.object(np, 'load', side_effect=AssertionError('specialist read')):
            np.testing.assert_array_equal(predictor.predict('LncDrug', p), p)


class SelectionTests(unittest.TestCase):
    def rows(self):
        candidates, seeds = kernels.recipes()[:2], [666, 0, 1]
        rows = [dict(recipe_id=cfg['id'], seed=seed, auc=.8, ap=.7, criterion=.75)
                for cfg in candidates for seed in seeds]
        return candidates, seeds, rows

    def test_selection_requires_both_metrics_and_multiple_seeds(self):
        candidates, seeds, rows = self.rows()
        self.assertEqual(kernels.choose(rows, candidates, seeds)[0]['recipe'], candidates[0])
        for r in rows[3:]:
            r.update(auc=.81, ap=.69, criterion=.75)
        self.assertEqual(kernels.choose(rows, candidates, seeds)[0]['recipe'], candidates[0])
        for r in rows[3:]:
            r.update(auc=.81, ap=.71, criterion=.76)
        self.assertEqual(kernels.choose(rows, candidates, seeds)[0]['recipe'], candidates[1])
        rows[-1]['seed'] = 666
        with self.assertRaises(ValueError):
            kernels.choose(rows, candidates, seeds)

    def test_missing_seed_and_nonfinite_metrics_are_rejected(self):
        candidates, seeds, rows = self.rows()
        with self.assertRaises(ValueError):
            kernels.choose(rows[:-1], candidates, seeds)
        rows[-1]['auc'] = np.nan
        with self.assertRaises(ValueError):
            kernels.choose(rows, candidates, seeds)

    def test_selection_precedes_every_outer_load(self):
        p = dict(tasks=['LncDrug', 'MiDrug'])
        with patch.object(experiment, 'selection', side_effect=ValueError('not frozen')), \
             patch.object(experiment.reference, 'features') as features, patch.object(np, 'load') as loader:
            with self.assertRaisesRegex(ValueError, 'not frozen'):
                experiment.data(Path('/unused'), p, 'outer')
            features.assert_not_called()
            loader.assert_not_called()

    def test_selection_tampering_and_lncdrug_path_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = dict(sha256='p', recipes=kernels.recipes(), baseline_recipes={'LncDrug':{'id':'old'}})
            chosen = dict(MiDrug={'recipe':kernels.recipes()[0]},
                          LncDrug=dict(path='identity', reference_recipe={'id':'old'}))
            value = base.seal(dict(protocol_sha256='p', chosen=chosen, artifacts_sha256={}))
            base.write_json(root/'SELECTION.json', value)
            experiment.selection(root, p)
            chosen['LncDrug']['path'] = 'adaptive'
            base.write_json(root/'SELECTION.json', base.seal(dict(protocol_sha256='p', chosen=chosen, artifacts_sha256={})))
            with self.assertRaises(ValueError):
                experiment.selection(root, p)


if __name__ == '__main__':
    unittest.main()
