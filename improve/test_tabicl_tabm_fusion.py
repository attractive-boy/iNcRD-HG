"""Checks for fusion leakage boundaries, alignment, and frozen decisions."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from improve import tabicl_tabm_fusion as fusion


class FusionChecks(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 0, 1, 1])
        self.pairs = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])

    def test_same_labels_do_not_hide_wrong_pair_order(self):
        order = [1, 0, 2, 3]
        np.testing.assert_array_equal(self.y, self.y[order])
        with self.assertRaisesRegex(ValueError, 'ordering'):
            fusion.check_alignment(self.y, self.pairs, self.y[order], self.pairs[:, order])

    def test_duplicate_pairs_are_rejected(self):
        pairs = self.pairs.copy()
        pairs[:, 1] = pairs[:, 0]
        with self.assertRaisesRegex(ValueError, 'unique'):
            fusion.check_alignment(self.y, pairs, self.y, pairs)

    def test_outer_cannot_load_before_selection(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(fusion.np, 'load') as loader:
            with self.assertRaises(FileNotFoundError):
                fusion.load_phase(Path(folder), dict(tasks=['LncDrug']), 'outer', 'LncDrug')
            loader.assert_not_called()

    def test_inner_selection_never_requests_outer_data(self):
        calls = []
        candidates = [fusion.recipes()[0], fusion.recipes()[20]]
        scores = {s: dict(tabicl=np.array([.1, .2, .8, .9]), tabm=np.array([.2, .7, .3, .8])) for s in (666, 0, 1)}

        def load(out, p, phase, task):
            self.assertEqual(phase, 'inner')
            calls.append((phase, task))
            return self.y, self.pairs, scores

        p = dict(tasks=['LncDrug'], seeds=[666, 0, 1], recipes=candidates)
        with patch.object(fusion, 'load_phase', side_effect=load):
            rows, chosen = fusion.inner_results(Path('/unused'), p)
        self.assertEqual(calls, [('inner', 'LncDrug')])
        self.assertEqual(len(rows), 6)
        self.assertEqual(set(chosen), {'LncDrug'})

    def test_selection_uses_all_seeds(self):
        candidates = [fusion.recipes()[5], fusion.recipes()[15]]
        rows = [dict(recipe_id=cfg['id'], seed=seed, auc=value, ap=value, criterion=value, macro_drug_ap=value)
                for cfg, values in zip(candidates, ((.99, .60, .60), (.8, .8, .8)))
                for seed, value in zip((666, 0, 1), values)]
        self.assertEqual(fusion.choose(rows, candidates, [666, 0, 1])['recipe'], candidates[1])
        rows[-1]['seed'] = 0
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
            fusion.choose(rows, candidates, [666, 0, 1])

    def test_conflicting_endpoint_probabilities_remain_finite(self):
        a, b = np.array([0., 1., 0., 1.]), np.array([1., 0., 0., 1.])
        with np.errstate(divide='raise', invalid='raise'):
            score = fusion.combine(a, b, dict(kind='logit', tabicl_weight=.65))
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(((score >= 0) & (score <= 1)).all())
        for weight, expected in ((0., b), (1., a)):
            np.testing.assert_array_equal(fusion.combine(a, b, dict(kind='logit', tabicl_weight=weight)), expected)

    def test_invalid_probabilities_and_weights_are_rejected(self):
        for a in (np.array([np.nan]), np.array([1.01]), np.array([-0.01])):
            with self.assertRaises(ValueError):
                fusion.combine(a, np.array([.5]), dict(kind='probability', tabicl_weight=.5))
        for weight in (np.nan, -.1, 1.1):
            with self.assertRaises(ValueError):
                fusion.combine(np.array([.4]), np.array([.5]), dict(kind='logit', tabicl_weight=weight))

    def test_query_batch_and_order_do_not_change_fusion(self):
        a, b = np.array([0., .04, .8, 1., .6]), np.array([.1, .4, .9, .8, .5])
        order = [2, 4, 0, 1, 3]
        for kind in ('probability', 'logit'):
            recipe = dict(kind=kind, tabicl_weight=.65)
            expected = fusion.combine(a, b, recipe)
            np.testing.assert_array_equal(fusion.combine(a[order], b[order], recipe), expected[order])
            split = np.concatenate([fusion.combine(a[:2], b[:2], recipe), fusion.combine(a[2:], b[2:], recipe)])
            np.testing.assert_array_equal(split, expected)

    def test_frozen_selection_and_inner_table_tampering_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            recipe = fusion.recipes()[10]
            p = dict(sha256='protocol', tasks=['LncDrug'], recipes=[recipe])
            (out/'inner_results.csv').write_text('original inner results\n')
            chosen = dict(protocol_sha256=p['sha256'], chosen={'LncDrug': {'recipe': recipe}},
                          artifacts_sha256={'inner_results.csv': fusion.base.digest(out/'inner_results.csv')})
            original = fusion.base.seal(chosen)
            fusion.base.write_json(out/'SELECTION.json', original)
            fusion.selection(out, p)
            tampered = dict(original, protocol_sha256='different protocol')
            fusion.base.write_json(out/'SELECTION.json', tampered)
            with self.assertRaisesRegex(ValueError, 'Frozen document'):
                fusion.selection(out, p)
            fusion.base.write_json(out/'SELECTION.json', original)
            (out/'inner_results.csv').write_text('changed inner results\n')
            with self.assertRaisesRegex(ValueError, 'inner results changed'):
                fusion.selection(out, p)

    def test_changed_parent_prediction_invalidates_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            source = out/'parent_scores.bin'
            source.write_bytes(b'original verified predictions')
            p = fusion.base.seal(dict(source_sha256={}, versions={}, reference_files={str(source): fusion.base.digest(source)}))
            fusion.base.write_json(out/'protocol.json', p)
            with patch.object(fusion, 'sources', return_value={}), patch.object(fusion, 'versions', return_value={}):
                fusion.protocol(out)
                source.write_bytes(b'changed predictions')
                with self.assertRaisesRegex(ValueError, 'source prediction'):
                    fusion.protocol(out)


if __name__ == '__main__':
    unittest.main()
