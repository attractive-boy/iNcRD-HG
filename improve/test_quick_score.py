"""Boundary and mechanism checks for the frozen-score rapid experiment."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from improve import quick_score_modules as rules
from improve import quick_score_eval as experiment


class QuickScoreTests(unittest.TestCase):
    def setUp(self):
        self.pairs = np.array([[0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1]])
        self.labels = np.array([1, 0, 0, 0, 1, 0])
        self.adapter = rules.FrozenScoreAdapter.fit(self.pairs, self.labels,
                                                   np.array([1., .25, 0., .1]), np.array([1., 0., 1.]))

    def test_unknown_entities_shrink_to_support_rate(self):
        self.assertEqual(self.adapter.rna_offset[3], 0.)
        self.assertEqual(self.adapter.drug_offset[2], 0.)
        changed = self.labels.copy()
        changed[2] = 1
        other = rules.FrozenScoreAdapter.fit(self.pairs, changed, self.adapter.rna_evidence, self.adapter.drug_evidence)
        self.assertNotEqual(other.rna_offset[2], self.adapter.rna_offset[2])

    def test_no_query_label_and_batch_dependence(self):
        q = np.array([[3, 0, 1, 2], [2, 1, 0, 2]])
        g, t = np.array([.5, .8, .1, .4]), np.array([.2, .7, .3, .6])
        for variant in rules.VARIANTS:
            full = self.adapter.predict(g, t, q, variant)
            repeated = self.adapter.predict(np.tile(g, 2), np.tile(t, 2), np.tile(q, (1, 2)), variant)
            np.testing.assert_array_equal(repeated, np.tile(full, 2))
            reverse = self.adapter.predict(g[::-1], t[::-1], q[:, ::-1], variant)[::-1]
            np.testing.assert_array_equal(full, reverse)
            for i in range(len(g)):
                one = self.adapter.predict(g[i:i+1], t[i:i+1], q[:, i:i+1], variant)
                np.testing.assert_array_equal(full[i:i+1], one)

    def test_drug_offset_preserves_within_drug_metrics(self):
        g = np.array([.8, .2, .1, .7, .6, .3])
        t = np.array([.7, .3, .2, .8, .5, .4])
        reference = self.adapter.predict(g, t, self.pairs, 'reference')
        corrected = self.adapter.predict(g, t, self.pairs, 'drug_frequency_correction')
        for drug in (0, 1):
            mask = self.pairs[1] == drug
            for metric in (roc_auc_score, average_precision_score):
                self.assertEqual(metric(self.labels[mask], reference[mask]), metric(self.labels[mask], corrected[mask]))

    def test_shared_gate_uses_both_evidence_ends(self):
        q = np.array([[0, 1, 2, 0], [0, 0, 0, 1]])
        score = self.adapter.predict(np.zeros(4), np.ones(4), q, 'coverage_gate')
        np.testing.assert_array_equal(score, [.75, .375, .25, .25])
        np.testing.assert_array_equal(np.sort(self.adapter.rna_evidence), np.sort(self.adapter.shuffled_rna_evidence))
        np.testing.assert_array_equal(np.sort(self.adapter.drug_evidence), np.sort(self.adapter.shuffled_drug_evidence))

    def test_logit_pool_is_class_symmetric_and_handles_boundaries(self):
        q = self.pairs[:, :4]
        g, t = np.array([.05, .4, .8, .9]), np.array([.2, .6, .7, .95])
        original = self.adapter.predict(g, t, q, 'logit_pool')
        inverted = self.adapter.predict(1-g, 1-t, q, 'logit_pool')
        np.testing.assert_allclose(original, 1-inverted, atol=1e-14, rtol=0)
        self.assertTrue(np.isfinite(self.adapter.predict(np.array([0., 1.]), np.array([.3, .7]), q[:, :2], 'logit_pool')).all())

    def test_rejects_bad_id_score_and_support(self):
        for pairs in (np.array([[-1], [0]]), np.array([[4], [0]]), np.array([[0.], [0.]])):
            with self.assertRaises(ValueError):
                self.adapter.predict(np.array([.3]), np.array([.4]), pairs, 'reference')
        for score in (np.array([np.nan]), np.array([1.1]), np.array([.2, .4])):
            with self.assertRaises(ValueError):
                self.adapter.predict(score, np.array([.4]), self.pairs[:, :1], 'reference')
        with self.assertRaises(ValueError):
            rules.observed_offsets(np.tile(self.pairs, (1, 2)), np.tile(self.labels, 2), (4, 3))

    def test_state_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.npz'
            self.adapter.save(path)
            other = rules.FrozenScoreAdapter.load(path)
            for key, value in vars(self.adapter).items():
                np.testing.assert_array_equal(value, getattr(other, key))

    def test_inner_only_loader(self):
        with patch.object(experiment.graph_context, 'protocol', return_value={'sha256': 'verified'}) as protocol:
            with patch.object(experiment.graph_context, 'features', return_value={}) as features:
                experiment.inner_arrays({'parent_dir': '/unused', 'parent_protocol_sha256': 'verified'}, 'MiDrug')
                self.assertEqual(features.call_args.args[-1], 'inner')
                self.assertEqual(protocol.call_count, 1)
        with self.assertRaises(ValueError):
            experiment.inner_arrays({}, 'holdout')

    def test_gate_never_hides_small_lnc_decline_or_selects_control(self):
        rows = [dict(task=task, variant=variant, seed=seed, auc=.8, ap=.7)
                for task in experiment.TASKS for variant in rules.VARIANTS for seed in experiment.SEEDS]
        for row in rows:
            if row['variant'] != 'reference':
                row['auc'] += .01
                row['ap'] += .01
            if row['task'] == 'LncDrug' and row['variant'] == 'logit_pool' and row['seed'] == 0:
                row['ap'] = .7 - 1e-12
        result = experiment.gate(rows)
        self.assertFalse(result['logit_pool']['passed'])
        self.assertTrue(result['coverage_gate']['passed'])
        self.assertNotIn('coverage_gate_shuffled', result)


if __name__ == '__main__':
    unittest.main()
