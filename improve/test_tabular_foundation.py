"""Protocol and data-flow checks for the official pretrained-model comparison."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from improve import tabular_foundation as experiment


class DataBoundaryTests(unittest.TestCase):
    def arrays(self):
        return dict(raw_train=np.zeros((4,36)), raw_val=np.ones((4,36)),
                    x_train=np.zeros((4,36)), x_val=np.ones((4,36)),
                    y_train=np.array([0,1,0,1]), y_val=np.array([1,0,1,0]),
                    pairs_train=np.array([[0,1,2,3],[0,0,0,0]]),
                    pairs_val=np.array([[4,5,6,7],[0,0,1,1]]))

    def test_duplicate_and_overlapping_pairs_rejected(self):
        a = self.arrays()
        experiment.validate_arrays(a)
        a['pairs_val'][:,0] = a['pairs_train'][:,0]
        with self.assertRaises(ValueError):
            experiment.validate_arrays(a)
        a = self.arrays()
        a['pairs_train'][:,1] = a['pairs_train'][:,0]
        with self.assertRaises(ValueError):
            experiment.validate_arrays(a)

    def test_outer_requires_selection_before_reading_arrays(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(experiment.np, 'load') as loader:
            with self.assertRaises(FileNotFoundError):
                experiment.load_features(Path(folder), dict(tasks=['LncDrug']), 'LncDrug', 'outer')
            loader.assert_not_called()

    def test_nonfinite_features_rejected(self):
        a = self.arrays()
        a['raw_val'][0,0] = np.nan
        with self.assertRaises(ValueError):
            experiment.validate_arrays(a)

    def test_smoke_subset_keeps_both_classes_without_replacement(self):
        y = np.r_[np.zeros(99),1]
        chosen = experiment.balanced_subset(y, 8, 7)
        self.assertEqual(len(set(chosen)), 8)
        self.assertEqual(set(y[chosen]), {0,1})
        np.testing.assert_array_equal(chosen, experiment.balanced_subset(y,8,7))

    def test_fit_receives_only_context_and_positive_column_is_identified(self):
        class FakeEstimator:
            classes_ = np.array([1,0])
            def fit(self, x, y):
                self.seen_x, self.seen_y = x.copy(), y.copy()
            def predict_proba(self, x):
                return np.tile([.7,.3], (len(x),1))
        a, model = self.arrays(), FakeEstimator()
        experiment.fit_context(model, a['raw_train'], a['y_train'])
        np.testing.assert_array_equal(model.seen_y, a['y_train'])
        np.testing.assert_array_equal(model.seen_x, a['raw_train'])
        np.testing.assert_allclose(experiment.query_scores(model,a['raw_val']), .7)

    def test_invalid_probability_rows_rejected(self):
        class BadEstimator:
            classes_ = np.array([0,1])
            def predict_proba(self, x):
                return np.tile([.7,.7], (len(x),1))
        with self.assertRaises(AssertionError):
            experiment.query_scores(BadEstimator(), np.zeros((2,36)))


class SelectionTests(unittest.TestCase):
    def rows(self):
        cfgs = experiment.configurations()[:2]
        scores = {cfgs[0]['id']:[.99,.7,.7], cfgs[1]['id']:[.90,.9,.9]}
        rows = [dict(config=cfg, seed=seed, criterion=value, auc=value, ap=value, macro_drug_ap=value)
                for cfg in cfgs for seed,value in zip([666,0,1],scores[cfg['id']])]
        return cfgs, rows

    def test_selection_uses_all_seed_means(self):
        cfgs, rows = self.rows()
        selected = experiment.select_recipe(cfgs,rows,[666,0,1])
        self.assertEqual(selected['config']['id'],'c1')

    def test_missing_or_duplicate_seeds_rejected(self):
        cfgs, rows = self.rows()
        with self.assertRaises(ValueError):
            experiment.select_recipe(cfgs,rows[:-1],[666,0,1])
        rows[-1]['seed'] = 0
        with self.assertRaises(ValueError):
            experiment.select_recipe(cfgs,rows,[666,0,1])

    def test_ties_are_stable(self):
        cfgs, rows = self.rows()
        for r in rows:
            r['criterion'] = .8
        self.assertEqual(experiment.select_recipe(cfgs,rows[::-1],[666,0,1])['config']['id'],'c0')

    def test_frozen_decision_detects_inner_artifact_change(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            experiment.write_json(out/'inner.json',dict(score=.8))
            value = experiment.seal(dict(protocol_sha256='frozen', chosen={},
                        results_sha256={'inner.json':experiment.digest(out/'inner.json')}))
            experiment.write_json(out/'SELECTION.json',value)
            experiment.selection(out,dict(sha256='frozen'))
            experiment.write_json(out/'inner.json',dict(score=.9))
            with self.assertRaises(ValueError):
                experiment.selection(out,dict(sha256='frozen'))

    def test_default_outer_panel_deduplicates_selected_default(self):
        cfgs = experiment.configurations()
        p = dict(tasks=['LncDrug'], seeds=[666,0,1], default_id='c0', configurations=cfgs)
        with mock.patch.object(experiment,'selection',return_value={'chosen':{'LncDrug':{'config':cfgs[0]}}}):
            self.assertEqual(len(list(experiment.jobs(Path('.'),p,'outer'))),3)
        with mock.patch.object(experiment,'selection',return_value={'chosen':{'LncDrug':{'config':cfgs[1]}}}):
            self.assertEqual(len(list(experiment.jobs(Path('.'),p,'outer'))),6)

    def test_metric_macro_excludes_single_class_drugs(self):
        y = np.array([0,1,0,1,1])
        scores = np.array([.1,.9,.9,.1,.8])
        pairs = np.array([[0,1,2,3,4],[0,0,1,1,2]])
        result = experiment.metrics(y,scores,pairs)
        self.assertEqual(result['macro_drugs'],2)
        self.assertEqual(result['macro_drug_ap'],.75)


if __name__ == '__main__':
    unittest.main()
