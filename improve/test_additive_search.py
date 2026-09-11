"""Checks of search fairness, checkpoint selection, and schedule transfer."""
import unittest

import numpy as np

from improve.additive_search import learning_factor, pick_inner, recipes


class SearchTests(unittest.TestCase):
    def test_equal_nonduplicate_search_budget(self):
        for arm in ('pca_query', 'additive'):
            configs = recipes(.0005, .0005, arm)
            self.assertEqual(len(configs), 8)
            fields = ('base_lr', 'weight_decay', 'schedule') if arm == 'pca_query' else ('base_lr', 'branch_lr', 'max_correction', 'schedule')
            self.assertEqual(len({tuple(c[k] for k in fields) for c in configs}), 8)
            self.assertTrue(any(c['base_lr'] == .0005 and c['weight_decay'] == 0 and c['schedule'] == 'constant' and c['max_correction'] == 1 for c in configs))

    def test_schedule_endpoints_and_refit_time(self):
        values = [learning_factor('cosine', e, 150) for e in range(1, 151)]
        self.assertAlmostEqual(values[0], 1.)
        self.assertAlmostEqual(values[-1], .05)
        self.assertTrue(np.all(np.diff(values) < 0))
        # A refit stopped at 80 must retain the first 80 steps of the original
        # 150-step schedule, rather than decay to its minimum at step 80.
        refit = [learning_factor('cosine', e, 150) for e in range(1, 81)]
        np.testing.assert_array_equal(refit, values[:80])
        self.assertGreater(refit[-1], learning_factor('cosine', 80, 80))
        self.assertEqual(learning_factor('constant', 80, 150), 1.)

    def test_one_joint_score_and_deterministic_ties(self):
        rows = [dict(stage='inner', config_id='c0', seed=666, selection_score=.95, auc=.99, aupr=.90),
                dict(stage='inner', config_id='c1', seed=0, selection_score=.96, auc=.98, aupr=.94),
                dict(stage='inner', config_id='c1', seed=2, selection_score=.96, auc=.98, aupr=.95)]
        chosen = pick_inner(rows)
        self.assertEqual((chosen['config_id'], chosen['seed']), ('c1', 0))

    def test_outer_results_cannot_select_models(self):
        with self.assertRaises(ValueError):
            pick_inner([dict(stage='outer', config_id='c0', seed=0, selection_score=1.)])
        with self.assertRaises(ValueError):
            pick_inner([dict(stage='inner', config_id='c0', seed=0, selection_score=float('nan'))])
        with self.assertRaises(ValueError):
            learning_factor('cosine', 151, 150)


if __name__ == '__main__':
    unittest.main()
