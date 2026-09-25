import unittest

import numpy as np

from improve import foundation_transfer_eval as ev
from improve import foundation_transfer_runner as runner


class LogitTests(unittest.TestCase):
    def test_logit_clips_and_is_monotone(self):
        p = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
        z = ev.logit(p)
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue(np.all(np.diff(z) > 0))

    def test_recipes_have_unit_logit_weight(self):
        for name, weights in ev.RECIPES.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0, msg=name)
            self.assertTrue(all(w >= 0 for w in weights.values()))


class GateTests(unittest.TestCase):
    def _rows(self, task, recipe, pairs):
        rows = []
        for seed, base, value in pairs:
            rows.append(dict(task=task, recipe='reference', seed=seed, auc=base[0], ap=base[1]))
            rows.append(dict(task=task, recipe=recipe, seed=seed, auc=value[0], ap=value[1]))
        return rows

    def test_midrug_needs_mean_and_two_of_three(self):
        rows = self._rows('MiDrug', 'tdp_tiered62', [
            (666, (0.5, 0.5), (0.51, 0.51)), (0, (0.5, 0.5), (0.51, 0.51)),
            (1, (0.5, 0.5), (0.49, 0.49))])
        result = ev.gate(rows)['tdp_tiered62']
        self.assertTrue(result['tasks']['MiDrug']['passed'])
        self.assertEqual(result['passed'], True)

    def test_midrug_fails_on_two_negative_seeds(self):
        rows = self._rows('MiDrug', 'tdp_tiered62', [
            (666, (0.5, 0.5), (0.49, 0.49)), (0, (0.5, 0.5), (0.49, 0.49)),
            (1, (0.5, 0.5), (0.51, 0.51))])
        self.assertFalse(ev.gate(rows)['tdp_tiered62']['passed'])

    def test_lncdrug_rejects_any_single_seed_drop(self):
        rows = self._rows('LncDrug', 'tdp_tiered62', [
            (666, (0.5, 0.5), (0.501, 0.501)), (0, (0.5, 0.5), (0.501, 0.501)),
            (1, (0.5, 0.5), (0.4999, 0.4999))])
        self.assertFalse(ev.gate(rows)['tdp_tiered62']['passed'])


class RunnerTests(unittest.TestCase):
    def test_checkpoint_hash_is_pinned(self):
        self.assertEqual(len(runner.CHECKPOINT_SHA256), 64)
        self.assertEqual(runner.BANKS, ('base36', 'tiered62', 'context76'))


if __name__ == '__main__':
    unittest.main()
