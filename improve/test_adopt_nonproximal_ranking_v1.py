"""Reject changed shared recipes and separate the two adoption stages."""
from copy import deepcopy
import unittest

from improve.adopt_nonproximal_ranking_v1 import cells, verify_recipe


class NonproximalAdoptionChecks(unittest.TestCase):
    def test_changed_recipe_is_rejected(self):
        common = dict(core_parameters={'width': 64}, screen_seed=666, confirm_seeds=[0, 1],
            epochs=400, check_interval=20, lrs=[.001, .003], weight_decay=.001,
            strong_weight=.1, lnc_policy='identity_strong_reference')
        current = dict(common, implementation_module='staco_ranking_v2')
        previous = dict(common, implementation_module='staco_ranking_v1')
        verify_recipe(current, previous)
        for key, value in [('confirm_seeds', [2, 3]), ('epochs', 100), ('lrs', [.003]), ('strong_weight', .2)]:
            changed = deepcopy(current)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                verify_recipe(changed, previous)

    def test_screen_and_confirm_sources_are_distinct(self):
        p = dict(screen_seed=666, confirm_seeds=[0, 1], lrs=[.001, .003])
        selection = dict(chosen={'no_proximal': dict(lr=.003)})
        self.assertEqual(cells(p, selection, 'screen'), [(.001, 666), (.003, 666)])
        with self.assertRaises(ValueError):
            cells(p, selection, 'confirm')
        with self.assertRaises(ValueError):
            cells(p, selection, 'outer')


if __name__ == '__main__':
    unittest.main()
