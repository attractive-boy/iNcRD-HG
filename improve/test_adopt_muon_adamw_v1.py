"""Reject nominally similar baselines with changed inputs or fitting recipes."""
from copy import deepcopy
import unittest

import numpy as np

from improve.adopt_muon_adamw_v1 import verify_arrays, verify_recipe


class BaselineAdoptionTests(unittest.TestCase):
    def test_recipe_changes_are_rejected(self):
        common = dict(core_parameters={'width': 64}, screen_seed=666, epochs=400,
            check_interval=20, lrs=[.001, .003], weight_decay=.001,
            strong_weight=.1, lnc_policy='identity_strong_reference')
        current = dict(common, implementation_module='muon_graph_v1',
                       optimizer_parameters=dict(betas=[.9, .999], eps=1e-8))
        previous = dict(common, implementation_module='sharpness_graph_v1',
                        optimizer_parameters=dict(clip=5.))
        verify_recipe(current, previous)
        for key, value in [('epochs', 200), ('lrs', [.003]), ('strong_weight', .05),
                           ('core_parameters', {'width': 32}), ('screen_seed', 1)]:
            changed = deepcopy(current); changed[key] = value
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                verify_recipe(changed, previous)

    def test_order_and_outcome_changes_are_rejected(self):
        arrays = dict(pairs_train=np.array([[0, 1], [1, 0]]), y_train=np.array([0., 1.]),
                      anchor_oof=np.array([.2, .8]), anchor_666=np.array([.4, .7]))
        verify_arrays(arrays, deepcopy(arrays))
        for key in arrays:
            changed = deepcopy(arrays); changed[key] = changed[key][..., ::-1]
            with self.assertRaises(AssertionError):
                verify_arrays(changed, arrays)
        with self.assertRaisesRegex(ValueError, 'members'):
            verify_arrays(dict(arrays, extra=np.zeros(1)), arrays)


if __name__ == '__main__':
    unittest.main()
