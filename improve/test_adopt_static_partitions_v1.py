"""Prevent mismatched recipe, partition, seed or nested-reference adoption."""
from copy import deepcopy
import unittest

from improve.adopt_static_partitions_v1 import verify_recipe


class StaticPartitionAdoptionChecks(unittest.TestCase):
    def test_exact_recipe_and_prohibited_changes(self):
        common = dict(data_protocol_sha256='data', model_seed=5, nested_protocol_sha256='nested',
            strong_weight=.1, lnc_policy='identity_strong_reference',
            training_parameters=dict(core_parameters={'width': 64}, epochs=400, weight_decay=.001,
                strong_weight=.1, lnc_policy='identity_strong_reference'))
        new = dict(common, family='cooperative_static', recipes={
            'MiDrug': {'full': {'lr': .003, 'epochs': 100}}, 'LncDrug': {'full': {'lr': .003, 'epochs': 0}}})
        old = dict(common, family='cooperative_local', recipes={
            'MiDrug': {'static_policy': {'lr': .003, 'epochs': 100}},
            'LncDrug': {'static_policy': {'lr': .003, 'epochs': 0}}})
        verify_recipe(new, old)
        for key, value in [('data_protocol_sha256', 'other'), ('model_seed', 6),
                           ('nested_protocol_sha256', 'other'), ('strong_weight', .2)]:
            changed = deepcopy(new)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                verify_recipe(changed, old)
        changed = deepcopy(new)
        changed['recipes']['MiDrug']['full']['epochs'] = 120
        with self.assertRaisesRegex(ValueError, 'steps'):
            verify_recipe(changed, old)


if __name__ == '__main__':
    unittest.main()
