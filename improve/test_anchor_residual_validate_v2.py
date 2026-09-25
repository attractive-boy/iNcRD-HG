"""An audit-only cache amendment must not change the frozen predictor."""
from copy import deepcopy
import unittest

from improve.anchor_residual_validate_v2 import verify_cache_equivalence, label_for


class CacheAmendmentTests(unittest.TestCase):
    def test_recipe_and_boundary_changes_are_rejected(self):
        candidate = dict(nested_reference='/old/cache', nested_protocol_sha256='old')
        old = dict(sha256='old', members=32, reference_seed=20260914,
                   graph_recipes={'MiDrug': {'arm': 'fold_match'}},
                   query_outcome_access=False, version='nested_anchor_v2')
        corrected = dict(old, sha256='new', version='nested_anchor_v3_constant_filter_audit',
            parent_reference_cache='/old/cache', parent_protocol_sha256='old',
            predictor_recipe_unchanged=True, correction_uses_evaluation_outcomes=False)
        verify_cache_equivalence(candidate, old, corrected)
        for name, value in (('members', 16), ('reference_seed', 0), ('query_outcome_access', True),
                            ('graph_recipes', {'MiDrug': {'arm': 'query_bag'}})):
            changed = deepcopy(corrected)
            changed[name] = value
            with self.assertRaisesRegex(ValueError, 'recipe or boundary changed'):
                verify_cache_equivalence(candidate, old, changed)

    def test_same_family_not_an_extra_innovation(self):
        self.assertEqual(label_for('geometry'), 'anchor_residual_geometry_v1_cachev3')
        with self.assertRaises(ValueError):
            label_for('unknown')


if __name__ == '__main__':
    unittest.main()
