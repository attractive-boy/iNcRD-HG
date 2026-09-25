"""Promotion must preserve previous evidence and reject changed conclusions."""
from pathlib import Path
import unittest
from unittest.mock import patch

from improve import anchor_copula_validate_v1 as validation


class CopulaValidationTests(unittest.TestCase):
    def test_recomputed_primary_controls_and_gates_must_match(self):
        fields = dict(primary_kind='full', primary_controls=['independent'],
                      declared_gate=False, confirmed_final_innovations=0)
        validation.compare_historical_fields(dict(fields, created_at_utc='preserved'), fields)
        for key, value in [('declared_gate', True), ('primary_kind', 'independent'),
                           ('primary_controls', []), ('confirmed_final_innovations', 1)]:
            changed = dict(fields); changed[key] = value
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                validation.compare_historical_fields(changed, fields)

    def test_actual_failed_assessment_is_read_only(self):
        parent = validation.ROOT / 'results/anchor_residual_energy_v2_20260916'
        path = parent / 'HISTORICAL_ASSESSMENT.json'
        if not path.exists():
            self.skipTest('Archived energy result not available in this checkout')
        before = validation.base.digest(path)
        with patch.object(validation.audit, 'assess', side_effect=AssertionError('Do not rewrite history')):
            result = validation.read_historical_assessment(parent)
        self.assertFalse(result['declared_gate'])
        self.assertEqual(before, validation.base.digest(path))


if __name__ == '__main__':
    unittest.main()
