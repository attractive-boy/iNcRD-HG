"""Protect historical evidence and the prospective zero-step identity branch."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from improve import anchor_plain_validate_v1 as validation


class PlainResidualValidationTests(unittest.TestCase):
    def test_real_historical_promotion_is_read_only(self):
        for label, passed in [('anchor_residual_loopy_path_v2_20260916', True),
                              ('anchor_residual_energy_v2_20260916', False)]:
            parent = validation.ROOT / 'results' / label
            path = parent / 'HISTORICAL_ASSESSMENT.json'
            if not path.exists():
                continue
            before = validation.base.digest(path)
            with patch.object(validation.audit, 'assess', side_effect=AssertionError('No reassessment writes')), \
                 patch.object(validation.base, 'write_json', side_effect=AssertionError('No historical writes')):
                result = validation.read_historical_assessment(parent)
            self.assertEqual(result['declared_gate'], passed)
            self.assertEqual(before, validation.base.digest(path))

    def test_only_prospectively_registered_identity_accepts_zero_steps(self):
        p = dict(epochs=400, lnc_policy='identity_strong_reference')
        record = dict(epoch=0, parameters=0, trainable_parameters=0,
                      training_steps=0, gradient_evaluations=0, trained_neural_model=False)
        self.assertEqual(validation.fixed_epochs([record] * 3, 'LncDrug', p), 0)
        for task, protocol in [('MiDrug', p), ('LncDrug', dict(epochs=400))]:
            with self.assertRaisesRegex(ValueError, 'positive'):
                validation.fixed_epochs([record] * 3, task, protocol)
        for key in ('epoch', 'parameters', 'trainable_parameters', 'training_steps', 'gradient_evaluations'):
            changed = deepcopy(record)
            changed[key] = 1
            with self.assertRaisesRegex(ValueError, 'Identity recipe'):
                validation.fixed_epochs([record, changed, record], 'LncDrug', p)
        self.assertEqual(validation.fixed_epochs([dict(epoch=n) for n in (100, 140, 200)],
                                                'MiDrug', p), 140)

    def test_changed_historical_conclusions_cannot_be_promoted(self):
        fields = dict(primary_kind='full', primary_controls=['no_path_message'],
                      declared_gate=True, confirmed_final_innovations=0)
        validation.compare_historical_fields(dict(fields, created_at_utc='preserved'), fields)
        changed = dict(fields, primary_controls=[])
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            validation.compare_historical_fields(changed, fields)

    def test_versioned_family_registry_prevents_wrong_parent(self):
        self.assertEqual(validation.label_for('loopy_path'), 'anchor_residual_loopy_path_v2')
        self.assertEqual(validation.label_for('friendly_sharpness'), 'anchor_priority_friendly_sharpness_v1')
        with self.assertRaisesRegex(ValueError, 'Unregistered'):
            validation.label_for('bernoulli_copula')


if __name__ == '__main__':
    unittest.main()
