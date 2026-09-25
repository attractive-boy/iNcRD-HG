"""Protect old evidence and distinguish adopted fits from identity scores."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from improve import anchor_plain_validate_v2 as validation


class PriorityValidationChecks(unittest.TestCase):
    def test_only_registered_identity_has_zero_epochs(self):
        p = dict(epochs=400, lnc_policy='identity_strong_reference')
        record = dict(task='LncDrug', epoch=0, parameters=0, trainable_parameters=0,
                      training_steps=0, gradient_evaluations=0, trained_neural_model=False)
        self.assertEqual(validation.fixed_epochs([record] * 3, 'LncDrug', p), 0)
        for task, protocol in [('MiDrug', p), ('LncDrug', dict(epochs=400))]:
            with self.assertRaises(ValueError):
                validation.fixed_epochs([record] * 3, task, protocol)
        self.assertEqual(validation.fixed_epochs([dict(epoch=x) for x in (100, 180, 340)], 'MiDrug', p), 180)

    def test_adoption_is_not_new_training(self):
        neural = dict(task='MiDrug', trained_neural_model=True, training_steps=400, gradient_evaluations=400)
        identity = dict(task='LncDrug', trained_neural_model=False, epoch=0,
                        parameters=0, trainable_parameters=0, training_steps=0, gradient_evaluations=0)
        adopted = dict(neural, adopted_without_new_training=True, new_training_steps=0)
        adopted_identity = dict(identity, adopted_without_new_training=True, new_training_steps=0)
        self.assertEqual(validation.historical_counts([neural, adopted, identity, adopted_identity]),
                         dict(new_neural_fits=1, adopted_neural_fits=1, new_identity_records=1, adopted_identity_records=1))
        with self.assertRaisesRegex(ValueError, 'Adoption'):
            validation.historical_counts([dict(adopted, new_training_steps=1)])
        for key in ('epoch', 'parameters', 'trainable_parameters', 'training_steps', 'gradient_evaluations'):
            broken = deepcopy(identity)
            broken[key] = 1
            with self.assertRaisesRegex(ValueError, 'identity'):
                validation.historical_counts([broken])

    def test_real_history_read_does_not_write(self):
        parent = validation.ROOT / 'results/anchor_priority_cooperative_local_v1_20260916'
        path = parent / 'HISTORICAL_ASSESSMENT.json'
        if not path.exists():
            self.skipTest('Confirmation is still running')
        before = validation.base.digest(path)
        with patch.object(validation.audit, 'assess', side_effect=AssertionError('No reassessment writes')), \
             patch.object(validation.base, 'write_json', side_effect=AssertionError('No writes')):
            validation.read_historical_assessment(parent)
        self.assertEqual(before, validation.base.digest(path))

    def test_registry_and_unchanged_conclusion(self):
        for family in ('cooperative_local', 'polynormer', 'spline_attention'):
            self.assertEqual(validation.label_for(family), 'anchor_priority_' + family + '_v1')
        with self.assertRaises(ValueError):
            validation.label_for('loopy_path')
        before = dict(primary_kind='full', primary_controls=['control'], declared_gate=True)
        validation.compare_historical_fields(before, before)
        with self.assertRaises(ValueError):
            validation.compare_historical_fields(dict(before, declared_gate=False), before)


if __name__ == '__main__':
    unittest.main()
