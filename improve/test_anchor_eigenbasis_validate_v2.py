"""Guard promotion, initialization-gradient budgets and read-only evidence."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from improve import anchor_eigenbasis_validate_v2 as validation
from improve import anchor_eigenbasis_experiment_v1 as experiment
from improve.test_anchor_eigenbasis_experiment_v1 import protocol, fixture
import numpy as np


class EigenbasisValidationTests(unittest.TestCase):
    def test_registered_executor_and_fixed_recipe(self):
        self.assertIs(validation.experiment_for('eigenbasis_adam'), experiment)
        with self.assertRaises(ValueError):
            validation.experiment_for('differential_attention')
        self.assertEqual(validation.fixed_epochs([dict(epoch=x) for x in (60, 120, 80)],
                                                 'MiDrug', dict(epochs=400)), 80)

    def test_real_fixed_update_trace_requires_initialization_gradient(self):
        a, _ = fixture()
        a['anchor_oof'] = np.zeros_like(a['y_train'], dtype=np.float64)
        for kind in ('full', 'separable_variance', 'adamw'):
            _, _, trace, record = experiment.fit(a, protocol(), kind, .001, 17, 'cpu', fixed_epochs=12)
            validation.validate_budget(record, 12, trace)
            with self.assertRaisesRegex(ValueError, 'budget'):
                validation.validate_budget(dict(record, gradient_evaluations=12), 12, trace)
            broken = deepcopy(trace)
            broken[4]['parameter_updates_min'] = 3
            with self.assertRaisesRegex(ValueError, 'trace'):
                validation.validate_budget(record, 12, broken)

    def test_adoption_and_identity_are_distinct_from_new_training(self):
        neural = dict(task='MiDrug', trained_neural_model=True, training_steps=400,
                      gradient_evaluations=401, initialization_gradients=1)
        identity = dict(task='LncDrug', trained_neural_model=False, epoch=0, parameters=0,
                        trainable_parameters=0, training_steps=0, gradient_evaluations=0,
                        initialization_gradients=0)
        source = validation.ROOT / 'results/anchor_priority_eigenbasis_adam_v1_20260916/runs/MiDrug/full/lr0.003/s666/result.json'
        adopted = validation.base.read_json(source)
        counts = validation.historical_counts([neural, adopted, identity])
        self.assertEqual(counts, dict(new_neural_fits=1, adopted_neural_fits=1,
                                    new_identity_records=1, adopted_identity_records=0))
        validation.validate_budget(identity, 0, [])
        with self.assertRaises(ValueError):
            validation.historical_counts([dict(neural, gradient_evaluations=400)])
        with self.assertRaises(ValueError):
            validation.validate_budget(dict(identity, initialization_gradients=1), 0, [])

    def test_confirmed_history_is_recomputed_without_writes(self):
        parent = validation.ROOT / 'results/anchor_priority_eigenbasis_adam_v1_20260916'
        path = parent / 'HISTORICAL_ASSESSMENT.json'
        before = validation.base.digest(path)
        with patch.object(validation.audit, 'assess', side_effect=AssertionError('No reassessment writes')), \
             patch.object(validation.base, 'write_json', side_effect=AssertionError('No writes')):
            result = validation.read_historical_assessment(parent)
        self.assertTrue(result['declared_gate'])
        records = [validation.base.read_json(path) for path in (parent/'runs').glob('*/*/*/*/result.json')]
        self.assertEqual(validation.historical_counts(records), dict(new_neural_fits=8, adopted_neural_fits=4, new_identity_records=8, adopted_identity_records=4))
        self.assertEqual(before, validation.base.digest(path))


if __name__ == '__main__':
    unittest.main()
