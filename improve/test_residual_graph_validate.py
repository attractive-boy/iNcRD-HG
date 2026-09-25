"""Ensure support-effect preparation cannot accidentally consume query data."""
import unittest
from unittest.mock import patch

from improve.residual_graph_validate import support_effects, BASE_TRAIN_KEYS
from improve.residual_graph_experiment import TRAIN_KEYS


class ResidualValidationTests(unittest.TestCase):
    def test_query_fields_are_rejected_before_offset_fitting(self):
        for field in ('pairs_val', 'y_val', 'offset_val', 'anchor', 'context76_val'):
            with self.subTest(field=field), patch('improve.residual_graph_validate.experiment.offset.context_offsets') as fit:
                with self.assertRaisesRegex(ValueError, 'training inputs only'):
                    support_effects({field: object()})
                fit.assert_not_called()

    def test_only_offset_arrays_are_added_to_training_contract(self):
        self.assertEqual(TRAIN_KEYS - BASE_TRAIN_KEYS, {'offset_nodes', 'offset_intercepts'})


if __name__ == '__main__':
    unittest.main()
