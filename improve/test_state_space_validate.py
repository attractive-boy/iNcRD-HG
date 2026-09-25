import unittest
from unittest.mock import patch

import numpy as np

from improve import state_space_validate as validation


class StateSpaceValidationTest(unittest.TestCase):
    def test_query_data_rejected_before_model_setup(self):
        for key in ('y_val', 'pairs_val', 'anchor', 'context76_val'):
            with self.subTest(key=key), patch.object(validation.method, 'setup') as setup:
                with self.assertRaisesRegex(ValueError, 'Only fitting data'):
                    validation.train_fixed({key: np.zeros(2)}, {}, 'full', {}, 0, 'cpu')
                setup.assert_not_called()

    def test_wrong_model_arm_cannot_fall_through_to_geometry(self):
        with patch.object(validation.method, 'setup') as setup:
            with self.assertRaisesRegex(ValueError, 'Unknown state-space control'):
                validation.train_fixed({}, {}, 'attention_only', {}, 0, 'cpu')
            setup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
