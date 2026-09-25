import unittest
from unittest.mock import patch

from improve.masked_teacher_validate import train_fixed


class MaskedTeacherBoundaryTest(unittest.TestCase):
    def test_query_data_cannot_enter_fixed_training(self):
        for field in ('pairs_val', 'y_val', 'anchor', 'context76_val'):
            with self.subTest(field=field), patch('improve.masked_teacher_validate.experiment.setup') as setup:
                with self.assertRaisesRegex(ValueError, 'training inputs only'):
                    train_fixed({field: object()}, {}, 'full', {}, 0, 'cpu')
                setup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
