import unittest
from unittest.mock import patch

from improve.residual_graph_experiment import fit


class ResidualBoundaryTest(unittest.TestCase):
    def test_fixed_fitting_rejects_query_and_offset_validation_scores(self):
        for field in ('pairs_val', 'y_val', 'offset_val', 'anchor_666'):
            with self.subTest(field=field), patch('improve.residual_graph_experiment.setup') as setup:
                with self.assertRaisesRegex(ValueError, 'training inputs only'):
                    fit({field: object()}, {}, 'full', .001, 666, 'cpu', fixed_epochs=20)
                setup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
