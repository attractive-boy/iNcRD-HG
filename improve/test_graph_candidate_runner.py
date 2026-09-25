import unittest
from unittest.mock import patch

from improve.graph_candidate_runner import fit


class CandidateRunnerTest(unittest.TestCase):
    def test_fixed_training_rejects_every_query_surface_before_import(self):
        for field in ('pairs_val', 'y_val', 'anchor_666', 'context76_val'):
            with self.subTest(field=field), patch('improve.graph_candidate_runner.importlib.import_module') as load:
                with self.assertRaisesRegex(ValueError, 'training inputs only'):
                    fit({field: object()}, {}, 'full', .001, 666, 'cpu', fixed_epochs=20)
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
