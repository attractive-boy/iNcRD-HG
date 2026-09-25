import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from improve.mechanism_outer_recheck import rebuild_physical, prepare_inputs


class OuterRecheckTests(unittest.TestCase):
    def test_outer_arrays_cannot_open_before_candidate_freeze(self):
        with tempfile.TemporaryDirectory() as directory, patch('numpy.load') as loader:
            with self.assertRaises(FileNotFoundError):
                prepare_inputs(directory, 'geometry', 'MiDrug')
            loader.assert_not_called()

    def test_reconstruction_replaces_exactly_target_support_and_keeps_other_relations(self):
        template = dict(node_counts=np.array([3, 2]), pair_offset=np.array([0, 3]), nd=np.array(2),
            edges=np.array([[0, 1, 0], [3, 4, 1]]), relation=np.array([0, 0, 1]),
            target_codes=np.array([0, 3, -1]), x_0=np.ones((3, 2)), x_1=np.ones((2, 4)))
        desc = dict(node_types=['miRNA', 'drug'], physical_relations=[['miRNA', 'MiDrug', 'drug'],
                    ['miRNA', 'MiMi', 'miRNA']])
        features = dict(pca_x_miRNA=template['x_0'], x_drug=template['x_1'])
        pairs, labels = np.array([[0, 1, 2], [1, 1, 0]]), np.array([1, 0, 1])
        actual = rebuild_physical(template, desc, 'MiDrug', pairs, labels, features)
        np.testing.assert_array_equal(actual['target_codes'], [1, 4, -1])
        np.testing.assert_array_equal(actual['edges'], [[0, 2, 0], [4, 3, 1]])
        np.testing.assert_array_equal(template['target_codes'], [0, 3, -1])
        changed = dict(features, pca_x_miRNA=np.zeros((3, 2)))
        with self.assertRaises(AssertionError):
            rebuild_physical(template, desc, 'MiDrug', pairs, labels, changed)


if __name__ == '__main__':
    unittest.main()
