"""Check biological source weighting and normalization in isolation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from improve import morphology_inputs_v1 as data


class InputChecks(unittest.TestCase):
    def test_laboratories_receive_equal_weight_and_controls_are_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            p = folder / 'toy.parquet'
            pd.DataFrame(dict(Metadata_JCP2022=['drug'] * 4 + ['DMSO'],
                Metadata_Source=['A', 'A', 'B', 'B', 'A'], X_1=[1., 3., 10., np.nan, 1000.],
                X_2=[2., 6., 20., 40., 2000.])).to_parquet(p, index=False)
            source = dict(name='toy', file=str(p), sha256=data.digest(p), s3_etag_verified=True)
            with patch.object(data, 'OUT', folder):
                keys, value, report = data.aggregate(source, {'DMSO'})
            np.testing.assert_array_equal(keys, ['drug'])
            np.testing.assert_allclose(value, [[6., 12.]])
            self.assertEqual(report['nonfinite_rows'], 1)
            self.assertEqual(report['control_rows'], 1)
            self.assertEqual(report['retained_rows'], 3)
            self.assertEqual(report['source_perturbation_groups'], 2)

    def test_constant_coordinates_are_finite_without_invented_variation(self):
        values = np.array([[2., 1.], [2., 3.], [2., 5.]], dtype=np.float32)
        result, mean, scale, report = data.normalize(values)
        np.testing.assert_array_equal(result[:, 0], np.zeros(3))
        np.testing.assert_allclose(result[:, 1], [-np.sqrt(1.5), 0., np.sqrt(1.5)])
        self.assertEqual(report['below_sd_floor'], 1)
        self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
