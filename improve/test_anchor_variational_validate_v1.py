"""Check real promotion records and anchor-aligned partition inference."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from improve import anchor_variational_validate_v1 as validation
from improve import anchor_variational_feature_experiment_v1 as experiment
from improve.test_variational_feature_learning_v1 import protocol
from improve.test_nested_anchor_v2 import fixture


class VariationalPartitionTests(unittest.TestCase):
    def test_registered_fit_and_frozen_median(self):
        self.assertIs(validation.experiment_for('variational_feature_learning'),experiment)
        self.assertEqual(validation.fixed_epochs([dict(epoch=x) for x in (60,100,160)],'MiDrug',dict(epochs=400)),100)
        record = dict(trained_neural_model=False,epoch=0,parameters=0,trainable_parameters=0,
                      training_steps=0,gradient_evaluations=0)
        self.assertEqual(validation.fixed_epochs([record]*3,'LncDrug',dict(epochs=400,lnc_policy='identity_strong_reference')),0)

    def test_real_history_and_counts_without_writes(self):
        parent = validation.ROOT/'results/anchor_priority_variational_feature_learning_v1_20260916'
        before = validation.base.digest(parent/'HISTORICAL_ASSESSMENT.json')
        with patch.object(validation.base,'write_json',side_effect=AssertionError('No writes')):
            result = validation.read_historical_assessment(parent)
        self.assertTrue(result['declared_gate'])
        self.assertEqual(before,validation.base.digest(parent/'HISTORICAL_ASSESSMENT.json'))
        records = [validation.base.read_json(p) for p in (parent/'runs').glob('*/*/*/*/result.json')]
        self.assertEqual(validation.historical_counts(records),dict(new_neural_fits=8,adopted_neural_fits=4,
                                                                  new_identity_records=8,adopted_identity_records=4))

    def test_partition_run_keeps_eval_reference_out_of_fit_and_aligned_in_prediction(self):
        a,_ = fixture()
        a.update(y_val=a['y_train'][:11].copy(),pairs_val=a['pairs_train'][:,:11].copy())
        p = protocol()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rd = out/'references/MiDrug/7'
            rd.mkdir(parents=True)
            anchor = np.linspace(-3,3,len(a['y_val']))
            np.savez_compressed(rd/'scores.npz',y=a['y_val'],pairs=a['pairs_val'],anchor=anchor,tabicl=anchor)
            validation.base.write_json(rd/'result.json',dict(data_protocol_sha256='fixture',
                files={'scores.npz':validation.base.digest(rd/'scores.npz')}))
            manifest = out/'offset.json'
            manifest.write_text('{}')
            dp = dict(split_seeds=[7],sha256='fixture')
            cp = dict(training_parameters=p,label='fixture',sha256='fixture',model_seed=17,
                      nested_reference='fixture',recipes={'MiDrug':{'full':dict(lr=.001,epochs=2)}})
            original = experiment.fit
            def fixed_fit(training,*args,**kwargs):
                self.assertTrue(set(training) <= experiment.TRAIN_KEYS)
                self.assertNotIn('y_val',training)
                return original(training,*args,**kwargs)
            with patch.object(validation,'checked',return_value=(dp,cp)), \
                 patch.object(validation.shared.inputs,'arrays',return_value=(a,None)), \
                 patch.object(experiment,'offset_arrays',return_value=(np.zeros_like(a['y_train'],dtype=np.float64),manifest)), \
                 patch.object(experiment,'fit',side_effect=fixed_fit):
                validation.run(out,'variational_feature_learning','MiDrug','cpu')
            result = validation.base.read_json(out/'runs/fixture/MiDrug/7/full/result.json')
            self.assertEqual(result['training_steps'],2)
            self.assertLessEqual(result['quadrature_reference_max_abs'],3e-5)
            scores = experiment.common.load(out/'runs/fixture/MiDrug/7/full/scores.npz')
            np.testing.assert_array_equal(scores['anchor'],anchor)
            np.testing.assert_array_equal(scores['anchor_composed'],anchor+.1*scores['standalone'])


if __name__ == '__main__':
    unittest.main()
