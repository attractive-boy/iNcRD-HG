"""Outer executor must use training-only nested offsets and frozen primaries."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from improve import anchor_residual_outer_v2 as outer
from improve.test_nested_anchor_v2 import fixture


class OuterBoundaryTests(unittest.TestCase):
    def test_offsets_never_receive_outer_outcomes(self):
        a, _ = fixture()
        a.update(y_val=np.array([1., 0.]), pairs_val=np.array([[0, 1], [2, 3]]),
                 anchor_666=np.array([9., -9.]), offset_nodes=np.ones((5, 23)))
        def offsets(cache, task, split, training):
            self.assertEqual(split, 'outer')
            self.assertEqual(set(training), outer.experiment.nested.TRAIN_KEYS)
            return np.arange(len(training['y_train']), dtype=float), 'manifest'
        with patch.object(outer.experiment, 'offset_arrays', side_effect=offsets):
            training, path = outer.training_inputs(a, 'cache', 'MiDrug')
        self.assertEqual(path, 'manifest')
        self.assertEqual(set(training), outer.experiment.TRAIN_KEYS)
        self.assertNotIn('y_val', training)
        self.assertNotIn('pairs_val', training)
        self.assertNotIn('anchor_666', training)

    def test_single_task_and_family_registry(self):
        self.assertEqual(outer.label_for('differential_attention'), 'anchor_priority_differential_attention_v1')
        with self.assertRaises(ValueError):
            outer.label_for('conflict_free')

    def test_identity_and_neural_execution_accounting(self):
        p = dict(lnc_policy='identity_strong_reference')
        identity = dict(trained_neural_model=False, epoch=0, parameters=0, trainable_parameters=0,
                        training_steps=0, gradient_evaluations=0)
        self.assertFalse(outer.audit_execution_record(identity, 'LncDrug', {'epochs': 0}, p))
        with self.assertRaisesRegex(ValueError, 'Identity'):
            outer.audit_execution_record(dict(identity, training_steps=1), 'LncDrug', {'epochs': 0}, p)
        neural = dict(trained_neural_model=True, training_steps=80, gradient_evaluations=80)
        self.assertTrue(outer.audit_execution_record(neural, 'MiDrug', {'epochs': 80}, p))
        with self.assertRaisesRegex(ValueError, 'budget'):
            outer.audit_execution_record(neural, 'MiDrug', {'epochs': 100}, p)

    def test_stale_or_partial_partition_evidence_cannot_advance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dp = dict(sha256='data', tasks=['MiDrug', 'LncDrug'], split_seeds=[7])
            cp = dict(sha256='candidate', label='fixture',
                      recipes={task: {'full': {}, 'control': {}} for task in dp['tasks']})
            assessment = dict(candidate_sha256=cp['sha256'], data_protocol_sha256=dp['sha256'],
                all_metrics_recomputed=True, additional_development_gate=True,
                performance_gate=True, primary_mechanism_gate=True, offset_null_gate=True,
                result_sha256={}, artifacts={})
            for task in dp['tasks']:
                for kind in cp['recipes'][task]:
                    dest = root / 'runs/fixture' / task / '7' / kind
                    dest.mkdir(parents=True)
                    payload = dest / 'scores.bin'
                    payload.write_bytes(b'paired predictions')
                    record = dict(candidate_sha256=cp['sha256'], data_protocol_sha256=dp['sha256'],
                                  files={'scores.bin': outer.base.digest(payload)})
                    outer.base.write_json(dest / 'result.json', record)
                    assessment['result_sha256'][str(dest / 'result.json')] = outer.base.digest(dest / 'result.json')
            report = root / 'reports/fixture'
            report.mkdir(parents=True)
            for name in ('metrics.csv', 'contrasts.csv', 'summary.csv', 'endpoint_uncertainty.json'):
                (report / name).write_text('verified output\n')
                assessment['artifacts'][name] = outer.base.digest(report / name)
            path = report / 'ASSESSMENT.json'
            outer.base.write_json(path, assessment)
            outer.verify_partition_report(path, dp, cp, root)
            changed = root / 'runs/fixture/MiDrug/7/full/scores.bin'
            changed.write_bytes(b'changed predictions')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                outer.verify_partition_report(path, dp, cp, root)
            changed.write_bytes(b'paired predictions')
            omitted = next(iter(assessment['result_sha256']))
            del assessment['result_sha256'][omitted]
            outer.base.write_json(path, assessment)
            with self.assertRaisesRegex(ValueError, 'missing or extra'):
                outer.verify_partition_report(path, dp, cp, root)


if __name__ == '__main__':
    unittest.main()
