"""Synthetic end-to-end adoption, independent replay, and corruption checks."""
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
import torch

from improve import nonlinear_pair_field_v1 as core
from improve import test_nonlinear_pair_field_v1 as mathematics
from improve import tabular_foundation as base
from improve.research.p_energy_next_20260916 import nonlinear_field_study_v1 as study

HERE = Path(__file__).resolve().parent / 'research/p_energy_next_20260916'


class StudyChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.probe, self.output = self.root / 'probe', self.root / 'out'
        self.probe.mkdir()
        matrix, graph, ids, y, offset = mathematics.fixture()
        self.a = dict(pairs_train=np.stack((ids // 2, ids % 2)), y_train=y,
                      folds=np.array([0, 1, 0, 1]), anchor_oof=offset, nd=np.array(2))
        self.manifest = self.root / 'manifest.json'
        base.write_json(self.manifest, dict(synthetic=True))
        fields, traces, initial = {}, {}, None
        for arm, exponent in (('quadratic', 2.), ('full', 3.)):
            field, trace = core.solve(graph, ids, y, offset, exponent, initial=initial)
            fields[arm], traces[arm], initial = field.numpy(), trace, field
        np.savez_compressed(self.probe / 'fields.npz', **fields)
        base.write_json(self.probe / 'traces.json', traces)
        base.write_json(self.probe / 'graph.json', dict(synthetic=True))
        probe = base.seal(dict(files={name: base.digest(self.probe / name)
            for name in ('fields.npz', 'traces.json', 'graph.json')}))
        base.write_json(self.root / 'NONLINEAR_FIELD_SMOKE_V1.json', probe)
        self.p = dict(sha256='synthetic-test', parameters=core.parameters(),
            adopted_probe_sha256=probe['sha256'], historical_reference_seeds=[666, 0, 1], controls=['quadratic'])
        query = np.array([1, 3, 5, 7])
        self.source = self.root / 'query.npz'
        np.savez_compressed(self.source, pairs_val=np.stack((query // 2, query % 2)),
                            y_val=np.array([0., 1., 1., 0.]), nd=np.array(2))
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(study, 'HERE', self.root))
        stack.enter_context(patch.object(study, 'OUT', self.output))
        stack.enter_context(patch.object(study, 'PROBE', self.probe))
        stack.enter_context(patch.object(study, 'training_arrays', return_value=(self.a, self.manifest)))
        stack.enter_context(patch.object(core, 'training_graph', return_value=(sparse.csr_matrix(matrix), dict(synthetic=True))))
        stack.enter_context(patch.object(study, 'entry', return_value=(self.source, None)))
        ln = dict(y=np.array([0., 1., 0., 1.]), pairs=np.array([[0, 1, 2, 3], [0, 0, 0, 0]]),
                  anchor=np.array([-2., 2., -1., 1.]))
        stack.enter_context(patch.object(study, 'reference_scores', return_value=(np.array([-.4, .8, 1.2, -.7]), ln)))

    def test_adoption_replay_and_exact_accounting(self):
        with patch.object(core, 'fit', side_effect=AssertionError('Adoption must not refit')), redirect_stdout(io.StringIO()):
            record = study.train('historical', self.p)
            result = study.evaluate('historical', self.p)
            study.verify_result('historical', self.p)
        self.assertEqual(record['new_numerical_solves'], 0)
        self.assertEqual(record['adopted_numerical_solves'], 2)
        self.assertEqual(len(result['contexts']), 3)
        self.assertTrue(result['context_count_is_not_fit_count'])
        self.assertEqual(result['query_order_and_batch_max_error'], 0.)

    def test_stationarity_corruption_blocks_evaluation_access(self):
        with redirect_stdout(io.StringIO()):
            study.train('historical', self.p)
        destination = self.output / 'runs/historical'
        fields = study.read(destination / 'fields.npz', study.ARMS)
        fields['full'][1] += .1
        np.savez_compressed(destination / 'fields.npz', **fields)
        record = base.read_json(destination / 'TRAINING.json')
        record.pop('sha256')
        record['files']['fields.npz'] = base.digest(destination / 'fields.npz')
        base.write_json(destination / 'TRAINING.json', base.seal(record))
        original_read, evaluation_reads = study.read, []
        def guarded_read(path, keys):
            if set(keys) & {'y_val', 'pairs_val'}:
                evaluation_reads.append(str(path))
                raise AssertionError('Evaluation read preceded stationarity check')
            return original_read(path, keys)
        with patch.object(study, 'read', side_effect=guarded_read):
            with self.assertRaisesRegex(ValueError, 'not stationary'):
                study.evaluate('historical', self.p)
        self.assertFalse(evaluation_reads)


if __name__ == '__main__':
    path = HERE / 'NONLINEAR_FIELD_STUDY_UNIT_V1.json'
    if path.exists():
        raise ValueError('Preserve existing pipeline check evidence')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StudyChecks))
    base.write_json(path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        success=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        source_sha256={str(Path(module)): base.digest(module)
            for module in (__file__, study.__file__, core.__file__, mathematics.__file__)},
        synthetic_data_only=True, real_evaluation_labels_read=False, new_formal_fits=0)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
