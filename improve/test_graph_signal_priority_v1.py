"""Leakage, CUDA scope, and inherited forward/reverse semantics."""
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch

from improve import graph_signal_priority_v1 as study
from improve import graph_signal_experiment as original


def fixture():
    return dict(pairs_train=np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                                     [0, 1, 0, 1, 1, 2, 0, 2, 1, 2]]),
        y_train=np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=np.float32),
        folds=np.repeat(np.arange(5), 2), kr=np.eye(5, dtype=np.float32),
        kd=np.eye(3, dtype=np.float32))


class PriorityDiffusionTests(TestCase):
    def test_current_query_labels_never_enter_graph_or_spectrum(self):
        a = fixture()
        for fold in range(5):
            changed = {key: value.copy() for key, value in a.items()}
            changed['y_train'][a['folds'] == fold] = 1 - changed['y_train'][a['folds'] == fold]
            left, right = study.support(a, fold), study.support(changed, fold)
            np.testing.assert_array_equal(left, right)
            pairs = a['pairs_train'][:, a['folds'] == fold]
            self.assertFalse(left[pairs[0], pairs[1]].any())
            for variant in study.VARIANTS:
                l = study.core.spectrum(left, a['kd'], variant, 3)
                r = study.core.spectrum(right, a['kd'], variant, 3)
                for x, y in zip(l, r):
                    np.testing.assert_array_equal(x, y)

    def test_training_rejects_evaluation_inputs_and_unauthorized_gpu(self):
        a = fixture()
        with self.assertRaises(ValueError):
            study.support(dict(a, y_val=np.zeros(1)), 0)
        for device in ('cuda:0', 'cuda:2', 'cpu'):
            with self.assertRaises(ValueError):
                study.fit(a, 0, study.configuration('annotation'), 6, device)

    def test_two_readouts_match_old_predictor_and_reload(self):
        a = fixture()
        cfg = study.configuration('annotation')
        cfg.update(epochs=2, width=4, rank=3)
        # torch.manual_seed calls manual_seed_all internally; this execution must not.
        with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('All-GPU seeding')):
            model, trace, detail = study.fit(a, 0, cfg, 8, 'cpu', smoke=True)
        graph = study.support(a, 0)
        actual = study.predict(model, graph)
        expected = original.predict(model, torch.as_tensor(graph))
        loaded = study.old.construct(graph, a['kr'], a['kd'], cfg, 'cpu')
        loaded.load_state_dict(model.state_dict())
        replay = study.predict(loaded, graph)
        for kind in study.READOUTS:
            np.testing.assert_array_equal(actual[kind], expected[kind])
            np.testing.assert_array_equal(actual[kind], replay[kind])
        self.assertEqual(trace[-1]['epoch'], 2)
        self.assertFalse(detail['checkpoint_selected_on_evaluation'])

    def test_recipe_and_calibration_are_the_inherited_rules(self):
        a = fixture()
        score = np.linspace(-2, 2, 10)
        calibration = study.old.fit_calibration(score, a['y_train'])
        query = np.array([-1., .5, 1., 3.])
        full = study.old.apply_calibration(query, calibration)
        np.testing.assert_array_equal(full, study.old.apply_calibration(query[::-1], calibration)[::-1])
        np.testing.assert_array_equal(full, np.concatenate([
            study.old.apply_calibration(query[:2], calibration),
            study.old.apply_calibration(query[2:], calibration)]))
        for variant in study.VARIANTS:
            cfg = study.configuration(variant)
            self.assertEqual((cfg['epochs'], cfg['steps'], cfg['width'], cfg['rank']), (200, 3, 64, 128))


if __name__ == '__main__':
    main()
