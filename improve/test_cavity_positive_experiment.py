import unittest

import numpy as np
import torch

from improve import cavity_positive_experiment as study


class PositiveCavityTest(unittest.TestCase):
    def test_matched_arms_have_identical_inputs_and_initial_parameters(self):
        a = dict(edges=np.array([[0, 1], [2, 3]]), relation=np.array([0, 0]),
                 target_codes=np.array([0, 3]), node_counts=np.array([2, 2]),
                 pairs_train=np.array([[0, 1, 0, 1], [0, 1, 1, 0]]),
                 y_train=np.array([1, 1, 0, 0]), nd=np.array(2), pair_offset=np.array([0, 2]),
                 folds=np.array([0, 1, 1, 0]), x_0=np.ones((2, 3), dtype=np.float32),
                 x_1=np.ones((2, 4), dtype=np.float32))
        p = dict(width=8, depth=1, dropout=.1, folds=2)
        states, graphs = [], []
        for kind in study.KINDS:
            torch.manual_seed(91)
            model, _, contexts, _ = study.setup(a, p, kind, 'cpu')
            states.append(model.state_dict())
            graphs.append(contexts)
        for state in states[1:]:
            for name in states[0]:
                torch.testing.assert_close(state[name], states[0][name], atol=0., rtol=0.)
        for contexts in graphs[1:]:
            for first, second in zip(graphs[0], contexts):
                for x, y in zip(first[:4], second[:4]):
                    torch.testing.assert_close(x, y, atol=0., rtol=0.)
                for x, y in zip(first[4], second[4]):
                    torch.testing.assert_close(x, y, atol=0., rtol=0.)


if __name__ == '__main__':
    unittest.main()
