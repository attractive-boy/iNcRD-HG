"""Numerical inverse and validation-only training parity checks on toy data."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch_geometric.data import HeteroData

from improve.expression_scale_mechanism import (block_shuffle_indices,
    build_model, reconstruct_from_moments, state_digest, train_val_only,
    validate_array_keys)
from improve.first_round_features import run_neural


class ScaleMechanismTests(unittest.TestCase):
    def test_minmax_collision_and_inverse_including_epsilon_and_constant(self):
        raw = np.array([[1., 2., 3.], [12., 14., 16.], [7., 7., 7.]])
        ideal = (raw[:2] - raw[:2].min(1, keepdims=True)) / raw[:2].ptp(1, keepdims=True)
        np.testing.assert_array_equal(ideal[0], ideal[1])
        self.assertNotEqual(raw[0].mean(), raw[1].mean())
        self.assertNotEqual(raw[0].std(), raw[1].std())
        # Also test a large epsilon; the inversion is for any affine map.
        for epsilon in (1e-8, 0.25):
            normalized = (raw - raw.min(1, keepdims=True)) / (raw.ptp(1, keepdims=True) + epsilon)
            recovered = reconstruct_from_moments(normalized, raw.mean(1), raw.std(1))
            np.testing.assert_allclose(recovered, raw, atol=1e-13, rtol=0)
        with self.assertRaises(ValueError):
            reconstruct_from_moments([[0., 0.]], [1.], [2.])

    def test_joint_shuffle_preserves_fit_groups_and_statistic_association(self):
        fit = np.array([0, 2, 3, 6, 8])
        index = block_shuffle_indices(10, fit)
        np.testing.assert_array_equal(np.sort(index), np.arange(10))
        np.testing.assert_array_equal(np.sort(index[fit]), fit)
        np.testing.assert_array_equal(index, block_shuffle_indices(10, fit))
        values = np.stack([np.arange(10), 2*np.arange(10), 3*np.arange(10)], axis=1)
        shuffled = values[index]
        np.testing.assert_array_equal(shuffled[:, 1], shuffled[:, 0] * 2)
        self.assertGreater(int((index != np.arange(10)).sum()), 0)

    @staticmethod
    def toy_inputs():
        data = HeteroData()
        data['lncRNA'].x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
        data['drug'].x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 15
        edges = torch.tensor([[0, 1, 2, 0, 2], [0, 1, 2, 3, 1]])
        data['lncRNA', 'LncDrug', 'drug'].edge_index = edges
        data['drug', 'rev_LncDrug', 'lncRNA'].edge_index = edges.flip(0)
        data['lncRNA', 'LncLnc', 'lncRNA'].edge_index = torch.stack([torch.arange(3)] * 2)
        data['drug', 'DrugDrug', 'drug'].edge_index = torch.stack([torch.arange(4)] * 2)
        arrays = {'x_' + nt: data[nt].x.numpy().copy() for nt in data.node_types}
        arrays.update({'weight_' + str(i): np.ones(data[k].edge_index.shape[1], dtype=np.float32)
                       for i, k in enumerate(data.edge_types)})
        arrays.update(pairs_train=np.array([[0, 1, 2, 0, 1, 2], [0, 1, 2, 3, 0, 3]], dtype=np.int64),
                      pairs_val=np.array([[0, 1, 2, 0], [1, 2, 3, 2]], dtype=np.int64),
                      y_train=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
                      y_val=np.array([0, 1, 0, 1], dtype=np.float32))
        return data, arrays

    def test_same_width_initialization_does_not_depend_on_input_values(self):
        data, arrays = self.toy_inputs()
        model, _ = build_model(arrays, data, 'LncDrug', 'cpu')
        changed = dict(arrays, x_lncRNA=arrays['x_lncRNA'] * 3 + 2)
        other, _ = build_model(changed, data, 'LncDrug', 'cpu')
        self.assertEqual(state_digest(model), state_digest(other))

    def test_val_only_trainer_matches_reference_on_cpu_toy_graph(self):
        torch.set_num_threads(2)
        data, arrays = self.toy_inputs()
        settings = dict(rna='lncRNA', train_epochs=3, lr=0.0005)
        config = {'identity': {'training': settings}}
        result, prediction, trace, state = train_val_only(arrays, data, 'LncDrug', config, 'cpu', 'unit')
        # The OLD runner requires a test fold. Use dummy copies of the toy
        # validation fixture only, never real experimental test inputs.
        old_arrays = dict(arrays, pairs_test=arrays['pairs_val'].copy(), y_test=arrays['y_val'].copy())
        old_spec = {'task_settings': {'LncDrug': settings}, 'gnn': {'alpha': 0.99}}
        old_result, old_prediction, _, old_trace, old_state = run_neural(
            SimpleNamespace(device='cpu'), old_spec, 'LncDrug', 'gnn', 666, old_arrays, data)
        np.testing.assert_array_equal(prediction, old_prediction)
        self.assertEqual(result['selected_epoch'], old_result['selected_epoch'])
        self.assertEqual(trace, old_trace)
        self.assertEqual(state.keys(), old_state.keys())
        for key in state:
            torch.testing.assert_close(state[key], old_state[key], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            validate_array_keys(old_arrays)


if __name__ == '__main__':
    unittest.main()
