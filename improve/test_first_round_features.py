"""Mechanism, leakage and comparison checks for the first-round experiment."""
import argparse
import unittest

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from improve.first_round_features import (METRICS, expected_tags, paired_summary,
    protocol_spec, run_neural)
from improve.first_round_inputs import (COMPARISONS, FOLDS, STRUCTURE_COLUMNS,
    clean_training_pairs, crossfit_structure, fit_expression_array, incoming_similarity, pair_features,
    structure_features)
from improve.network_feature_inputs import graph_weights


class FirstRoundTests(unittest.TestCase):
    def test_training_cleanup_preserves_evaluation_and_records_every_removal(self):
        rows = {
            'train': (np.array([[0, 0, 1, 2, 3], [0, 0, 1, 2, 3]]), np.array([1, 1, 1, 0, 0])),
            'val': (np.array([[1, 4], [1, 4]]), np.array([1, 0])),
            'test': (np.array([[5, 6], [5, 6]]), np.array([1, 0])),
        }
        cleaned, audit = clean_training_pairs(rows)
        np.testing.assert_array_equal(cleaned['train'][0], rows['train'][0][:, [0, 3, 4]])
        for fold in ('val', 'test'):
            for original, result in zip(rows[fold], cleaned[fold]):
                np.testing.assert_array_equal(original, result)
        self.assertEqual([r['row'] for r in audit['removed_rows']], [1, 2])
        self.assertEqual([r['reason'] for r in audit['removed_rows']],
                         ['duplicate_training_pair', 'overlaps_evaluation'])
        self.assertEqual(rows['train'][0].shape[1], 5)

    def test_exactly_three_independent_comparisons(self):
        spec = protocol_spec()
        self.assertEqual(len(spec['comparisons']), 3)
        self.assertEqual(spec['models'], ['gnn', 'mlp', 'extratrees'])
        self.assertEqual(len(expected_tags(spec)), 150)
        self.assertEqual(len(set(expected_tags(spec))), 150)
        self.assertEqual(COMPARISONS['drug_pooling'], ('drug_unmasked', 'drug_masked'))
        self.assertNotIn('expr_ae64', spec['arms'])

    def test_held_out_expression_cannot_change_fit_or_training_representation(self):
        rng = np.random.RandomState(5)
        raw = rng.uniform(0, 10, (15, 9)).astype(np.float32)
        raw[::2, ::3] = 0
        fit = np.arange(10)
        first, parameters = fit_expression_array(raw, fit, components=3)
        changed = raw.copy()
        changed[10:] += 10000
        second, changed_parameters = fit_expression_array(changed, fit, components=3)
        np.testing.assert_allclose(first[fit], second[fit], rtol=0, atol=0)
        for name in parameters:
            np.testing.assert_array_equal(parameters[name], changed_parameters[name])
        recovered = first[:, -3:] * parameters['statistics_scale'] + parameters['statistics_mean']
        expected = np.column_stack([raw.mean(1), raw.std(1), (raw == 0).mean(1)])
        np.testing.assert_allclose(recovered, expected, rtol=1e-6, atol=1e-6)

    def test_simple_models_receive_every_raw_endpoint_coordinate(self):
        nodes = {'lncRNA': np.arange(15, dtype=np.float32).reshape(5, 3),
                 'drug': np.arange(12, dtype=np.float32).reshape(6, 2)}
        pairs = np.array([[0, 1, 2, 3], [2, 3, 4, 5]])
        rows = {f: (pairs, np.array([0, 1, 0, 1], dtype=np.float32)) for f in FOLDS}
        structures = {f: np.ones((4, len(STRUCTURE_COLUMNS)), dtype=np.float32) for f in FOLDS}
        features, scaler, layout = pair_features(rows, nodes, 'lncRNA', structures)
        recovered = features['train'] * scaler['scale'] + scaler['mean']
        np.testing.assert_allclose(recovered[:, :3], nodes['lncRNA'][pairs[0]], atol=1e-6)
        np.testing.assert_allclose(recovered[:, 3:5], nodes['drug'][pairs[1]], atol=1e-6)
        self.assertEqual(layout['rna_vector'], [0, 3])
        self.assertEqual(layout['drug_vector'], [3, 5])
        changed = dict(rows, test=(np.array([[4, 4, 4, 4], [5, 5, 5, 5]]), rows['test'][1]))
        _, other_scaler, _ = pair_features(changed, nodes, 'lncRNA', structures)
        np.testing.assert_array_equal(scaler['mean'], other_scaler['mean'])
        np.testing.assert_array_equal(scaler['scale'], other_scaler['scale'])

    def test_continuous_weights_reach_simple_models_without_adding_edges(self):
        key = ('drug', 'DrugDrug', 'drug')
        data = HeteroData()
        data['drug'].x = torch.ones(3, 2)
        data[key].edge_index = torch.tensor([[0, 1, 2, 0, 2, 1], [0, 0, 0, 1, 1, 2]])
        matrix = np.array([[1, .4, .7], [.4, 1, .6], [.7, .6, 1]], dtype=np.float32)
        binary = graph_weights(data, {key: matrix}, False)
        weighted = graph_weights(data, {key: matrix}, True)
        B = incoming_similarity(data, key, binary)
        W = incoming_similarity(data, key, weighted)
        np.testing.assert_array_equal(B != 0, W != 0)
        self.assertFalse(np.allclose(B, W))
        # Check direction explicitly: destination 0 receives three messages,
        # with weights (1, .4, .7) rescaled to sum to 3; self excluded in table.
        self.assertAlmostEqual(float(W[0, 1]), .4 * 3 / 2.1, places=6)
        self.assertEqual(float(W[0, 0]), 0.0)
        A = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.float32)
        pairs = np.array([[0, 1, 2], [1, 2, 0]])
        f0 = structure_features(pairs, A, B, B, A, A)
        f1 = structure_features(pairs, A, W, W, A, A)
        np.testing.assert_array_equal(f0[:, :3], f1[:, :3])
        self.assertFalse(np.allclose(f0[:, 3:8], f1[:, 3:8]))
        expected = A @ W.T
        np.testing.assert_allclose(f1[:, 5], expected[pairs[0], pairs[1]])

    def test_crossfit_statistics_remove_entire_held_positive_fold(self):
        pairs = np.array([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 0, 3]])
        y = np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.float32)
        A = np.zeros((4, 4), dtype=np.float32)
        A[pairs[0, y == 1], pairs[1, y == 1]] = 1
        original = A.copy()
        rows = {f: (pairs, y) for f in FOLDS}
        S = np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)
        bridge = np.zeros_like(A)
        features, assignment = crossfit_structure(rows, A, S, S, bridge, bridge, n_folds=2)
        for fold in range(2):
            take = assignment == fold
            Ak = A.copy()
            held = pairs[:, take & (y == 1)]
            Ak[held[0], held[1]] = 0
            expected = structure_features(pairs[:, take], Ak, S, S, bridge, bridge)
            np.testing.assert_array_equal(features['train'][take], expected)
        np.testing.assert_array_equal(A, original)
        np.testing.assert_array_equal(features['test'], structure_features(pairs, A, S, S, bridge, bridge))

    def test_test_labels_and_features_cannot_select_checkpoint(self):
        rng = np.random.RandomState(7)
        arrays = {}
        for fold in FOLDS:
            arrays['phi_' + fold] = rng.normal(size=(12, 5)).astype(np.float32)
            arrays['y_' + fold] = np.tile([0, 1], 6).astype(np.float32)
        args = argparse.Namespace(device='cpu')
        spec = protocol_spec(smoke=True)
        spec['task_settings']['LncDrug']['train_epochs'] = 4
        a = run_neural(args, spec, 'LncDrug', 'mlp', 666, arrays, None)
        changed = dict(arrays, y_test=1 - arrays['y_test'], phi_test=arrays['phi_test'] * -100)
        b = run_neural(args, spec, 'LncDrug', 'mlp', 666, changed, None)
        self.assertEqual(a[0]['selected_epoch'], b[0]['selected_epoch'])
        self.assertEqual(a[0]['selection_criterion'], b[0]['selection_criterion'])
        np.testing.assert_array_equal(a[1], b[1])
        for name in a[4]:
            torch.testing.assert_close(a[4][name], b[4][name], rtol=0, atol=0)

    def test_pooling_delta_uses_matched_control_and_test_does_not_drive_screen(self):
        spec = protocol_spec()
        rows = []
        for task in spec['tasks']:
            for model in spec['models']:
                for arm in spec['arms']:
                    for seed in spec['seeds']:
                        val = {'base': .99, 'drug_unmasked': .50, 'drug_masked': .60,
                               'expr_pca64_stats': .98, 'edge_weight': .97}[arm]
                        test = .01 if arm == 'drug_masked' else .99
                        row = dict(task=task, model=model, arm=arm, seed=seed)
                        row.update({m: val if m.startswith('val') else test for m in METRICS})
                        rows.append(row)
        _, paired, evidence = paired_summary(pd.DataFrame(rows), spec)
        pooling = paired[paired.comparison == 'drug_pooling']
        np.testing.assert_allclose(pooling.val_criterion_delta_mean, .10)
        self.assertTrue(pooling.validation_screen_pass.all())
        self.assertTrue((pooling.test_criterion_delta_mean < 0).all())
        self.assertEqual(len(paired), 18)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
