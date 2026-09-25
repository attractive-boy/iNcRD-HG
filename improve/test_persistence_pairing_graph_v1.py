"""Joint tuple information, marginal oracle and exact parent-fit compatibility."""
from datetime import datetime, timezone
from pathlib import Path
import unittest

import numpy as np
import torch

from improve import anchor_spectral_persistence_experiment_v1 as experiment
from improve import persistence_pairing_graph_v1 as core
from improve import spectral_persistence_graph_v1 as parent
from improve import tabular_foundation as base
from improve.test_spectral_persistence_graph_v1 import example

HERE = Path(__file__).resolve().parent / 'research/persistence_next_20260916'
EVIDENCE = {}


def assert_nested_equal(case, actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
    elif isinstance(actual, dict):
        case.assertEqual(set(actual), set(expected))
        for key in actual:
            assert_nested_equal(case, actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        case.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            assert_nested_equal(case, a, b)
    else:
        case.assertEqual(actual, expected)


class PairingChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(213)

    def test_independent_marginal_oracle_and_piecewise_gradient(self):
        count = torch.tensor([3, 2, 4])
        slices = torch.cat((torch.zeros(1, dtype=torch.long), count.cumsum(0)))
        groups = torch.repeat_interleave(torch.arange(len(count)), count)
        raw = torch.randn(int(count.sum()), 3, dtype=torch.float64, requires_grad=True)
        def evaluate(value):
            tuples = torch.cat((value.new_zeros((len(value), 1)), value), -1)
            return core.marginal_tuple(tuples, groups, slices, count)
        actual = evaluate(raw)
        expected = np.concatenate([np.sort(np.c_[np.zeros(int(count[i])),
            raw.detach().numpy()[slices[i]:slices[i+1]]], axis=0) for i in range(len(count))])
        np.testing.assert_array_equal(actual.detach().numpy(), expected)
        self.assertTrue(torch.autograd.gradcheck(evaluate, (raw,), eps=1e-6, atol=1e-7, rtol=1e-5))
        permutation = torch.tensor([2, 0, 1, 4, 3, 7, 5, 8, 6])
        torch.testing.assert_close(evaluate(raw[permutation]), actual, atol=0., rtol=0.)
        split = torch.cat([core.marginal_tuple(actual[slices[i]:slices[i+1]],
            torch.zeros(int(count[i]), dtype=torch.long), torch.tensor([0, count[i]]), count[i:i+1])
            for i in range(len(count))])
        torch.testing.assert_close(split, actual, atol=0., rtol=0.)
        EVIDENCE['marginal_oracle'] = dict(queries=3, every_coordinate_and_cardinality_preserved=True,
            node_permutation_and_query_batching=True, piecewise_finite_differences=True)

    def test_same_marginals_different_joint_counterexample(self):
        first = torch.tensor([[0., .2, .1, .3], [0., .5, .4, .6], [0., 1., .8, .9]], dtype=torch.float64)
        second = first.clone()
        second[:, 2] = second[torch.tensor([1, 0, 2]), 2]
        grouping, slices, counts = torch.zeros(3, dtype=torch.long), torch.tensor([0, 3]), torch.tensor([3])
        a = core.marginal_tuple(first, grouping, slices, counts)
        b = core.marginal_tuple(second, grouping, slices, counts)
        torch.testing.assert_close(a, b, atol=0., rtol=0.)
        first_joint, second_joint = float((first[:, 1] * first[:, 2]).sum()), float((second[:, 1] * second[:, 2]).sum())
        self.assertNotEqual(first_joint, second_joint)
        EVIDENCE['joint_counterexample'] = dict(first_joint_moment=first_joint, second_joint_moment=second_joint,
            marginal_control_identical=True, no_claim_synthetic_control_is_valid_graph=True)

    def test_exact_primary_parent_training_and_equal_control_rng(self):
        a = example()
        a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
        p = dict(smoke=True, family='persistence_pairing', implementation_module='persistence_pairing_graph_v1',
            source_sha256={'persistence_pairing_graph_v1.py': base.digest(core.__file__)},
            core_parameters=dict(core.parameters(), width=8, depth=2, dropout=.1), kinds=list(core.KINDS),
            epochs=6, check_interval=3, weight_decay=.001, strong_weight=.1)
        previous = dict(p, family='spectral_persistence', implementation_module='spectral_persistence_graph_v1',
            source_sha256={'spectral_persistence_graph_v1.py': base.digest(parent.__file__)}, kinds=list(parent.KINDS))
        old, _, old_trace, old_detail = experiment.fit(a, previous, 'no_spectrum', .003, 223, 'cpu', fixed_epochs=6)
        model, _, trace, detail = experiment.fit(a, p, 'full', .003, 223, 'cpu', fixed_epochs=6)
        assert_nested_equal(self, model.state_dict(), old.state_dict())
        assert_nested_equal(self, model.training_final_state, old.training_final_state)
        self.assertEqual(trace, old_trace)
        control, _, control_trace, control_detail = experiment.fit(a, p, 'marginal_only', .003, 223, 'cpu', fixed_epochs=6)
        for key in ('training_steps', 'gradient_evaluations', 'optimizer_steps', 'parameters', 'trainable_parameters'):
            self.assertEqual(detail[key], control_detail[key])
        torch.testing.assert_close(model.training_final_state['cpu_rng'], control.training_final_state['cpu_rng'], atol=0., rtol=0.)
        self.assertEqual(detail['training_steps'], 6)
        self.assertTrue(all('anchor_composed' not in row for row in trace + control_trace))
        EVIDENCE['parent_training_compatibility'] = dict(synthetic_fits=3, steps_each=6,
            primary_weights_optimizer_rng_and_trace_bitwise_identical=True, equal_control_budget_and_rng=True)

    def test_graph_boundary_identity_and_nonzero_replay(self):
        a, p = example(), dict(core.parameters(), width=8, depth=2, dropout=.1)
        initial = None
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(217)
            model, xs, graphs, folds = core.setup(a, p, kind, 'cpu')
            if initial is None:
                initial = experiment.cpu_copy(model.state_dict())
            else:
                assert_nested_equal(self, model.state_dict(), initial)
            model.eval()
            with torch.no_grad():
                model.readout[-1].weight.normal_(std=.2)
            ids = folds[0].numpy()
            query = torch.as_tensor(a['pairs_train'][:, ids] + a['pair_offset'][:, None])
            original = model(xs, graphs[0], query)
            reversed_queries = model(xs, graphs[0], query.flip(1)).flip(0)
            batched = torch.cat([model(xs, graphs[0], query[:, i:i+1]) for i in range(query.shape[1])])
            torch.testing.assert_close(original, reversed_queries, atol=2e-7, rtol=2e-6)
            torch.testing.assert_close(original, batched, atol=2e-7, rtol=2e-6)
            self.assertGreater(float(original.abs().max()), 1e-8)
            for u, v in query.T.tolist():
                self.assertEqual(graphs[0].adjacency[u, v], 0.)
                self.assertEqual(graphs[0].adjacency[v, u], 0.)
            changed = {key: value.copy() for key, value in a.items()}
            changed['y_train'][ids] = 1. - changed['y_train'][ids]
            _, _, other, _ = core.setup(changed, p, kind, 'cpu')
            self.assertEqual((graphs[0].adjacency != other[0].adjacency).nnz, 0)
            assert_nested_equal(self, graphs[0].neural_graph, other[0].neural_graph)
            with self.assertRaisesRegex(ValueError, 'training arrays only'):
                core.setup(dict(a, y_val=np.ones(1)), p, kind, 'cpu')
        a['pair_offset'] = np.array([0, 5])
        identity, xs, graphs, _ = core.setup(a, p, 'full', 'cpu')
        self.assertFalse(list(identity.parameters()))
        self.assertTrue(all(graph is None for graph in graphs))
        torch.testing.assert_close(identity(xs, graphs[0], query), torch.zeros(query.shape[1]), atol=0., rtol=0.)
        EVIDENCE['boundaries'] = dict(query_fact_and_reverse_removed=True, query_label_invariant=True,
            training_arrays_only=True, nonzero_reorder_and_batch_replay=True, lnc_exact_identity=True)


if __name__ == '__main__':
    output = HERE / 'PERSISTENCE_PAIRING_UNIT_V1.json'
    if output.exists():
        raise ValueError('Preserve previous implementation evidence')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PairingChecks))
    sources = [Path(__file__), Path(core.__file__), Path(parent.__file__), Path(experiment.__file__)]
    base.write_json(output, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        success=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        evidence=EVIDENCE, source_sha256={str(path): base.digest(path) for path in sources},
        new_evaluation_labels_read=False, new_formal_fits=0, prior_parent_outcomes_seen=True)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
