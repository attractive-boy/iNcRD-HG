"""Enumerated probability, gradient, schema and training checks for typed SOCS."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from improve import sos_circuit_graph_v1 as core
from improve import anchor_sos_circuit_experiment_v1 as experiment
from improve import tabular_foundation as base
from improve.test_nested_anchor_v2 import fixture

HERE = Path(__file__).resolve().parent/'research/structured_probabilistic_20260916'
EVIDENCE = {}


def arrays():
    a, _ = fixture()
    a['anchor_oof'] = np.linspace(-.5, .5, len(a['y_train']))
    return a


def small_protocol():
    return dict(smoke=True, family='sos_circuit', implementation_module='sos_circuit_graph_v1',
        source_sha256={'sos_circuit_graph_v1.py':base.digest(core.__file__)},
        core_parameters=dict(core.parameters(), components=2, component_rank=3), kinds=list(core.KINDS),
        epochs=5, check_interval=5, weight_decay=.001, strong_weight=.1)


class CircuitChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(278)

    def test_gram_marginals_and_gradients_match_enumeration(self):
        s = torch.randn(4, 2, 3, dtype=torch.float64, requires_grad=True)
        r = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        o = torch.randn(5, 2, 3, dtype=torch.float64, requires_grad=True)
        errors = {}
        for kind in core.KINDS:
            # Independently expand the CP sum into Cartesian pairs and scalar
            # products; this does not call the implementation's mass function.
            terms = []
            for left in s:
                for right in o:
                    factors = left*r*right
                    if kind == 'single_square': value = sum(factors.flatten())**2
                    elif kind == 'monotone': value = sum(v*v for v in factors.flatten())
                    else: value = sum(sum(row)**2 for row in factors)
                    terms.append(value)
            dense = torch.stack(terms).sum()
            fast = core.marginal_mass(s, r, o, kind)
            torch.testing.assert_close(fast, dense, atol=2e-12, rtol=2e-12)
            gf = torch.autograd.grad(fast, (s, r, o), retain_graph=True)
            gd = torch.autograd.grad(dense, (s, r, o), retain_graph=True)
            for a, b in zip(gf, gd): torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)
            self.assertTrue(torch.autograd.gradcheck(lambda a, b, c:core.marginal_mass(a, b, c, kind),
                                                   (s, r, o), atol=1e-5, rtol=1e-4))
            errors[kind] = dict(normalizer=float((fast-dense).detach().abs()),
                gradient=max(float((a-b).abs().max()) for a, b in zip(gf, gd)))
        EVIDENCE['marginal_enumeration'] = errors

    def test_circuit_classes_and_phase_cancellation_are_distinct(self):
        s = torch.ones(1, 2, 2, dtype=torch.float64)
        r = torch.ones(2, 2, dtype=torch.float64)
        o = torch.tensor([[[2., -1.], [1., -2.]]], dtype=torch.float64)
        values = {k:float(core.nonnegative_mass(s, r, o, k)) for k in core.KINDS}
        self.assertEqual(values, dict(full=2., single_square=0., monotone=10., no_joint=2.))
        shifted = o.clone()
        shifted[:, 1] *= -1
        self.assertEqual(float(core.nonnegative_mass(s, r, shifted, 'full')), 2.)
        self.assertEqual(float(core.nonnegative_mass(s, r, shifted, 'single_square')), 4.)
        EVIDENCE['mechanism_counterexample'] = dict(values=values, component_sign_gauge=True,
            interference_within_squares=True, coherent_cross_component_control=True)

    def test_probability_normalization_schema_and_entity_relabeling(self):
        a, p = arrays(), small_protocol()
        for kind in core.KINDS:
            model, _, _, _ = core.setup(a, p['core_parameters'], kind, 'cpu')
            model.double()
            errors = []
            for relation, (left, right) in enumerate(model.schema):
                pairs = np.indices((model.counts[left], model.counts[right])).reshape(2, -1)
                pairs += np.array([model.offsets[left], model.offsets[right]])[:, None]
                pairs = torch.as_tensor(pairs)
                q = model.log_density(pairs, relation).exp()
                self.assertAlmostEqual(float(q.detach().sum()), 1., places=12)
                self.assertGreaterEqual(float(q.detach().min()), p['core_parameters']['uniform_mixture']/len(q))
                errors.append(abs(float(q.detach().sum())-1.))
            with self.assertRaisesRegex(ValueError, 'schema'):
                model.log_density(torch.tensor([[0], [0]]), model.target_relation)
            model.slope.data.fill_(.7)
            query = torch.as_tensor(a['pairs_train']+a['pair_offset'][:, None])
            score = model([], None, query)
            self.assertGreater(float(score.detach().abs().max()), 1e-6)
            reverse = model([], None, query.flip(1)).flip(0)
            chunks = torch.cat([model([], None, query[:, i:i+13]) for i in range(0, query.shape[1], 13)])
            torch.testing.assert_close(score, reverse, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(score, chunks, atol=1e-12, rtol=1e-12)
            renamed = copy.deepcopy(model)
            mapping = torch.arange(sum(model.counts))
            with torch.no_grad():
                for i in range(len(model.counts)):
                    lo, hi = model.offsets[i:i+2]
                    old = torch.randperm(hi-lo)+lo
                    mapping[old] = torch.arange(lo, hi)
                    renamed.subject[lo:hi].copy_(model.subject[old])
                    renamed.object[lo:hi].copy_(model.object[old])
            transformed = renamed([], None, mapping[query])
            torch.testing.assert_close(score, transformed, atol=1e-12, rtol=1e-12)
            EVIDENCE.setdefault('probability', {})[kind] = dict(max_sum_error=max(errors),
                query_permutation_and_batching=True, consistent_ID_relabeling=True, schema_rejection=True)

    def test_positive_and_sampled_zero_support_boundaries(self):
        a, p = arrays(), small_protocol()
        model, xs, graphs, folds = core.setup(a, p['core_parameters'], 'full', 'cpu')
        n = int(a['node_counts'].sum())
        queries = a['pairs_train']+a['pair_offset'][:, None]
        for fold, graph in enumerate(graphs):
            ids = folds[fold].numpy()
            forbidden = queries[0, ids]*n+queries[1, ids]
            for relation in (model.target_relation, model.zero_relation):
                edges = graph[relation].numpy()
                self.assertFalse(np.intersect1d(forbidden, edges[0]*n+edges[1]).size)
            altered = {k:v.copy() for k,v in a.items()}
            altered['y_train'][ids] = 1-altered['y_train'][ids]
            _, _, other_graphs, _ = core.setup(altered, p['core_parameters'], 'full', 'cpu')
            for left, right in zip(graph, other_graphs[fold]):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            core.setup(dict(a, y_val=np.zeros(2)), p['core_parameters'], 'full', 'cpu')
        EVIDENCE['boundaries'] = dict(positive_query_overlap=0, sampled_zero_query_overlap=0,
            omitted_fold_labels_invariant=True, evaluation_fields_rejected=True)

    def test_training_states_gradients_and_lnc_identity(self):
        a, p = arrays(), small_protocol()
        initial_state, initial_rng, final_rng = None, None, None
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(718)
            model, _, _, _ = core.setup(a, p['core_parameters'], kind, 'cpu')
            state = experiment.cpu_copy(model.state_dict())
            rng = torch.random.get_rng_state()
            query = torch.as_tensor(a['pairs_train']+a['pair_offset'][:, None])
            torch.testing.assert_close(model([], None, query), torch.zeros(query.shape[1], dtype=torch.float64), atol=0, rtol=0)
            if initial_state is None: initial_state, initial_rng = state, rng
            else:
                for name, v in state.items(): torch.testing.assert_close(v, initial_state[name], atol=0, rtol=0)
                torch.testing.assert_close(rng, initial_rng, atol=0, rtol=0)
            model, value, trace, detail = experiment.fit(a, p, kind, .003, 718, 'cpu', fixed_epochs=5)
            self.assertIsNone(value)
            self.assertFalse(any('anchor_composed' in r for r in trace))
            self.assertEqual(detail['circuit_diagnostics']['forwards'], 5)
            self.assertEqual(detail['circuit_diagnostics']['auxiliary_calls'], 5)
            for name, v in model.named_parameters():
                self.assertIsNotNone(v.grad, name)
                self.assertTrue(torch.isfinite(v.grad).all(), name)
            final = model.training_final_state
            self.assertEqual(final['steps'], 5)
            for row in final['optimizer']['state'].values(): self.assertEqual(int(row['step']), 5)
            if final_rng is None: final_rng = final['cpu_rng']
            else: torch.testing.assert_close(final['cpu_rng'], final_rng, atol=0, rtol=0)
        lnc = arrays()
        lnc['pair_offset'] = np.array([0, 14])
        for kind in core.KINDS:
            model, _, trace, detail = experiment.fit(lnc, p, kind, .003, 718, 'cpu', fixed_epochs=5)
            self.assertEqual(detail['training_steps'], 0)
            self.assertFalse(detail['trained_neural_model'])
            self.assertFalse(model.state_dict())
            self.assertFalse(trace)
        EVIDENCE['training'] = dict(common_initial_states=True, common_initial_and_final_rng=True,
            finite_all_parameter_gradients=True, exact_step_counts=True, lnc_identity_no_parameters=True)

    def test_partial_likelihood_failure_and_fixed_input_rejection(self):
        a, p = arrays(), small_protocol()
        original, counter = core.SumSquaresCircuit.auxiliary_nll, [0]
        def failing(model, graph):
            counter[0] += 1
            if counter[0] == 3: raise FloatingPointError('injected normalization failure')
            return original(model, graph)
        with patch.object(core.SumSquaresCircuit, 'auxiliary_nll', failing):
            with self.assertRaises(FloatingPointError) as caught:
                experiment.fit(a, p, 'full', .003, 718, 'cpu', fixed_epochs=5)
        error = caught.exception
        self.assertEqual(error.training_partial_state['steps'], 2)
        self.assertEqual(error.training_failure_context['current_step'], 3)
        self.assertEqual(error.training_failure_context['operation'], 'auxiliary_likelihood')
        for row in error.training_partial_state['optimizer']['state'].values(): self.assertEqual(int(row['step']), 2)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.fit(dict(a, y_val=np.zeros(2)), p, 'full', .003, 718, 'cpu', fixed_epochs=5)
        EVIDENCE['failure'] = dict(injected_step=3, completed_optimizer_steps=2, exact_state_saved=True)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CircuitChecks))
    path = HERE/'SOS_CIRCUIT_UNIT_V1.json'
    if path.exists(): raise ValueError('Preserve earlier evidence')
    base.write_json(path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        success=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        no_performance_metrics=True, evidence=EVIDENCE,
        source_sha256={str(Path(p).resolve()):base.digest(p) for p in
            [__file__, core.__file__, experiment.__file__, Path(__file__).with_name('test_nested_anchor_v2.py')]})))
    raise SystemExit(0 if result.wasSuccessful() else 1)
