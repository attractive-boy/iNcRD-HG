"""Dense determinant oracle, pinned author greedy rule and graph/label boundaries."""
import ast
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from improve import determinantal_selection_graph_v1 as core
from improve import anchor_determinantal_selection_experiment_v1 as experiment
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent / 'research/diversity_next_20260916'
EVIDENCE = {}


def dense_kernel(feature, quality, ridge, diagonal=False):
    x, q = np.asarray(feature), np.asarray(quality)
    norms = (np.square(x).sum(1) > 1e-20).astype(float)
    gram = x @ x.T if not diagonal else np.zeros((len(x), len(x)))
    np.fill_diagonal(gram, norms + ridge)
    return q[:, None] * gram * q[None, :]


def dense_greedy(kernel, priority, budget):
    chosen, logdet = [], 0.
    while len(chosen) < min(budget, len(kernel)):
        candidates = [i for i in range(len(kernel)) if i not in chosen]
        values = []
        for i in candidates:
            subset = chosen + [i]
            sign, value = np.linalg.slogdet(kernel[np.ix_(subset, subset)])
            assert sign == 1
            values.append(value)
        maximum = max(values)
        tied = [i for i, value in zip(candidates, values) if value == maximum]
        pick = min(tied, key=lambda i: (priority[i], i))
        chosen.append(pick)
        logdet = maximum
    return chosen, logdet


def example():
    rng = np.random.default_rng(98)
    a = dict(node_counts=np.array([2, 3, 2, 1]), pair_offset=np.array([2, 5]), nd=np.array(2),
        pairs_train=np.array([[0, 1, 2, 0, 1, 2], [0, 0, 1, 1, 1, 0]]),
        y_train=np.array([1., 0., 1., 0., 1., 0.], dtype=np.float32),
        folds=np.array([0, 1, 2, 3, 4, 0]),
        edges=np.array([[2, 4, 3, 0, 1, 0, 1, 0, 2, 3], [5, 6, 6, 2, 2, 5, 5, 1, 7, 4]]),
        relation=np.array([0, 0, 0, 1, 1, 2, 2, 1, 2, 2]),
        target_codes=np.array([0, 5, 3, -1, -1, -1, -1, -1, -1, -1]))
    for i, n in enumerate(a['node_counts']):
        a['x_' + str(i)] = rng.normal(size=(n, 4 + i)).astype(np.float32)
    return a


class DeterminantalChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(217)

    def test_ragged_greedy_matches_independent_dense_log_determinants(self):
        rng = np.random.default_rng(6659)
        lengths = [0, 1, 8, 5, 11]
        raw = torch.tensor(rng.normal(size=(sum(lengths), 12)), dtype=torch.float64)
        raw[1] = 0.
        feature = core.normalize_rows(raw)
        groups = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
        priority = torch.tensor(rng.permutation(len(raw)))
        error = 0.
        for diagonal, uniform in ((False, False), (False, True), (True, False)):
            quality = torch.ones(len(raw), dtype=torch.float64) if uniform else torch.tensor(rng.uniform(.1, 1., len(raw)))
            actual = core.greedy_subset(feature, quality, groups, priority, len(lengths), 3, 1e-5, diagonal=diagonal)
            start = 0
            for number, length in enumerate(lengths):
                kernel = dense_kernel(feature[start:start+length], quality[start:start+length], 1e-5, diagonal)
                selected, logdet = dense_greedy(kernel, priority[start:start+length].numpy(), 3)
                got = actual['indices'][number]
                got = (got[got >= 0] - start).tolist()
                self.assertEqual(got, selected)
                self.assertAlmostEqual(float(actual['logdet'][number]), logdet, places=10)
                error = max(error, abs(float(actual['logdet'][number]) - logdet))
                start += length
        EVIDENCE['dense_oracle'] = dict(ragged_graphs=len(lengths), arms=3, maximum_logdet_error=error,
            empty_and_zero_vectors_included=True, exact_greedy_indices=True)

    def test_pinned_author_cholesky_selection(self):
        path = HERE / 'cdpruner_author_v1_llava__model__llava_arch.py'
        self.assertEqual(base.digest(path), '1e011ec547bb20dd28cb8a9adb8a4987a2ad8b476e7ce51d5bd3e0bba71c848f')
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LlavaMetaForCausalLM')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'encode_images')
        namespace = dict(torch=torch)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        for seed in range(8):
            generator = torch.Generator().manual_seed(seed + 91)
            raw = torch.randn(1, 9, 12, generator=generator)
            query = torch.randn(1, 12, generator=generator)
            data = SimpleNamespace(mm_projector=torch.nn.Identity(),
                get_vision_tower=lambda: lambda images, texts=None: (raw.clone(), raw.clone(), -query.clone()))
            owner = SimpleNamespace(get_model=lambda: data, visual_token_num=3)
            _, mask = namespace['encode_images'](owner, None)
            # The published code negates cosine. Supply -query above and reproduce
            # its single-image relevance arithmetic; the local graph adapter uses
            # positive cosine directly and normalizes each query separately.
            source_feature = raw[0] / raw[0].norm(dim=-1, keepdim=True)
            source_query = query[0] / query[0].norm()
            relevance = source_feature @ source_query
            quality = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min())
            normalized = core.normalize_rows(raw[0].double())
            actual = core.greedy_subset(normalized, quality.double(), torch.zeros(9, dtype=torch.long),
                torch.arange(9), 1, 3, 0.)
            self.assertEqual(sorted(actual['indices'][0].tolist()), torch.nonzero(mask[0], as_tuple=True)[0].tolist())
        EVIDENCE['author_comparison'] = dict(cases=8, exact_selected_subsets=True,
            original_method_ast_executed=True, original_relevance_sign_accounted_for=True,
            compared_only_nondegenerate_single_image_cases=True)

    def test_diversity_quality_and_degenerate_inputs(self):
        feature = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)
        quality = torch.tensor([1., .99, .85], dtype=torch.float64)
        groups, priority = torch.zeros(3, dtype=torch.long), torch.arange(3)
        full = core.greedy_subset(feature, quality, groups, priority, 1, 2, 1e-5)
        independent = core.greedy_subset(feature, quality, groups, priority, 1, 2, 1e-5, diagonal=True)
        self.assertEqual(full['indices'].tolist(), [[0, 2]])
        self.assertEqual(independent['indices'].tolist(), [[0, 1]])
        kernel = dense_kernel(feature, quality, 1e-5)
        self.assertGreater(np.linalg.det(kernel[np.ix_([0, 2], [0, 2])]),
                           np.linalg.det(kernel[np.ix_([0, 1], [0, 1])]))
        constant = core.conditional_quality(feature, torch.zeros(1, 2, dtype=torch.float64), groups, 1, 1e-4)
        torch.testing.assert_close(constant, torch.ones_like(constant), atol=0, rtol=0)
        empty = core.greedy_subset(torch.empty(0, 2, dtype=torch.float64), torch.empty(0, dtype=torch.float64),
            torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), 3, 5, 1e-5)
        self.assertTrue(torch.all(empty['indices'] == -1))
        self.assertTrue(torch.all(empty['logdet'] == 0))
        with self.assertRaises(ValueError):
            core.greedy_subset(feature, torch.zeros_like(quality), groups, priority, 1, 2, 1e-5)
        with self.assertRaises(FloatingPointError):
            core.normalize_rows(torch.tensor([[float('nan'), 1.]]))
        EVIDENCE['mechanism_counterexample'] = dict(full=[0, 2], no_diversity=[0, 1],
            repeated_neighbor_avoided=True, constant_quality_and_empty_groups_supported=True)

    def test_permutation_batching_and_piecewise_gather_gradient(self):
        raw = torch.randn(9, 7, dtype=torch.float64, requires_grad=True)
        query = core.normalize_rows(torch.randn(2, 7, dtype=torch.float64))
        group = torch.tensor([0] * 5 + [1] * 4)
        priority = torch.tensor([5, 3, 8, 4, 1, 2, 7, 0, 6])
        def evaluate(value):
            feature = core.normalize_rows(value)
            quality = core.conditional_quality(feature, query, group, 2, 1e-4)
            selected = core.greedy_subset(feature, quality, group, priority, 2, 2, 1e-5)['indices']
            return value[selected].tanh().sum(1)
        self.assertTrue(torch.autograd.gradcheck(evaluate, (raw,), eps=1e-6, atol=1e-7, rtol=1e-5))
        f = core.normalize_rows(raw.detach())
        q = core.conditional_quality(f, query, group, 2, 1e-4)
        normal = core.greedy_subset(f, q, group, priority, 2, 2, 1e-5)
        perm = torch.tensor([7, 1, 6, 0, 5, 4, 2, 8, 3])
        mixed = core.greedy_subset(f[perm], q[perm], group[perm], priority[perm], 2, 2, 1e-5)
        torch.testing.assert_close(perm[mixed['indices']], normal['indices'], atol=0, rtol=0)
        for i in (0, 1):
            ids = torch.nonzero(group == i, as_tuple=True)[0]
            part_quality = core.conditional_quality(f[ids], query[i:i+1], torch.zeros(len(ids), dtype=torch.long), 1, 1e-4)
            torch.testing.assert_close(part_quality, q[ids], atol=0, rtol=0)
            part = core.greedy_subset(f[ids], part_quality, torch.zeros(len(ids), dtype=torch.long),
                priority[ids], 1, 2, 1e-5)
            torch.testing.assert_close(ids[part['indices'][0]], normal['indices'][i], atol=0, rtol=0)
        EVIDENCE['permutation_and_gradients'] = dict(interleaved_groups=True, query_batching=True,
            unchanged_selection_region_finite_differences=True)

    def test_graph_query_labels_and_all_candidates(self):
        a, p = example(), dict(core.parameters(), width=8, depth=2, dropout=0., retained_neighbors=2)
        model, xs, graphs, folds = core.setup(a, p, 'full', 'cpu')
        query = a['pairs_train'] + a['pair_offset'][:, None]
        for f, graph in enumerate(graphs):
            ids = folds[f].numpy()
            for u, v in query[:, ids].T:
                self.assertEqual(graph.adjacency[u, v], 0.)
                self.assertEqual(graph.adjacency[v, u], 0.)
                nodes, membership = graph.extract(u, v)
                expected = (set(graph.adjacency[u].indices) | set(graph.adjacency[v].indices)) - {int(u), int(v)}
                self.assertEqual(set(nodes), expected)
                self.assertTrue(np.all(membership.sum(1) > 0))
            changed = {k: v.copy() for k, v in a.items()}
            changed['y_train'][ids] = 1. - changed['y_train'][ids]
            _, _, other, _ = core.setup(changed, p, 'full', 'cpu')
            self.assertEqual((graph.adjacency != other[f].adjacency).nnz, 0)
            for value, expected in zip(graph.neural_graph, other[f].neural_graph):
                torch.testing.assert_close(value, expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            core.setup(dict(a, y_val=np.ones(1)), p, 'full', 'cpu')
        with self.assertRaisesRegex(ValueError, 'Query fact'):
            graphs[1].extract(2, 5)
        a['pair_offset'] = np.array([0, 5])
        identity, ix, ig, _ = core.setup(a, p, 'full', 'cpu')
        self.assertFalse(list(identity.parameters()))
        self.assertFalse(ix)
        self.assertTrue(all(x is None for x in ig))
        torch.testing.assert_close(identity([], None, torch.as_tensor(query)), torch.zeros(query.shape[1]))
        EVIDENCE['boundaries'] = dict(folds=5, entire_union_retained_before_selection=True,
            query_reverse_and_labels_isolated=True, training_only_entry=True, lnc_identity=True)

    def test_same_initialization_nonzero_output_and_query_replay(self):
        a, p = example(), dict(core.parameters(), width=8, depth=2, dropout=.1, retained_neighbors=2)
        initial, rng = None, None
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(332)
            model, xs, graphs, folds = core.setup(a, p, kind, 'cpu')
            if initial is None:
                initial, rng = experiment.cpu_copy(model.state_dict()), torch.random.get_rng_state().clone()
            else:
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
                torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
            model.eval()
            with torch.no_grad():
                model.readout[-1].weight.normal_(std=.2)
            query = torch.as_tensor(a['pairs_train'][:, folds[0].numpy()] + a['pair_offset'][:, None])
            value = model(xs, graphs[0], query)
            reverse = model(xs, graphs[0], query.flip(1)).flip(0)
            chunks = torch.cat([model(xs, graphs[0], query[:, i:i+1]) for i in range(query.shape[1])])
            torch.testing.assert_close(value, reverse, atol=2e-7, rtol=2e-6)
            torch.testing.assert_close(value, chunks, atol=2e-7, rtol=2e-6)
            value.square().sum().backward()
            self.assertGreater(sum(float(v.grad.abs().sum()) for v in model.evidence.parameters()), 0.)
            self.assertEqual(model.selection_diagnostics['selected'], query.shape[1] * 2 * 3)
        EVIDENCE['initialization'] = dict(identical_parameters_and_rng=True, nonzero_evidence_gradients=True,
            nonzero_output_query_replay=True)

    def test_training_budget_and_optimizer_state(self):
        a = example()
        a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
        p = dict(smoke=True, family='determinantal_selection', implementation_module='determinantal_selection_graph_v1',
            source_sha256={'determinantal_selection_graph_v1.py': base.digest(core.__file__)},
            core_parameters=dict(core.parameters(), width=8, depth=2, dropout=.1, retained_neighbors=2),
            kinds=list(core.KINDS), epochs=6, check_interval=3, weight_decay=.001, strong_weight=.1)
        rngs = []
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, p, kind, .003, 77, 'cpu', fixed_epochs=6)
            self.assertIsNone(value)
            self.assertEqual(detail['gradient_evaluations'], 6)
            self.assertEqual(detail['optimizer_steps'], 6)
            self.assertEqual(detail['selection_diagnostics']['calls'], 6)
            self.assertTrue(all('anchor_composed' not in row for row in trace))
            final = model.training_final_state
            self.assertTrue(all(int(state['step']) == 6 for state in final['optimizer']['state'].values()))
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, final['model'][name], atol=0, rtol=0)
            rngs.append(final['cpu_rng'])
        for value in rngs[1:]:
            torch.testing.assert_close(value, rngs[0], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.fit(dict(a, y_val=np.zeros(1)), p, 'full', .003, 77, 'cpu', fixed_epochs=6)
        EVIDENCE['budget'] = dict(synthetic_fits=3, steps_each=6, gradients_each=6,
            optimizer_states_checked=True, final_rng_equal=True)


if __name__ == '__main__':
    output = HERE / 'DETERMINANTAL_UNIT_V2.json'
    if output.exists():
        raise ValueError('Preserve previous unit evidence')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DeterminantalChecks))
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__)]
    base.write_json(output, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        success=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        evidence=EVIDENCE, source_sha256={str(path): base.digest(path) for path in sources},
        evaluation_labels_read=False, formal_fits=0)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
