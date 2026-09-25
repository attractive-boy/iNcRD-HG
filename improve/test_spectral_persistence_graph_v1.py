"""Independent spectral-event oracle, pinned author comparison and query isolation."""
from datetime import datetime, timezone
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import torch

from improve import anchor_spectral_persistence_experiment_v1 as experiment
from improve import spectral_persistence_graph_v1 as core
from improve import tabular_foundation as base
from improve.spectral_persistence_build_v1 import load_extension

HERE = Path(__file__).resolve().parent / 'research/persistence_next_20260916'
EVIDENCE = {}


def oracle(fv, fe, edges):
    """Rebuild each threshold graph; no union-find or native pairing indices."""
    fv, fe, edges = np.asarray(fv), np.asarray(fe), np.asarray(edges).reshape(-1, 2)
    n = len(fv)
    gamma = np.ones(n)
    for i, value in enumerate(fe):
        for v in edges[i]:
            gamma[v] = min(gamma[v], value)
    alpha = [(fv[v], -gamma[v], v) for v in range(n)]
    deaths = np.ones(n)
    r0 = np.zeros((n, n))
    alive = set(range(n))
    previous = np.arange(n)
    matrix = np.zeros((n, n))
    one = []

    def spectrum(nodes):
        sub = matrix[np.ix_(nodes, nodes)]
        eigen = np.linalg.eigvalsh(np.diag(sub.sum(1)) - sub)
        eigen[np.abs(eigen) <= 1e-10] = 0.
        return np.pad(eigen, (0, n - len(eigen)))

    for level in np.unique(fe):
        added = edges[fe == level]
        matrix[added[:, 0], added[:, 1]] = matrix[added[:, 1], added[:, 0]] = 1.
        count, components = connected_components(sparse.csr_matrix(matrix), directed=False)
        for component in range(count):
            nodes = np.flatnonzero(components == component)
            old = sorted(alive.intersection(nodes), key=lambda node: alpha[node])
            killed = old[1:]
            eigen = spectrum(nodes)
            for v in killed:
                deaths[v], r0[v] = level, eigen
                alive.remove(v)
            new_edges = sum(components[u] == component for u, _ in added)
            merges = len(np.unique(previous[nodes])) - 1
            for _ in range(new_edges - merges):
                one.append(np.r_[1., level, 0., 0., eigen])
        previous = components
    for v in alive:
        r0[v] = spectrum(np.flatnonzero(previous == previous[v]))
    zero = np.c_[np.zeros(n), deaths, fv, gamma, r0]
    return zero, np.asarray(one).reshape(-1, n + 4), spectrum(np.arange(n)), len(alive)


def packed(graphs):
    vertices, values, edges, vs, es = [], [], [], [0], [0]
    for fv, fe, edge in graphs:
        vertices.extend(fv)
        values.extend(fe)
        edges.extend((np.asarray(edge).reshape(-1, 2) + vs[-1]).tolist())
        vs.append(vs[-1] + len(fv))
        es.append(es[-1] + len(fe))
    return (torch.tensor(vertices, dtype=torch.float64), torch.tensor(values, dtype=torch.float64),
            torch.tensor(edges, dtype=torch.int64).reshape(-1, 2),
            torch.tensor(vs), torch.tensor(es))


def ordered(rows):
    return np.asarray(sorted(np.asarray(rows).tolist())).reshape(np.asarray(rows).shape)


def author():
    folder = HERE / 'author_build_v1'
    assert base.digest(folder / 'spectre_cpu.cpp') == '874ee43b369bdb3013e36753d0136257d2517ea2c57413f73028c3c6f3e6a1f6'
    assert base.digest(folder / 'unionfind.hh') == '43c693f87c5f270013a3c70e1a3c28c98e15b2eb9e9169748b352ca10331727c'
    return load_extension('reviewed_spectre_author_v1', folder / 'spectre_cpu.cpp', folder,
                          headers=[folder / 'unionfind.hh'], optimization='-O0')


def author_diagrams(args):
    fv, fe, edges, vs, es = args
    zero, one = author().compute_spectre_batched_mt(fv[None], fe[None], edges, vs, es)
    zero, one = zero[0], one[0]
    t0 = torch.cat((torch.zeros_like(zero[:, :1]), torch.nan_to_num(zero[:, [0, 2, 1]], nan=1.)), 1)
    t1 = torch.cat((torch.ones_like(one[:, :1]), one[:, :3]), 1)
    return t0, t1, zero[:, 3:], one[:, 3:], one[:, 0] > 0.


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


class PersistenceChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(293)

    def test_exact_matrix_cache_reuses_only_identical_spectra(self):
        graph = ([.1, .2, .3, .4, .5], [.1, .2, .3, .4, .5],
                 [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)])
        args = packed([graph] * 12 + [([.4], [], [])])
        cached = core.native().event_spectra(*args, 4096)
        uncached = core.native().event_spectra(*args, 0)
        for value, expected in zip(cached[:6], uncached[:6]):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        torch.testing.assert_close(cached[6][:, :2], uncached[6][:, :2], atol=0, rtol=0)
        self.assertLess(int(cached[6][:, 2].sum()), int(uncached[6][:, 2].sum()))
        EVIDENCE['exact_cache'] = dict(output_bitwise_equal=True,
            cached_eigendecompositions=int(cached[6][:, 2].sum()),
            uncached_eigendecompositions=int(uncached[6][:, 2].sum()),
            scope='per native call and worker', maximum_entries_per_worker=4096)

    def test_independent_threshold_component_oracle(self):
        rng = np.random.default_rng(9361)
        graphs = []
        for i in range(24):
            n = 1 + i % 10
            edges = np.array([(u, v) for u in range(n) for v in range(u + 1, n)
                              if rng.random() < .42], dtype=np.int64).reshape(-1, 2)
            fv = rng.uniform(.05, .95, size=n)
            fe = rng.uniform(.05, .95, size=len(edges)) if i % 2 else rng.integers(1, 5, len(edges)) / 5.
            graphs.append((fv, fe, edges))
        args = packed(graphs)
        result = core.diagrams(*args)
        width = result['rho0'].shape[1]
        vb, eb, worst = 0, 0, 0.
        for fv, fe, edges in graphs:
            n, m = len(fv), len(fe)
            want0, want1, final, components = oracle(fv, fe, edges)
            got0 = torch.cat((result['tuple0'][vb:vb + n], result['rho0'][vb:vb + n]), 1).numpy()
            mask = result['cycle_mask'][eb:eb + m]
            got1 = torch.cat((result['tuple1'][eb:eb + m][mask], result['rho1'][eb:eb + m][mask]), 1).numpy()
            want0, want1 = [np.pad(value, ((0, 0), (0, width - n))) for value in (want0, want1)]
            np.testing.assert_allclose(got0, want0, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(ordered(got1), ordered(want1), atol=2e-12, rtol=2e-12)
            index = list(args[3]).index(torch.tensor(vb))
            np.testing.assert_allclose(result['final_rho'][index], np.pad(final, (0, width - n)), atol=2e-12)
            self.assertEqual(int(result['counts'][index, 0]), components)
            self.assertEqual(int(mask.sum()), m - n + components)
            worst = max(worst, float(np.max(abs(got0 - want0))))
            vb, eb = vb + n, eb + m
        EVIDENCE['threshold_oracle'] = dict(graphs=len(graphs), maximum_error=worst,
                                           includes_isolated_vertices_and_empty_edge_sets=True)

    def test_pinned_author_unique_filtrations_and_gradients(self):
        graphs = [([.15, .8, .35, .6], [.12, .37, .55, .72], [(0, 1), (1, 2), (0, 2), (2, 3)]),
                  ([.2, .3, .7], [.16, .82], [(0, 1), (1, 2)])]
        args = list(packed(graphs))
        args[0].requires_grad_(True)
        args[1].requires_grad_(True)
        got, want = core.diagrams(*args), author_diagrams(args)
        torch.testing.assert_close(got['tuple0'], want[0], atol=0, rtol=0)
        torch.testing.assert_close(got['cycle_mask'], want[4], atol=0, rtol=0)
        torch.testing.assert_close(got['tuple1'][want[4]], want[1][want[4]], atol=0, rtol=0)
        width = got['rho0'].shape[1]
        err = 0.
        for value, expected in zip((got['rho0'], got['rho1']), want[2:4]):
            torch.testing.assert_close(value, expected[:, :width], atol=3e-6, rtol=3e-6)
            err = max(err, float((value - expected[:, :width]).abs().max()))
        w0, w1 = torch.randn_like(got['tuple0']), torch.randn_like(got['tuple1'])
        gradients = []
        for zero, one in ((got['tuple0'], got['tuple1']), want[:2]):
            loss = (zero * w0).sum() + (one[want[4]] * w1[want[4]]).sum()
            gradients.append(torch.autograd.grad(loss, args[:2], retain_graph=True))
        for value, expected in zip(*gradients):
            torch.testing.assert_close(value, expected, atol=1e-14, rtol=1e-14)
        EVIDENCE['author_unique_values'] = dict(graphs=2, spectrum_maximum_error=err,
            tuple_values_equal=True, gather_gradients_equal=True,
            author_spectrum_dtype='float32', local_spectrum_dtype='float64')

    def test_simultaneous_disconnected_components_counterexample(self):
        graph = ([.1, .3, .5, .2, .4], [.2, .2, .2], [(0, 1), (1, 2), (3, 4)])
        args = packed([graph])
        got, want = core.diagrams(*args), author_diagrams(args)
        zero, _, _, _ = oracle(*graph)
        np.testing.assert_allclose(got['rho0'], zero[:, 4:], atol=2e-12)
        self.assertGreater(float((got['rho0'] - want[2][:, :5]).abs().max()), .9)
        np.testing.assert_allclose(got['rho0'][1], [0., 1., 3., 0., 0.], atol=2e-12)
        np.testing.assert_allclose(want[2][1, :5], [0., 2., 0., 0., 0.], atol=3e-6)
        EVIDENCE['same_value_counterexample'] = dict(vertices=list(graph[0]), edges=list(graph[2]),
            filtration=list(graph[1]), native_path_event=got['rho0'][1].tolist(),
            author_path_event=want[2][1, :5].tolist(), independent_definition_passed=True)

    def test_relabel_edge_order_packing_and_finite_differences(self):
        graph = (np.array([.1, .3, .5, .2, .4, .8]), np.array([.2, .2, .2, .4, .6, .6]),
                 np.array([(0, 1), (1, 2), (3, 4), (0, 2), (2, 5), (4, 5)]))
        fv, fe, edges = graph
        permutation, edge_order = np.array([2, 4, 1, 0, 5, 3]), np.array([3, 1, 5, 2, 0, 4])
        inverse = np.argsort(permutation)
        permuted = (fv[permutation], fe[edge_order], inverse[edges[edge_order]])
        baseline = core.diagrams(*packed([graph]))
        other = core.diagrams(*packed([permuted]))
        for dim in (0, 1):
            normal = torch.cat((baseline['tuple' + str(dim)], baseline['rho' + str(dim)]), 1)
            changed = torch.cat((other['tuple' + str(dim)], other['rho' + str(dim)]), 1)
            if dim == 1:
                normal, changed = normal[baseline['cycle_mask']], changed[other['cycle_mask']]
            np.testing.assert_allclose(ordered(normal), ordered(changed), atol=2e-12)
        bigger = (np.linspace(.1, .9, 9), [.2], [(0, 1)])
        batch = core.diagrams(*packed([graph, bigger]))
        for dim, count in ((0, len(fv)), (1, len(fe))):
            torch.testing.assert_close(batch['rho' + str(dim)][:count, :6], baseline['rho' + str(dim)], atol=0, rtol=0)
        small = core.diagrams(*packed([([.4], [], [])]))
        self.assertEqual(small['rho0'].shape[1], 1)
        torch.testing.assert_close(small['tuple0'], torch.tensor([[0., 1., .4, 1.]], dtype=torch.float64))
        # Sorting is unchanged in this perturbation region; spectra have zero local derivative.
        args = list(packed([(fv, np.linspace(.1, .8, len(fe)), edges)]))
        args[0].requires_grad_(True)
        args[1].requires_grad_(True)
        def function(v, e):
            data = core.diagrams(v, e, *args[2:])
            return torch.cat((data['tuple0'].flatten(), data['tuple1'][data['cycle_mask']].flatten(),
                              data['rho0'].flatten(), data['rho1'].flatten()))
        self.assertTrue(torch.autograd.gradcheck(function, tuple(args[:2]), eps=1e-6, atol=1e-7, rtol=1e-6))
        EVIDENCE['invariance_and_derivatives'] = dict(node_relabel=True, edge_reorder=True,
            batch_padding_independent=True, isolated_vertex=True, finite_differences=True)

    def test_query_reverse_fact_and_label_isolation(self):
        a, p = example(), dict(core.parameters(), width=8, depth=2, dropout=0.)
        model, xs, graphs, folds = core.setup(a, p, 'full', 'cpu')
        query = a['pairs_train'] + a['pair_offset'][:, None]
        for f, graph in enumerate(graphs):
            ids = folds[f].numpy()
            for u, v in query[:, ids].T:
                self.assertEqual(graph.adjacency[u, v], 0.)
                self.assertEqual(graph.adjacency[v, u], 0.)
                nodes, edges = graph.extract(u, v)
                expected = set(graph.adjacency[u].indices).intersection(graph.adjacency[v].indices)
                self.assertEqual(set(nodes), expected | {int(u), int(v)})
                reverse_nodes, reverse_edges = graph.extract(v, u)
                self.assertEqual({tuple(sorted(edge)) for edge in nodes[edges]},
                                 {tuple(sorted(edge)) for edge in reverse_nodes[reverse_edges]})
            changed = {k: v.copy() for k, v in a.items()}
            changed['y_train'][ids] = 1. - changed['y_train'][ids]
            _, _, altered, _ = core.setup(changed, p, 'full', 'cpu')
            self.assertEqual((graph.adjacency != altered[f].adjacency).nnz, 0)
            for value, expected in zip(graph.neural_graph, altered[f].neural_graph):
                torch.testing.assert_close(value, expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            core.setup(dict(a, y_val=np.zeros(1)), p, 'full', 'cpu')
        with self.assertRaisesRegex(ValueError, 'Query fact'):
            graphs[1].extract(2, 5)
        a['pair_offset'] = np.array([0, 5])
        identity, ix, ig, _ = core.setup(a, p, 'full', 'cpu')
        self.assertFalse(list(identity.parameters()))
        self.assertFalse(ix)
        self.assertTrue(all(g is None for g in ig))
        torch.testing.assert_close(identity([], None, torch.as_tensor(query)), torch.zeros(query.shape[1]))
        EVIDENCE['boundaries'] = dict(folds=5, query_and_reverse_absent=True,
            all_common_neighbors_retained=True, query_label_invariance=True, lnc_identity=True)

    def test_three_arms_initialization_random_stream_and_nonzero_gradients(self):
        a, cp = example(), dict(core.parameters(), width=8, depth=2, dropout=.1)
        states, rngs = [], []
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(817)
            model, xs, graphs, folds = core.setup(a, cp, kind, 'cpu')
            states.append(experiment.cpu_copy(model.state_dict()))
            rngs.append(torch.random.get_rng_state().clone())
        for state, rng in zip(states[1:], rngs[1:]):
            self.assertEqual(set(state), set(states[0]))
            for key in state:
                torch.testing.assert_close(state[key], states[0][key], atol=0, rtol=0)
            torch.testing.assert_close(rng, rngs[0], atol=0, rtol=0)
        query = torch.as_tensor(a['pairs_train'][:, folds[0].numpy()] + a['pair_offset'][:, None])
        with torch.no_grad():
            model.readout[-1].weight.normal_(std=.2)
        initial = experiment.cpu_copy(model.state_dict())
        summaries = {}
        for kind in core.KINDS:
            model, xs, graphs, _ = core.setup(a, cp, kind, 'cpu')
            model.load_state_dict(initial)
            model.eval()
            value = model(xs, graphs[0], query)
            reverse = model(xs, graphs[0], query.flip(1)).flip(0)
            single = torch.cat([model(xs, graphs[0], query[:, i:i + 1]) for i in range(query.shape[1])])
            torch.testing.assert_close(value, reverse, atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(value, single, atol=1e-7, rtol=1e-6)
            value.square().sum().backward()
            norm = sum(float(v.grad.abs().sum()) for v in model.spectrum.parameters())
            if kind == 'no_spectrum':
                self.assertEqual(norm, 0.)
            else:
                self.assertGreater(norm, 0.)
            summaries[kind] = dict(spectrum_gradient_l1=norm)
        EVIDENCE['arms'] = dict(same_initial_state=True, same_initial_rng=True,
            nonzero_head_query_replay=True, gradients=summaries)

    def test_training_budget_and_training_only_entry(self):
        a = example()
        a['anchor_oof'] = np.linspace(-1., 1., len(a['y_train']))
        p = dict(smoke=True, family='spectral_persistence', implementation_module='spectral_persistence_graph_v1',
            source_sha256={'spectral_persistence_graph_v1.py': base.digest(core.__file__)},
            core_parameters=dict(core.parameters(), width=8, depth=2, dropout=.1), kinds=list(core.KINDS),
            epochs=6, check_interval=3, weight_decay=.001, strong_weight=.1)
        with self.assertRaisesRegex(ValueError, 'training arrays only'):
            experiment.fit(dict(a, y_val=np.zeros(1)), p, 'full', .003, 83, 'cpu', fixed_epochs=6)
        rngs = []
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, p, kind, .003, 83, 'cpu', fixed_epochs=6)
            self.assertIsNone(value)
            self.assertEqual(detail['gradient_evaluations'], 6)
            self.assertEqual(detail['optimizer_steps'], 6)
            self.assertTrue(all('anchor_composed' not in row for row in trace))
            final = model.training_final_state
            self.assertEqual(final['steps'], 6)
            self.assertTrue(all(int(state['step']) == 6 for state in final['optimizer']['state'].values()))
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, final['model'][name], atol=0, rtol=0)
            rngs.append(final['cpu_rng'])
        for state in rngs[1:]:
            torch.testing.assert_close(state, rngs[0], atol=0, rtol=0)
        EVIDENCE['training'] = dict(arms=3, steps_each=6, gradients_each=6,
            final_optimizer_verified=True, final_random_streams_equal=True, labels_from_training_only=True)


if __name__ == '__main__':
    output = HERE / 'PERSISTENCE_UNIT_V2.json'
    if output.exists():
        raise ValueError('Preserve previous unit evidence')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PersistenceChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = [Path(__file__), Path(core.__file__), Path(experiment.__file__),
               core.ROOT / 'spectral_persistence_native_v1.cpp', core.ROOT / 'spectral_persistence_build_v1.py']
    base.write_json(output, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        success=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        evidence=EVIDENCE, source_sha256={str(p): base.digest(p) for p in sources},
        evaluation_labels_read=False, formal_fits=0)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
