"""Independent operator, gradient, permutation, and target-fact boundary checks."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest

import numpy as np
import torch

from improve import triangular_pair_graph_v1 as pair
from improve.test_nested_anchor_v2 import fixture


def explicit_triangle(q, k, v1, v2, kind):
    rows = []
    for i in range(q.shape[0]):
        columns = []
        for j in range(k.shape[1]):
            heads = []
            for h in range(q.shape[2]):
                logits = torch.stack([(q[i, t, h] * k[t, j, h]).sum()
                                      for t in range(k.shape[0])]) / math.sqrt(q.shape[-1])
                weights = logits.exp() / logits.exp().sum()
                values = [v1[i, t, h] * v2[t, j, h] if kind == 'full'
                          else .5 * (v1[i, t, h] + v2[t, j, h]) for t in range(k.shape[0])]
                heads.append(sum(weights[t] * v for t, v in enumerate(values)))
            columns.append(torch.stack(heads))
        rows.append(torch.stack(columns))
    return torch.stack(rows)


class TriangularPairTests(unittest.TestCase):
    def test_operator_and_all_four_gradients_against_explicit_definition(self):
        torch.manual_seed(121)
        shape = (4, 4, 2, 3)
        tensors = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for _ in range(4)]
        weight = torch.randn(shape, dtype=torch.float64)
        for kind in pair.KINDS:
            expected = explicit_triangle(*tensors, kind)
            actual = pair.triangle_block(*tensors, kind)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            expected_grad = torch.autograd.grad((expected * weight).sum(), tensors)
            actual_grad = torch.autograd.grad((actual * weight).sum(), tensors)
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)

    def test_normalization_is_over_intermediate_and_preserves_all_nodes(self):
        n = 5
        q = torch.zeros(n, n, 1, 1, dtype=torch.float64)
        k = torch.zeros_like(q)
        v1 = torch.arange(1, n + 1, dtype=torch.float64)[None, :, None, None].expand_as(q)
        v2 = torch.arange(1, n + 1, dtype=torch.float64)[:, None, None, None].expand_as(q)
        actual = pair.triangle_block(q, k, v1, v2, 'full')
        torch.testing.assert_close(actual, torch.full_like(actual, 11.))
        # The last intermediate contributes independently to every (i,j).
        changed = v2.clone(); changed[-1] += 10
        difference = pair.triangle_block(q, k, v1, changed, 'full') - actual
        torch.testing.assert_close(difference, torch.full_like(actual, 10.))

    def test_chunk_checkpoint_values_gradients_and_entity_permutation(self):
        torch.manual_seed(812)
        for kind in pair.KINDS:
            chunked = pair.TriangleLayer(8, 2, 2, 0., kind).double().train()
            dense = deepcopy(chunked); dense.row_chunk = 20
            state = torch.randn(7, 7, 8, dtype=torch.float64, requires_grad=True)
            other = state.detach().clone().requires_grad_()
            actual, expected = chunked(state), dense(other)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            actual.square().sum().backward(); expected.square().sum().backward()
            torch.testing.assert_close(state.grad, other.grad, atol=2e-11, rtol=2e-11)
            for a, b in zip(chunked.parameters(), dense.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=2e-11, rtol=2e-11)
            permutation = torch.tensor([2, 5, 0, 6, 1, 3, 4])
            chunked.eval()
            moved = chunked(state.detach()[permutation][:, permutation])
            torch.testing.assert_close(moved, actual.detach()[permutation][:, permutation],
                                       atol=2e-12, rtol=2e-12)

    def test_query_facts_and_query_labels_cannot_enter_pair_tokens(self):
        a, _ = fixture()
        omitted = np.flatnonzero(a['folds'] == 0)
        graph = pair.context(a, omitted, 'cpu')
        changed = {k: v.copy() for k, v in a.items()}
        changed['y_train'][omitted] = 1 - changed['y_train'][omitted]
        changed['y_val'] = np.full(100, 777)
        changed['anchor_oof'] = np.full(len(a['y_train']), 555.)
        other = pair.context(changed, omitted, 'cpu')
        for name in ('physical_relation', 'observed_zero', 'diagonal', 'global_to_local'):
            torch.testing.assert_close(getattr(graph, name), getattr(other, name), atol=0, rtol=0)
        query = torch.as_tensor(a['pairs_train'][:, omitted] + a['pair_offset'][:, None])
        u, v = graph.global_to_local[query]
        self.assertFalse(graph.physical_relation[u, v].any())
        self.assertFalse(graph.physical_relation[v, u].any())
        self.assertFalse(graph.observed_zero[u, v].any())
        self.assertFalse(graph.observed_zero[v, u].any())
        self.assertFalse(np.isin(graph.support_zero_indices, omitted).any())

    def test_zero_null_control_parameters_nonzero_gradients_and_query_replay(self):
        torch.manual_seed(471)
        a, _ = fixture()
        parameters = pair.parameters(); parameters.update(width=8, depth=2, row_chunk=3, dropout=0.)
        model, xs, graphs, folds = pair.setup(a, parameters, 'full', 'cpu')
        control = deepcopy(model); control.kind = 'additive_value'
        for layer in control.layers:
            layer.kind = control.kind
        self.assertEqual(set(model.state_dict()), set(control.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in control.parameters()))
        queries = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])[:, folds[0]]
        model.eval(); control.eval()
        null = model(xs, graphs[0], queries)
        torch.testing.assert_close(null, torch.zeros_like(null), atol=0, rtol=0)
        (-null.sum()).backward()
        self.assertGreater(model.readout[-1].weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        torch.nn.init.normal_(model.readout[-1].weight, std=.2)
        control.load_state_dict(model.state_dict())
        value = model(xs, graphs[0], queries)
        self.assertGreater((value - control(xs, graphs[0], queries)).abs().max().item(), 1e-5)
        value.square().sum().backward()
        for projection in model.layers[0].projections:
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        reverse = model(xs, graphs[0], queries.flip(1)).flip(0)
        chunks = torch.cat([model(xs, graphs[0], queries[:, i:i + 3])
                            for i in range(0, queries.shape[1], 3)])
        torch.testing.assert_close(value, reverse, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(value, chunks, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(model.relation.weight[0], torch.zeros(8), atol=0, rtol=0)

    def test_complete_model_is_equivariant_under_within_type_reindexing(self):
        torch.manual_seed(102)
        a, _ = fixture()
        parameters = pair.parameters(); parameters.update(width=8, depth=2, row_chunk=3, dropout=0.)
        model, xs, graphs, _ = pair.setup(a, parameters, 'full', 'cpu')
        torch.nn.init.normal_(model.readout[-1].weight, std=.2)
        model.eval()
        graph = graphs[0]
        orders = [torch.randperm(len(x)) for x in xs]
        permutation = torch.cat([orders[0], orders[1] + len(xs[0])])
        inverse = torch.argsort(permutation)
        mapping = graph.global_to_local.clone()
        present = mapping >= 0
        mapping[present] = inverse[mapping[present]]
        permuted_graph = replace(graph,
            physical_relation=graph.physical_relation[permutation][:, permutation],
            observed_zero=graph.observed_zero[permutation][:, permutation],
            diagonal=graph.diagonal[permutation][:, permutation], global_to_local=mapping)
        queries = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
        expected = model(xs, graph, queries)
        actual = model([x[p] for x, p in zip(xs, orders)], permuted_graph, queries)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-7)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
