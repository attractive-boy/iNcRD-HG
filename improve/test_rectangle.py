"""Numerical, information-boundary, and model-function checks."""
import copy
import unittest

import torch
from torch import nn

from improve.rectangle_modules import (RectangleBranch, RectangleModel,
    evidence_grid, rectangle_message)


def brute(a, b, c, mode):
    nr, nd, _ = a.shape
    rows = []
    for r in range(nr):
        cols = []
        for d in range(nd):
            vals = []
            for s in range(nr):
                for e in range(nd):
                    if s != r and e != d:
                        vals.append(a[r, e] * b[s, e] * c[s, d] if mode == 'product'
                                    else (a[r, e] + b[s, e] + c[s, d]) / 3)
            cols.append(torch.stack(vals).mean(0) if vals else (a[r, d] + b[r, d] + c[r, d]) * 0)
        rows.append(torch.stack(cols))
    return torch.stack(rows)


def fixture():
    torch.manual_seed(8)
    z = dict(lncRNA=torch.randn(3, 6), miRNA=torch.randn(2, 6), drug=torch.randn(4, 6))
    edges = {('lncRNA', 'LncDrug', 'drug'): torch.tensor([[0, 1], [1, 2]]),
             ('miRNA', 'MiDrug', 'drug'): torch.tensor([[0, 1], [0, 3]])}
    support = torch.tensor([[0, 1, 2, 0], [1, 2, 2, 3]])
    labels = torch.tensor([1., 1., 0., 0.])
    query = torch.tensor([[0, 2], [0, 1]])
    return z, edges, support, labels, query


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(12, 1)

    def encoder(self, x, edges, weights):
        return x

    def decoder(self, z, pairs):
        return self.head(torch.cat((z['lncRNA'][pairs[0]], z['drug'][pairs[1]]), -1)).flatten()


class RectangleTests(unittest.TestCase):
    def test_explicit_rectangle_values_and_gradients(self):
        for shape in ((3, 4, 2), (4, 2, 3)):
            for mode in ('product', 'additive'):
                inputs = [torch.randn(shape, dtype=torch.double, requires_grad=True) for _ in range(3)]
                actual, expected = rectangle_message(*inputs, mode=mode), brute(*inputs, mode)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                ga = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
                ge = torch.autograd.grad(expected.square().sum(), inputs)
                for a, e in zip(ga, ge):
                    torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)

    def test_no_direct_self_factor(self):
        for mode in ('product', 'additive'):
            inputs = [torch.randn(3, 4, 2, dtype=torch.double, requires_grad=True) for _ in range(3)]
            out = rectangle_message(*inputs, mode=mode)[1, 2].sum()
            for grad in torch.autograd.grad(out, inputs):
                torch.testing.assert_close(grad[1, 2], torch.zeros(2, dtype=torch.double), atol=1e-12, rtol=0)

    def test_degenerate_field_and_invalid_shapes(self):
        for shape in ((1, 3, 2), (4, 1, 2)):
            v = torch.randn(shape, requires_grad=True)
            for mode in ('product', 'additive'):
                out = rectangle_message(v, v, v, mode)
                self.assertEqual(out.abs().sum().item(), 0)
                out.sum().backward()
        with self.assertRaises(ValueError):
            rectangle_message(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3))

    def test_query_positive_and_zero_are_absent(self):
        z, edges, support, labels, _ = fixture()
        query = support[:, [0, 2]]
        grid, offsets = evidence_grid(z, edges, ('lncRNA', 'miRNA'), 'lncRNA', support, labels, query)
        self.assertEqual(grid[query[0], query[1]].sum().item(), 0)
        self.assertEqual(grid[1, 2, 0].item(), 1)
        self.assertEqual(grid[0, 3, 1].item(), 1)
        self.assertEqual(grid[offsets['miRNA'], 0, 0].item(), 1)
        changed = labels.clone()
        changed[[0, 2]] = 1 - changed[[0, 2]]
        other, _ = evidence_grid(z, edges, ('lncRNA', 'miRNA'), 'lncRNA', support, changed, query)
        torch.testing.assert_close(grid, other)

    def test_initial_function_and_equal_parameters(self):
        z, edges, support, labels, query = fixture()
        base = FakeBase()
        model = RectangleModel(base, 6, 'lncRNA', width=4)
        got = model(z, edges, query, {}, support, labels)
        torch.testing.assert_close(got, base.decoder(z, query), rtol=0, atol=0)
        control = copy.deepcopy(model)
        control.mode = 'additive'
        for a, b in zip(model.state_dict().values(), control.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_permutation_and_query_chunk_invariance(self):
        z, edges, support, labels, query = fixture()
        model = RectangleBranch(6, 'lncRNA', width=4).eval()
        with torch.no_grad():
            model.readout[-1].weight.normal_()
        actual = model(z, edges, query, support, labels)
        chunked = torch.cat([model(z, edges, query[:, i:i+1], support, labels) for i in range(query.shape[1])])
        torch.testing.assert_close(actual, chunked, atol=1e-6, rtol=1e-6)
        perm = {'lncRNA': torch.tensor([2, 0, 1]), 'miRNA': torch.tensor([1, 0]), 'drug': torch.tensor([3, 1, 0, 2])}
        inverse = {nt: torch.argsort(p) for nt, p in perm.items()}
        zz = {nt: x[perm[nt]] for nt, x in z.items()}
        ee = {key: torch.stack((inverse[key[0]][v[0]], inverse[key[2]][v[1]])) for key, v in edges.items()}
        mapped = lambda p: torch.stack((inverse['lncRNA'][p[0]], inverse['drug'][p[1]]))
        got = model(zz, ee, mapped(query), mapped(support), labels)
        torch.testing.assert_close(actual, got, atol=2e-6, rtol=2e-6)

    def test_joint_gradients_and_additive_intervention(self):
        z, edges, support, labels, query = fixture()
        model = RectangleModel(FakeBase(), 6, 'lncRNA', width=4)
        with torch.no_grad():
            model.branch.readout[-1].weight.normal_(std=.1)
        out = model(z, edges, query, {}, support, labels)
        out.square().sum().backward()
        for name, value in model.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all().item(), name)
        self.assertGreater(model.branch.layers[0].factors.weight.grad.abs().sum().item(), 0)
        direct = copy.deepcopy(model)
        direct.mode = 'additive'
        torch.testing.assert_close(direct(z, edges, query, {}, support, labels),
            model(z, edges, query, {}, support, labels, intervention='additive'))
        prior = model(z, edges, query, {}, support, labels, intervention='no_branch')
        torch.testing.assert_close(prior, model.base.decoder(z, query))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
