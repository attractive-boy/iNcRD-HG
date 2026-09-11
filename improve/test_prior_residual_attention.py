"""Numerical contracts for prior-constrained message passing."""
import copy
import unittest

import numpy as np
import torch
from torch_geometric.nn import GeneralConv

from improve.attention_mechanisms import parameter_hash
from improve.drug_condition_attention import make_base, set_seed
from improve.edge_path_attention import configure
from improve.edge_path_modules import AttributeConv
from improve.prior_residual_modules import ARMS, NEW_ARMS, PriorResidualConv, make_model
from improve.test_edge_path_attention import fixture


def local_fixture():
    set_seed(32)
    # Includes a duplicate self edge, a node with a single neighbour, and an
    # isolated node. Unequal priors make normalization errors observable.
    edges = torch.tensor([[0, 1, 2, 0, 0, 1, 2], [0, 0, 0, 0, 1, 1, 2]])
    signed = torch.tensor([1., -.4, .9, 1., .6, 1., 1.])
    attributes = torch.stack([signed, signed.abs()], -1)
    return GeneralConv((5, 5), 7), edges, attributes, torch.randn(4, 5)


class PriorResidualChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_zero_correction_matches_fixed_prior_and_uniform_control(self):
        original, edges, attrs, x = local_fixture()
        fixed = AttributeConv(original, edges, attrs, 4, 'edge_fixed', 11, 4)
        for arm in ('prior_bounded', 'prior_free'):
            module = PriorResidualConv(original, edges, attrs, 4, arm, 11, 4)
            torch.testing.assert_close(module((x, x), edges), fixed((x, x), edges), rtol=0, atol=0)
        uniform = PriorResidualConv(original, edges, attrs, 4, 'bounded_uniform', 11, 4)
        torch.testing.assert_close(uniform((x, x), edges), original((x, x), edges), rtol=0, atol=0)

    def test_degree_self_loops_and_normalized_trust_region_under_extreme_scores(self):
        original, edges, attrs, x = local_fixture()
        for arm in NEW_ARMS:
            module = PriorResidualConv(original, edges, attrs, 4, arm, 11, 4)
            torch.nn.init.normal_(module.scorer.weight, std=100.)
            actual = module((x, x), edges)
            self.assertTrue(torch.isfinite(actual).all())
            weights = module.last_weights
            mask = edges[0] != edges[1]
            np.testing.assert_array_equal(weights[~mask].numpy(), np.ones(int((~mask).sum())))
            total = np.bincount(edges[1, mask], weights=weights[mask], minlength=4)
            degree = np.bincount(edges[1, mask], minlength=4)
            np.testing.assert_allclose(total, degree, rtol=1e-6, atol=1e-6)
            if arm != 'prior_free':
                ratio = weights[mask] / module.fixed_weights[mask]
                self.assertTrue(bool((ratio >= .6 - 1e-6).all()))
                self.assertTrue(bool((ratio <= 1 / .6 + 1e-6).all()))

    def test_dense_reference_matches_messages_and_gradients(self):
        original, edges, attrs, x = local_fixture()
        for arm in ('prior_bounded', 'prior_free'):
            module = PriorResidualConv(original, edges, attrs, 4, arm, 11, 4).double()
            torch.nn.init.normal_(module.scorer.weight, std=.7)
            reference = copy.deepcopy(module)
            actual = module((x.double(), x.double()), edges)
            normalized = torch.nn.functional.layer_norm(x.double(), (5,))
            weights = []
            for k, (s, d) in enumerate(edges.t().tolist()):
                if s == d:
                    weights.append(x.new_tensor(1., dtype=torch.double))
                    continue
                members = [j for j, (a, b) in enumerate(edges.t().tolist()) if b == d and a != b]
                masses = []
                for j in members:
                    a, b = edges[:, j]
                    hidden = reference.source(normalized[a]) + reference.target(normalized[b]) + reference.attribute(attrs[j].double())
                    score = reference.scorer(torch.nn.functional.leaky_relu(hidden, .2)).squeeze()
                    factor = (1 + .25 * score.tanh()) if arm == 'prior_bounded' else (.25 * score).exp()
                    masses.append(attrs[j, 1].double() * factor)
                masses = torch.stack(masses)
                weights.append(len(members) * masses[members.index(k)] / masses.sum())
            expected = reference.core((x.double(), x.double()), edges, edge_attr=torch.stack(weights))
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            actual.square().sum().backward()
            expected.square().sum().backward()
            for (name, a), (_, b) in zip(module.named_parameters(), reference.named_parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=1e-8, atol=1e-9, msg=name)

    def test_shuffling_changes_only_the_prior(self):
        original, _, _, _ = local_fixture()
        edges = torch.tensor([[0, 1, 0, 2, 1, 2, 0, 1, 2], [1, 0, 2, 0, 2, 1, 0, 1, 2]])
        signed = torch.tensor([.4, .4, -.5, -.5, .8, .8, 1., 1., 1.])
        attrs = torch.stack([signed, signed.abs()], -1)
        module = PriorResidualConv(original, edges, attrs, 3, 'prior_shuffled', 11, 4)
        torch.testing.assert_close(module.attributes, attrs, rtol=0, atol=0)
        torch.testing.assert_close(module.prior_strength.sort().values, attrs[module.nonself, 1].sort().values)
        self.assertFalse(torch.equal(module.prior_strength, attrs[module.nonself, 1]))
        for i in (0, 2, 4):
            self.assertEqual(module.prior_strength[i], module.prior_strength[i + 1])

    def test_empty_similarity_edges_remain_finite(self):
        original, _, _, x = local_fixture()
        edges = torch.empty(2, 0, dtype=torch.long)
        attrs = torch.empty(0, 2)
        for arm in NEW_ARMS:
            module = PriorResidualConv(original, edges, attrs, 4, arm, 11, 4)
            actual = module((x, x), edges)
            torch.testing.assert_close(actual, original((x, x), edges), rtol=0, atol=0)

    def test_trainable_scorer_and_interventions_restore_original_prediction(self):
        data, key, assets, _ = fixture()
        relation = ('miRNA', 'MiMi', 'miRNA')
        # Make the real-graph fixture priors nonuniform without changing topology.
        for k, (a, b) in enumerate(data[relation].edge_index.t().tolist()):
            if a != b:
                value = .4 + .1 * ((a + b) % 4)
                assets['attributes'][relation][k] = torch.tensor([value, value])
        cfg = dict(hidden=8, rna='miRNA', aggr='mean')
        protocol = dict(width=4, edge_width=4, epsilon=.25)
        for arm in ARMS:
            states = []
            for repeat in range(2):
                base = make_base(data, cfg, 42, 'cpu')
                set_seed(42 + 3000003)
                model = make_model(base, data, assets, cfg, arm, 42, protocol)
                set_seed(42 + 9000007)
                before = {n: p.clone() for n, p in model.named_parameters()}
                optimizer = torch.optim.RMSprop((p for p in model.parameters() if p.requires_grad), lr=.001)
                for _ in range(4):
                    optimizer.zero_grad()
                    score = model(data.x_dict, data.edge_index_dict, 'train')
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, data[key].edge_label)
                    loss.backward()
                    for p in model.parameters():
                        if p.grad is not None:
                            self.assertTrue(torch.isfinite(p.grad).all())
                    optimizer.step()
                if arm in NEW_ARMS:
                    self.assertTrue(any(n.endswith('scorer.weight') and not torch.equal(before[n], p)
                                        for n, p in model.named_parameters()))
                    normal = model(data.x_dict, data.edge_index_dict, 'train').detach()
                    model(data.x_dict, data.edge_index_dict, 'train', 'prior_only')
                    replay = model(data.x_dict, data.edge_index_dict, 'train')
                    torch.testing.assert_close(normal, replay, rtol=0, atol=0)
                states.append(parameter_hash(model))
            self.assertEqual(states[0], states[1], arm)


if __name__ == '__main__':
    unittest.main()
