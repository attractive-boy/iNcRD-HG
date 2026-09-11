"""Information isolation and numerical tests for the replacement backbone."""
import unittest

import numpy as np
import torch

from improve.backbone_replacement import choose_checkpoint, lr_at
from improve.pair_exchange_modules import remove_target_pairs
from improve.pencil_backbone import HeteroPencil, SupportGraph


def fixture():
    torch.manual_seed(13)
    x = {'lncRNA': torch.randn(5, 7), 'drug': torch.randn(4, 9), 'miRNA': torch.randn(3, 6)}
    target = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]])
    edges = {('lncRNA', 'LncDrug', 'drug'): target,
             ('drug', 'rev_LncDrug', 'lncRNA'): target.flip(0),
             ('lncRNA', 'LncLnc', 'lncRNA'): torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
             ('lncRNA', 'second', 'lncRNA'): torch.tensor([[0, 1], [1, 0]]),
             ('miRNA', 'MiDrug', 'drug'): torch.tensor([[0, 1, 2], [0, 1, 2]])}
    weights = {k: torch.ones(v.shape[1]) for k, v in edges.items()}
    return x, edges, weights


class PencilTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_removed_queries_cannot_enter_tokens_or_counts(self):
        x, edges, weights = fixture()
        edges, weights = remove_target_pairs(edges, weights, 'LncDrug', 'lncRNA', 4,
                                              np.array([[0], [0]]))
        graph = SupportGraph(x, edges, weights)
        drug = graph.offsets['drug']
        self.assertEqual(int(graph.bits[drug, 0]), 0)
        self.assertEqual(int(graph.bits[0, drug]), 0)
        self.assertEqual(float(graph.degrees[drug, 0]), 1.)
        self.assertEqual(float(graph.degrees[0, 1]), 0.)
        ids, valid, adj, counts = graph.local(np.array([[0, drug, 1, -1]]))
        self.assertEqual(float(adj[0, :, 0, 1].sum()), 0.)
        self.assertEqual(float(adj[0, :, 1, 0].sum()), 0.)
        self.assertEqual(float(adj[0, :, 3, :].sum()), 0.)
        self.assertEqual(float(adj[0, :, :, 3].sum()), 0.)
        self.assertEqual(float(counts[0, 3].sum()), 0.)

    def test_parallel_relation_types_are_not_overwritten(self):
        x, edges, weights = fixture()
        graph = SupportGraph(x, edges, weights)
        _, _, adj, _ = graph.local(np.array([[0, 1, -1, -1]]))
        self.assertEqual(float(adj[0, 2, 0, 1]), 1.)
        self.assertEqual(float(adj[0, 3, 0, 1]), 1.)
        self.assertEqual(float(adj[0, 0, 0, 1]), 0.)

    def test_sampling_is_batch_independent_unique_and_endpoint_first(self):
        x, edges, weights = fixture()
        graph = SupportGraph(x, edges, weights)
        pairs = np.array([[0, 1, 4], [1, 2, 3]])
        full = graph.sample(pairs, 'lncRNA', 8, 91)
        singles = np.concatenate([graph.sample(pairs[:, i:i+1], 'lncRNA', 8, 91) for i in range(3)])
        np.testing.assert_array_equal(full, singles)
        for i, row in enumerate(full):
            self.assertEqual(row[0], pairs[0, i])
            self.assertEqual(row[1], pairs[1, i]+graph.offsets['drug'])
            self.assertEqual(len(np.unique(row[row >= 0])), int((row >= 0).sum()))

    def test_finite_gradients_and_deterministic_batched_inference(self):
        x, edges, weights = fixture()
        graph = SupportGraph(x, edges, weights)
        pairs = np.array([[0, 1, 4], [1, 2, 3]])
        sampled = graph.sample(pairs, 'lncRNA', 8, 42)
        model = HeteroPencil({k: v.shape[1] for k, v in x.items()}, tuple(edges),
                             'lncRNA', width=16, heads=2, budget=8, dropout=0.)
        logits = model(x, graph, sampled)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0., 1.]))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(float(model.layers[0].transformer.self_attn.in_proj_weight.grad.abs().sum()), 0.)
        self.assertIsNone(model.structural.weight.grad)
        model.eval()
        with torch.no_grad():
            full = model(x, graph, sampled)
            singles = torch.cat([model(x, graph, sampled[i:i+1]) for i in range(3)])
            torch.testing.assert_close(full, singles, atol=2e-6, rtol=2e-6)
            self.assertGreater(float((full-model(x, graph, sampled, 'no_graph')).abs().max()), 1e-7)
            self.assertGreater(float((full-model(x, graph, sampled, 'no_counts')).abs().max()), 1e-7)

    def test_checkpoint_rule_keeps_both_metrics_together_and_first_tie(self):
        trace = [dict(criterion=x) for x in [.5, .7, .9, .7, .9]]
        epoch, score = choose_checkpoint(trace, 2)
        self.assertEqual(epoch, 3)
        self.assertAlmostEqual(score, .8)

    def test_refit_learning_rate_uses_original_horizon(self):
        cfg = dict(lr=1e-3, schedule='cosine')
        first_part = [lr_at(cfg, e, 150) for e in range(1, 41)]
        self.assertEqual(first_part, [lr_at(cfg, e, 150) for e in range(1, 151)][:40])
        self.assertNotEqual(first_part[-1], lr_at(cfg, 40, 40))


if __name__ == '__main__':
    unittest.main()
