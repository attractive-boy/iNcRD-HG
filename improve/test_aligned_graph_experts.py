import unittest

import numpy as np
import torch

from improve.aligned_graph_experts import context, transformed, mix_experts, setup, KINDS


def toy():
    rng = np.random.default_rng(13)
    return dict(node_counts=np.array([3, 2]), nd=np.array(2), pair_offset=np.array([0, 3]),
        pairs_train=np.array([[0, 1, 2], [0, 1, 1]]), y_train=np.array([1, 1, 0]),
        folds=np.array([0, 1, 0]), edges=np.array([[0, 1, 0, 1], [3, 4, 1, 2]]),
        relation=np.array([0, 0, 1, 1]), target_codes=np.array([0, 3, -1, -1]),
        x_0=rng.normal(size=(3, 2)).astype(np.float32),
        x_1=rng.normal(size=(2, 4)).astype(np.float32))


class AlignedExpertsTests(unittest.TestCase):
    def test_task_mixture_cannot_backpropagate_into_gate_but_trains_experts(self):
        experts = torch.randn(7, 3, 5, requires_grad=True)
        gate = torch.randn(7, 3, requires_grad=True)
        prediction = mix_experts(experts, gate, 'full')
        prediction.square().sum().backward()
        self.assertIsNone(gate.grad)
        self.assertGreater(experts.grad.abs().sum().item(), 0)
        expected = (experts.detach() * gate.detach().softmax(-1)[..., None]).sum(1)
        torch.testing.assert_close(prediction, expected)

    def test_synthetic_shifts_never_reinsert_queries_or_change_inverse_edge_pairing(self):
        a = toy()
        graph = context(a, 0, 'cpu')
        xs = [torch.as_tensor(a['x_0']), torch.as_tensor(a['x_1'])]
        original_rng = torch.random.get_rng_state().clone()
        for code in range(4):
            changed, physical, target = transformed(xs, graph, torch.Generator().manual_seed(4), code)
            edges, relations, counts = physical
            self.assertFalse(((edges[0] == 0) & (edges[1] == 3)).any())
            self.assertFalse(((edges[0] == 3) & (edges[1] == 0)).any())
            self.assertEqual(counts[0, 0].item(), 0)
            forward = edges[:, relations < graph['nr']]
            backward = edges[:, (relations >= graph['nr']) & (relations < 2 * graph['nr'])]
            torch.testing.assert_close(forward, backward.flip(0))
            self.assertEqual(target.tolist(), [float(code == 0), float(bool(code & 1)), float(bool(code & 2))])
            if not code & 2:
                for before, after in zip(xs, changed):
                    torch.testing.assert_close(before, after, atol=0, rtol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), original_rng, atol=0, rtol=0)

    def test_alignment_has_both_gradient_paths_and_uniform_is_literal_average(self):
        experts = torch.randn(4, 3, 5, requires_grad=True)
        reference = torch.randn(4, 5, requires_grad=True)
        gate = torch.randn(4, 3, requires_grad=True)
        mixed = mix_experts(experts, gate, 'uniform')
        torch.testing.assert_close(mixed, experts.mean(1))
        (torch.linalg.vector_norm(mixed - reference) / 4).backward()
        self.assertGreater(experts.grad.abs().sum().item(), 0)
        self.assertGreater(reference.grad.abs().sum().item(), 0)
        self.assertIsNone(gate.grad)

    def test_matched_initialization_query_reordering_and_actual_training_batch(self):
        a = toy()
        p = dict(width=16, depth=2, dropout=0., folds=2)
        states = []
        for kind in KINDS:
            torch.manual_seed(7)
            model, xs, graphs, _ = setup(a, p, kind, 'cpu')
            states.append(model.state_dict())
            pairs = torch.tensor([[0, 1], [3, 4]])
            prediction = model(xs, graphs[0], pairs)
            torch.testing.assert_close(prediction, model(xs, graphs[0], pairs.flip(1)).flip(0))
            values, auxiliary, detail = model.training_batch(xs, graphs[0], pairs, torch.Generator().manual_seed(11))
            (values.square().mean() + auxiliary).backward()
            self.assertTrue(torch.isfinite(model.encoder.project[0].weight.grad).all())
            if kind != 'erm':
                self.assertGreater(model.gate_head[-1].weight.grad.abs().sum().item(), 0)
            else:
                self.assertEqual(auxiliary.item(), 0)
        for state in states[1:]:
            for key in state:
                torch.testing.assert_close(state[key], states[0][key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
