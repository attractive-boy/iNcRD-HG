import unittest

import numpy as np
import torch

from improve import masked_graph_teacher as method
from improve.overlap_block_models import graph_inputs


def tiny():
    torch.manual_seed(77)
    xs = [torch.randn(6, 5), torch.randn(4, 7)]
    edges = np.array([[0, 1, 2, 3, 4, 5], [6, 6, 7, 8, 9, 9]])
    graph = graph_inputs(edges, np.zeros(6, dtype=np.int64), np.empty((2, 0), dtype=np.int64), 10, 1, 'cpu')
    model = method.MaskedGraphTeacher([5, 7], [6, 4], 1, width=8, depth=1, heads=2, dropout=0.)
    return xs, graph, model


class MaskedGraphTeacherTest(unittest.TestCase):
    def test_teacher_is_stop_gradient_and_updates_by_ema(self):
        xs, graph, model = tiny()
        model.train()
        before = next(model.teacher_encoder.parameters()).detach().clone()
        loss, _ = model.auxiliary(xs, graph, torch.Generator().manual_seed(901))
        loss.backward()
        self.assertGreater(float(next(model.encoder.parameters()).grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.teacher_encoder.parameters()))
        self.assertFalse(model.teacher_encoder.training)
        with torch.no_grad():
            next(model.encoder.parameters()).add_(1.)
        model.update_teacher()
        torch.testing.assert_close(next(model.teacher_encoder.parameters()), before + .01)

    def test_masked_encoder_cannot_observe_replaced_raw_attributes(self):
        xs, graph, model = tiny()
        model.eval()
        mask = torch.zeros(10, dtype=torch.bool)
        mask[2] = True
        first = model.encoder(xs, graph, mask)
        changed = [x.clone() for x in xs]
        changed[0][2] = torch.arange(5) * 1000.
        torch.testing.assert_close(first, model.encoder(changed, graph, mask), atol=0., rtol=0.)

    def test_masks_cover_each_type_without_duplicates(self):
        mask = method.node_mask([6, 4], .5, 'cpu', torch.Generator().manual_seed(402))
        self.assertEqual(int(mask[:6].sum()), 3)
        self.assertEqual(int(mask[6:].sum()), 2)
        self.assertEqual(mask.dtype, torch.bool)

    def test_cosine_loss_has_expected_scale_and_ignores_missing_vectors(self):
        x = torch.tensor([[1., 0.], [0., 0.]], requires_grad=True)
        self.assertEqual(float(method.cosine_error(x, x.detach())), 0.)
        self.assertEqual(float(method.cosine_error(-x, x.detach())), 4.)
        missing = method.cosine_error(x, torch.zeros_like(x))
        missing.backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
