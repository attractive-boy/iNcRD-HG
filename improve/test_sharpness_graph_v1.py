"""Shared architecture, support boundaries, and exact LncDrug identity."""
import unittest

import numpy as np
import torch

from improve import sharpness_graph_v1 as core
from improve import anchor_residual_graph_v2 as executor
from improve.test_nested_anchor_v2 import fixture


class SharpnessGraphTests(unittest.TestCase):
    def test_midrug_arms_have_identical_architecture_and_omitted_support(self):
        a, _ = fixture()
        p = core.parameters(); p.update(width=8, heads=2, dropout=0.)
        reference = None
        for kind in core.KINDS:
            torch.manual_seed(179)
            model, xs, graphs, folds = core.setup(a, p, kind, 'cpu')
            if reference is None:
                reference = model.state_dict()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, reference[key], atol=0, rtol=0)
            omitted = folds[0].numpy()
            changed = {k: v.copy() for k, v in a.items()}
            changed['y_train'][omitted] = 1 - changed['y_train'][omitted]
            changed['y_val'] = np.full(6, 888)
            _, _, other, _ = core.setup(changed, p, kind, 'cpu')
            for x, y in zip(graphs[0], other[0]):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            query = torch.as_tensor(a['pairs_train'][:, omitted] + a['pair_offset'][:, None])
            edges = set(map(tuple, graphs[0][0].numpy().T))
            self.assertFalse(any(tuple(q) in edges or tuple(q[::-1]) in edges for q in query.numpy().T))
            torch.testing.assert_close(model(xs, graphs[0], query), torch.zeros(len(omitted)))

    def test_lnc_policy_is_exact_anchor_for_every_optimizer_arm(self):
        a, _ = fixture()
        keep = a['pairs_train'][0] < int(a['node_counts'][0])
        for key in ('y_train', 'folds'):
            a[key] = a[key][keep]
        a['pairs_train'] = a['pairs_train'][:, keep]
        a['pair_offset'] = np.array([0, 14])
        p = core.parameters()
        query = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
        for kind in core.KINDS:
            model, xs, graphs, _ = core.setup(a, p, kind, 'cpu')
            self.assertEqual(list(model.parameters()), [])
            self.assertEqual(xs, [])
            residual = executor.predict(model, xs, graphs, query, {})
            anchor = torch.linspace(-15., 15., query.shape[1], dtype=torch.float64)
            torch.testing.assert_close(anchor + .1 * residual, anchor, atol=0, rtol=0)
            self.assertEqual(model.state_dict(), {})


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
