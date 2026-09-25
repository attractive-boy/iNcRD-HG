"""Mechanism, missingness, source isolation and execution-budget checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from improve import anchor_morphological_alignment_v1 as experiment
from improve import morphological_alignment_v1 as core
from improve import tabular_foundation as base


def fixture(folder):
    rng = np.random.default_rng(32018)
    matrix = np.zeros((8, 9), dtype=np.float32)
    for i in range(7):
        indices = np.unique([i, (i + 1) % 9, (2 * i + 1) % 9])
        matrix[i, indices] = 1 / len(indices)
    dti = np.array([(d, (d + j) % 9) for d in range(7) for j in (0, 1)], dtype=np.int64).T
    data = dict(drug_profile=rng.normal(size=(7, 4)).astype(np.float32),
        gene_profile=rng.normal(size=(9, 3)).astype(np.float32), rna_gene=matrix,
        rna_gene_counts=np.count_nonzero(matrix, axis=1).astype(np.float32),
        rna_permutation=rng.permutation(9), project_drug_index=np.array([0, 1, 2, 3, 4, -1]),
        dti_pairs=dti, drug_keys=np.array(['d%d' % i for i in range(7)]),
        gene_ids=np.array(['g%d' % i for i in range(9)]), drug_names=np.array(['p%d' % i for i in range(6)]),
        rna_names=np.array(['r%d' % i for i in range(8)]))
    np.savez_compressed(folder / 'arrays.npz', **data)
    base.write_json(folder / 'INPUTS.json', base.seal(dict(association_outcomes_read=False,
        new_neural_fits=0, files={'arrays.npz': base.digest(folder / 'arrays.npz')})))
    return data


def training(lnc=False):
    r, d = np.meshgrid(np.arange(5 if lnc else 8), np.arange(6), indexing='ij')
    pairs = np.stack((r.ravel(), d.ravel())).astype(np.int64)
    return dict(node_counts=np.array([5, 8, 6, 3]), pair_offset=np.array([0 if lnc else 5, 13]),
        pairs_train=pairs, folds=np.arange(pairs.shape[1]) % 5,
        y_train=((pairs[0] + pairs[1]) % 3 == 0).astype(np.int64),
        anchor_oof=np.linspace(-1, 1, pairs.shape[1]).astype(np.float64))


class MechanismChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.data = fixture(self.folder)
        self.c = core.parameters(self.folder)
        self.c.update(width=8, hidden=12, dti_batch=16)
        self.p = dict(smoke=True, implementation_module='morphological_alignment_v1',
            source_sha256={'morphological_alignment_v1.py': base.digest(core.__file__)},
            core_parameters=self.c, kinds=list(core.KINDS), epochs=3, check_interval=20,
            weight_decay=.001, strong_weight=.1, private_rng_offset=32017)
        torch.set_num_threads(2)

    def tearDown(self):
        self.temp.cleanup()

    def test_multipositive_value_gradient_and_external_mask(self):
        x = torch.tensor([[.2, -.5, .8], [.3, .9, -.1], [-.2, .1, .7]], dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=torch.bool)
        actual = core.multi_positive_loss(x, mask)
        terms = []
        for logits, m in [(x, mask), (x.T, mask.T)]:
            directional = []
            for i in range(3):
                vals = [-(logits[i, j] - torch.logsumexp(logits[i], 0)) for j in range(3) if m[i, j]]
                directional.append(torch.stack(vals).mean())
            terms.append(torch.stack(directional).mean())
        expected = torch.stack(terms).mean()
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0],
                                   torch.autograd.grad(expected, x)[0], atol=1e-13, rtol=1e-13)
        keys = torch.tensor([0, 1, 4, 8])
        result = core.positive_mask(keys, torch.tensor([0, 1, 2, 0]), torch.tensor([0, 1, 2]), 3)
        expected = torch.tensor([[(3 * i + j) in (0, 1, 4, 8) for j in (0, 1, 2)] for i in (0, 1, 2, 0)])
        self.assertTrue(torch.equal(result, expected))
        with self.assertRaises(ValueError):
            core.multi_positive_loss(x, torch.zeros_like(mask))

    def test_permutation_preserves_sets_but_changes_rna_proxy_only(self):
        a = training()
        full, xs, _, _ = core.setup(a, self.c, 'full', 'cpu')
        shuffled, _, _, _ = core.setup(a, self.c, 'shuffled_rna_targets', 'cpu')
        shuffled.load_state_dict(full.state_dict())
        full.eval(); shuffled.eval()
        real, wrong = xs['rna_gene'], xs['rna_gene_shuffled']
        torch.testing.assert_close(real.sum(1), wrong.sum(1))
        torch.testing.assert_close(real @ real.T, wrong @ wrong.T)
        f, s = full.encode(xs), shuffled.encode(xs)
        self.assertTrue(torch.equal(f[0], s[0]) and torch.equal(f[1], s[1]))
        self.assertGreater(float((f[2] - s[2]).abs().max()), 1e-4)
        g1 = torch.Generator().manual_seed(12)
        g2 = torch.Generator().manual_seed(12)
        loss1, detail1 = core.alignment_loss(f, xs, self.c, g1)
        loss2, detail2 = core.alignment_loss(s, xs, self.c, g2)
        self.assertTrue(torch.equal(loss1, loss2))
        self.assertEqual(detail1, detail2)
        self.assertTrue(torch.equal(g1.get_state(), g2.get_state()))

    def test_missingness_zero_initialization_and_query_replay(self):
        a = training()
        model, xs, graphs, _ = core.setup(a, self.c, 'full', 'cpu')
        query = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
        model.eval()
        self.assertTrue(torch.equal(model(xs, None, query), torch.zeros(query.shape[1])))
        with torch.no_grad():
            model.readout[-1].weight.fill_(.1)
            model.readout[-1].bias.fill_(.2)
        value = model(xs, None, query)
        self.assertLessEqual(float(value.abs().max()), 2)
        self.assertTrue(torch.equal(value[~model.covered(xs, query)], torch.zeros_like(value[~model.covered(xs, query)])))
        torch.testing.assert_close(value, model(xs, None, query.flip(1)).flip(0), atol=3e-5, rtol=3e-5)
        chunks = torch.cat([model(xs, None, query[:, i:i+7]) for i in range(0, query.shape[1], 7)])
        torch.testing.assert_close(value, chunks, atol=3e-5, rtol=3e-5)
        self.assertEqual(graphs, [None] * 5)
        with self.assertRaises(ValueError):
            model(xs, torch.zeros(1), query)

    def test_external_inputs_ignore_project_graph_outcomes_and_attributes(self):
        a = training()
        b = {k: v.copy() for k, v in a.items()}
        b['y_train'] = 1 - b['y_train']
        b['anchor_oof'] *= -11
        b['edges'] = np.array([[99999], [-1]])
        b['relation'] = np.array([9123])
        b['x_1'] = np.full((8, 4), np.nan)
        m1, x1, g1, _ = core.setup(a, self.c, 'full', 'cpu')
        m2, x2, g2, _ = core.setup(b, self.c, 'full', 'cpu')
        self.assertEqual(g1, g2)
        for key in x1:
            self.assertTrue(torch.equal(x1[key], x2[key]), key)
        m2.load_state_dict(m1.state_dict())
        m1.eval(); m2.eval()
        q = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None])
        self.assertTrue(torch.equal(m1(x1, None, q), m2(x2, None, q)))
        changed = {k: v.copy() for k, v in self.data.items()}
        changed['rna_gene_counts'][0] += 1
        with self.assertRaises(ValueError):
            core.validate_inputs(changed)

    def test_fixed_training_budget_rng_and_no_validation_access(self):
        a = training()
        streams, params = [], []
        for kind in core.KINDS:
            model, value, trace, detail = experiment.fit(a, self.p, kind, .001, 32019, 'cpu', fixed_epochs=3)
            self.assertIsNone(value)
            self.assertEqual(len(trace), 3)
            self.assertEqual(detail['optimizer_steps'], 3)
            self.assertEqual(detail['gradient_evaluations'], 3)
            self.assertEqual(detail['dti_pairs_per_update'], 16)
            self.assertEqual([r['training_fold'] for r in trace], [0, 1, 2])
            self.assertFalse(any('anchor_composed' in r for r in trace))
            for r in trace:
                self.assertAlmostEqual(r['loss'], r['bce'] + r['alignment_multiplier'] * r['alignment_loss'], places=7)
                self.assertTrue(r['external_positive_supervision_verified'])
                self.assertFalse(r['association_graph_used'])
                self.assertEqual(r['sampled_pairs'], 16)
            state = model.training_final_state
            self.assertTrue(state['optimizer']['state'])
            self.assertTrue(all(int(v['step']) == 3 for v in state['optimizer']['state'].values()))
            streams.append((torch.random.get_rng_state().clone(), state['private_rng']))
            params.append(detail['parameters'])
        self.assertEqual(len(set(params)), 1)
        for state in streams[1:]:
            self.assertTrue(torch.equal(state[0], streams[0][0]))
            self.assertTrue(torch.equal(state[1], streams[0][1]))
        with self.assertRaises(ValueError):
            experiment.fit(dict(a, y_val=np.ones(2)), self.p, 'full', .001, 0, 'cpu', fixed_epochs=3)

    def test_lnc_identity_does_not_load_phenotypes_or_train(self):
        a = training(lnc=True)
        p = dict(self.p, core_parameters=dict(self.c, input_dir='/nonexistent/forbidden-phenotype-path'))
        model, value, trace, detail = experiment.fit(a, p, 'full', .001, 4, 'cpu', fixed_epochs=3)
        self.assertIsNone(value)
        self.assertEqual(trace, [])
        self.assertEqual(list(model.parameters()), [])
        self.assertEqual(detail['training_steps'], 0)
        self.assertFalse(detail['trained_neural_model'])
        self.assertIsNone(model.training_final_state)


if __name__ == '__main__':
    unittest.main()
