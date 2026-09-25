"""Leakage boundaries, matched controls, aggregation and frozen selection."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from improve import tabular_foundation as base
from improve import tabicl_graph_context as experiment
from improve import tabicl_graph_context_modules as kernels


def small_arrays():
    pairs = np.array([(i//4, i % 4) for i in range(20)], dtype=np.int64).T
    query = np.array([(i//4, i % 4) for i in range(20, 28)], dtype=np.int64).T
    return pairs, (np.arange(20) % 2).astype(np.float32), np.arange(20) % 5, query


class ProtocolTests(unittest.TestCase):
    def test_context_positives_and_queries_are_never_support(self):
        pairs, labels, folds, query = small_arrays()
        plan = kernels.support_plan(pairs, labels, folds, query)
        self.assertEqual(len(plan), 5)
        np.testing.assert_array_equal(np.sort(np.concatenate([v['context'] for v in plan])), np.arange(len(labels)))
        for item in plan:
            self.assertFalse(set(item['context']) & set(item['support']))
            self.assertTrue(np.all(labels[item['support']] == 1))
            self.assertTrue(np.all(folds[item['support']] != item['fold']))

    def test_overlap_duplicate_and_incomplete_fold_rejected(self):
        pairs, labels, folds, query = small_arrays()
        for bad_query in (pairs[:, :2], np.repeat(query[:, :1], 2, axis=1)):
            with self.assertRaises(ValueError):
                kernels.support_plan(pairs, labels, folds, bad_query)
        with self.assertRaises(ValueError):
            kernels.support_plan(pairs, labels, np.zeros(len(labels)), query)

    def test_frozen_and_half_weight_are_exact(self):
        baseline = np.array([.1, .7, .4])
        values = {key:np.array([.3, .9, .2]) for key in kernels.ARMS}
        cfgs = kernels.recipes()
        np.testing.assert_array_equal(kernels.combine(baseline, values, cfgs[0]), baseline)
        for cfg in cfgs[1:]:
            expected = baseline if cfg['weight'] == 0 else (1-cfg['weight'])*baseline+cfg['weight']*values[cfg['arm']]
            np.testing.assert_array_equal(kernels.combine(baseline, values, cfg), expected)
        with self.assertRaises(ValueError):
            kernels.combine(baseline, values, dict(id='posthoc', arm='query_bag', weight=.25))

    def test_average_requires_five_aligned_finite_probability_views(self):
        values = np.arange(15).reshape(5, 3)/20
        np.testing.assert_array_equal(kernels.average_views(values, 3), values.mean(0))
        for bad in (values[:4], values.T, values*10, np.full_like(values, np.nan)):
            with self.assertRaises(ValueError):
                kernels.average_views(bad, 3)

    def test_selection_uses_all_seeds_and_retains_tied_baseline(self):
        cfgs, seeds = kernels.recipes()[:2], [666, 0, 1]
        rows = [dict(recipe_id=cfg['id'], seed=seed, auc=.8, ap=.7, criterion=.75, macro_drug_ap=.6)
                for cfg in cfgs for seed in seeds]
        self.assertEqual(kernels.choose(rows, cfgs, seeds)['recipe'], cfgs[0])
        # One favorable seed cannot override worse mean results.
        rows[3]['criterion'] = .76
        rows[4]['criterion'] = .72
        self.assertEqual(kernels.choose(rows, cfgs, seeds)['recipe'], cfgs[0])
        rows[4]['criterion'] = .76
        self.assertEqual(kernels.choose(rows, cfgs, seeds)['recipe'], cfgs[1])
        rows[-1]['seed'] = 666
        with self.assertRaises(ValueError):
            kernels.choose(rows, cfgs, seeds)

    def test_nonfinite_selection_is_rejected(self):
        cfg = kernels.recipes()[0]
        with self.assertRaises(ValueError):
            kernels.choose([dict(recipe_id=cfg['id'], seed=666, auc=np.nan, ap=.8,
                criterion=.8, macro_drug_ap=.8)], [cfg], [666])

    def test_paired_control_has_identical_context_and_ignores_query_labels(self):
        pairs, y, folds, query = small_arrays()
        a = dict(raw_train=np.arange(20*36).reshape(20, 36), raw_val=np.ones((8, 36)),
                 raw_query_views=np.stack([np.full((8, 36), i+2) for i in range(5)]),
                 y_train=y, y_val=np.zeros(8), fold_ids=folds)
        p = dict(input_configs={'LncDrug':dict(input_view='raw')})
        x, labels, views = experiment.context_and_queries(p, 'LncDrug', a, 'fold2')
        np.testing.assert_array_equal(x, a['raw_train'][folds == 2])
        np.testing.assert_array_equal(labels, y[folds == 2])
        np.testing.assert_array_equal(views[0], a['raw_val'])
        np.testing.assert_array_equal(views[1], a['raw_query_views'][2])
        a['y_val'][:] = 1
        for first, second in zip((x, labels, views), experiment.context_and_queries(p, 'LncDrug', a, 'fold2')):
            np.testing.assert_array_equal(first, second)

    def test_own_selection_precedes_any_outer_npz_load(self):
        p = dict(tasks=['LncDrug'])
        with patch.object(experiment, 'selection', side_effect=ValueError('not selected')), \
             patch.object(base, 'load_features') as loader, patch.object(np, 'load') as np_loader:
            with self.assertRaisesRegex(ValueError, 'not selected'):
                experiment.features(Path('/unused'), p, 'LncDrug', 'outer')
            loader.assert_not_called()
            np_loader.assert_not_called()

    def test_selection_tamper_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = dict(sha256='protocol', tasks=['LncDrug'], recipes=kernels.recipes())
            value = base.seal(dict(protocol_sha256=p['sha256'],
                chosen={'LncDrug':dict(recipe=kernels.recipes()[0])}, artifacts_sha256={}))
            base.write_json(root/'SELECTION.json', value)
            self.assertEqual(experiment.selection(root, p), value)
            value['chosen']['LncDrug']['recipe'] = kernels.recipes()[1]
            base.write_json(root/'SELECTION.json', value)
            with self.assertRaises(ValueError):
                experiment.selection(root, p)


@unittest.skipUnless(importlib.util.find_spec('torch_geometric'), 'Requires the historical graph environment')
class GraphRebuildTests(unittest.TestCase):
    def test_rebuild_ignores_graph_stored_target_and_reverse_edges(self):
        import torch
        from torch_geometric.data import HeteroData
        from improve.graph_tabm_modules import PairFeatureEncoder
        from improve.tabicl_graph_context_features import build_views
        pairs, y, folds, query = small_arrays()
        graph = HeteroData()
        generator = torch.Generator().manual_seed(12)
        for node, n in [('lncRNA', 8), ('miRNA', 8), ('drug', 4)]:
            graph[node].x = torch.randn(n, 5, generator=generator)
        relations = [
            (('lncRNA','LncDrug','drug'), pairs), (('drug','rev_LncDrug','lncRNA'), pairs[::-1].copy()),
            (('miRNA','MiDrug','drug'), pairs), (('drug','rev_MiDrug','miRNA'), pairs[::-1].copy()),
            (('lncRNA','LncLnc','lncRNA'), np.array([[0,1,2,3],[1,0,3,2]])),
            (('miRNA','MiMi','miRNA'), np.array([[0,1,2,3],[1,0,3,2]])),
            (('drug','DrugDrug','drug'), np.array([[0,1,2,3],[1,0,3,2]])),
            (('lncRNA','LncMi','miRNA'), np.array([[0,1,2,3],[0,1,2,3]]))]
        for key, edges in relations:
            graph[key].edge_index = torch.from_numpy(edges.copy()).long()
        for task, rna in [('LncDrug', 'lncRNA'), ('MiDrug','miRNA')]:
            encoder = PairFeatureEncoder(graph, task)
            raw = None
            for item in kernels.support_plan(pairs, y, folds, query):
                values = encoder.transform(pairs[:,item['context']], pairs[:,item['support']]).values
                if raw is None:
                    raw = np.empty((len(y), values.shape[1]), dtype=np.float32)
                raw[item['context']] = values
            full = encoder.transform(query, pairs[:,y == 1]).values
            views, _, checks, error = build_views(encoder, pairs, y, folds, query, raw, full)
            self.assertEqual(error, 0.)
            self.assertTrue(all(v['query_support_overlap'] == 0 for v in checks))
            changed = graph.clone()
            # Deliberately insert held-out query edges into graph-stored target
            # relations. Explicit support reconstruction must still ignore them.
            changed[(rna,task,'drug')].edge_index = torch.from_numpy(query.copy()).long()
            changed[('drug','rev_'+task,rna)].edge_index = torch.from_numpy(query[::-1].copy()).long()
            replay, _, _, _ = build_views(PairFeatureEncoder(changed, task), pairs, y, folds, query, raw, full)
            np.testing.assert_array_equal(replay, views)

    def test_graph_worker_checks_own_outer_selection_before_loaders(self):
        from improve import tabicl_graph_context_features as builder
        args = argparse.Namespace(out_dir='/unused', phase='outer', task='LncDrug')
        with patch.object(experiment, 'header', return_value={}), \
             patch.object(experiment, 'selection', side_effect=ValueError('not selected')), \
             patch.object(builder.graph_parent, 'stage_inputs') as loader, patch.object(np, 'load') as np_loader:
            with self.assertRaisesRegex(ValueError, 'not selected'):
                builder.build(args)
            loader.assert_not_called()
            np_loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
