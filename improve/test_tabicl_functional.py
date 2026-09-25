import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from scipy import sparse

from improve import tabular_foundation as base
from improve import tabicl_functional as experiment
from improve import tabicl_functional_modules as feature
from improve.tabicl_context_program_modules import ContextPrograms
from improve.pathway_context_data import CONTEXT_NAMES


class FunctionalEvidenceTests(unittest.TestCase):
    def inputs(self):
        gp=sparse.csr_matrix(np.eye(5))
        dg=sparse.csr_matrix([[1,1,1,1,0],[0,1,0,0,0],[0,0,0,0,0]],dtype=float)
        rg=sparse.csr_matrix([[1,0,0,0,0],[0,1,1,0,0],[0,0,0,0,0]],dtype=float)
        layers=dict(functional=rg,binding=rg[:,[1,0,2,3,4]],association=sparse.csr_matrix(rg.shape))
        return gp,dg,{'MiDrug':layers,'LncDrug':layers}

    def model(self):
        gp,dg,tiers=self.inputs()
        return feature.FunctionalPrograms(gp,dg,dg,dg,tiers)

    def test_same_data_summary_exactly_matches_accepted_26_columns(self):
        gp,dg,tiers=self.inputs()
        old=ContextPrograms(gp,dg,tiers,{t:np.zeros((3,3,len(CONTEXT_NAMES))) for t in tiers})
        new=self.model()
        for task in tiers:
            np.testing.assert_array_equal(old.table(task,'tiered'),new.table(task,'expanded'))
            for variant in feature.VARIANTS:
                self.assertEqual(new.table(task,variant).shape[-1],26)
                np.testing.assert_array_equal(new.table('MiDrug',variant),new.table('LncDrug',variant))

    def test_target_local_features_match_hand_computed_pathway_scores(self):
        table=self.model().table('MiDrug','target_local')
        self.assertAlmostEqual(table[0,0,16],.25)
        self.assertAlmostEqual(table[0,0,17],1/3,places=6)
        np.testing.assert_array_equal(table[:,2,[16,17,20,21,24,25]],0)
        np.testing.assert_array_equal(table[2,:,[14,15,16,17,18,19,20,21,22,23,24,25]],0)

    def test_gene_and_pathway_order_do_not_change_local_features(self):
        gp,dg,tiers=self.inputs();order=np.array([3,0,4,1,2])
        permuted=feature.FunctionalPrograms(gp[order][:,order[::-1]],dg[:,order],dg[:,order],dg[:,order],
            {t:{k:v[:,order] for k,v in layer.items()} for t,layer in tiers.items()})
        for variant in feature.VARIANTS:
            np.testing.assert_allclose(self.model().table('MiDrug',variant),permuted.table('MiDrug',variant),atol=1e-7)

    def test_rewiring_preserves_old_edges_and_both_degree_sequences(self):
        old=sparse.csr_matrix(np.eye(8))
        added=sparse.csr_matrix((np.ones(16),(np.repeat(np.arange(8),2),
            np.array([[(i+1)%8,(i+3)%8] for i in range(8)]).ravel())),shape=(8,8))
        expanded=old+added
        rewired,info=feature.rewire_additions(old,expanded)
        np.testing.assert_array_equal(rewired.sum(0),expanded.sum(0))
        np.testing.assert_array_equal(rewired.sum(1),expanded.sum(1))
        np.testing.assert_array_equal(rewired.multiply(old).toarray(),old.toarray())
        self.assertEqual(info['accepted_swaps'],info['target_swaps'])
        self.assertGreater(info['fraction_added_edges_changed'],.25)
        np.testing.assert_array_equal(rewired.toarray(),feature.rewire_additions(old,expanded)[0].toarray())

    def test_pair_order_and_repeated_queries_leave_features_unchanged(self):
        pairs=np.array([[0,1,2,0],[0,1,2,0]]);model=self.model()
        for variant in feature.VARIANTS:
            expected=model.transform('MiDrug',pairs,variant)
            np.testing.assert_array_equal(expected[::-1],model.transform('MiDrug',pairs[:,::-1],variant))
            np.testing.assert_array_equal(expected[0],expected[-1])
        with self.assertRaises(ValueError):
            model.transform('MiDrug',np.array([[-1],[0]]),'expanded')


class ProtocolTests(unittest.TestCase):
    def test_replacement_keeps_original_graph_weight(self):
        graph=np.array([.1,.5,.9]);old=np.array([.9,.3,.2]);new=np.array([.3,.2,.4])
        baseline=.5*graph+.5*old
        np.testing.assert_array_equal(feature.compose(baseline,graph,new,.5,0),baseline)
        np.testing.assert_allclose(feature.compose(baseline,graph,new,.5,.25),.5*graph+.375*old+.125*new)
        np.testing.assert_array_equal(feature.compose(baseline,graph,old,.5,1),baseline)

    def test_gate_rejects_one_tiny_lnc_drop_and_all_controls(self):
        rows=[dict(task=t,seed=s,recipe=r,auc=.8+(.01 if t=='MiDrug' and r=='c' else 0),
                   ap=.7+(.01 if t=='MiDrug' and r=='c' else 0),macro_drug_ap=.5)
              for t in ('MiDrug','LncDrug') for s in (666,0,1) for r in ('baseline','c')]
        config=dict(id='c',variant='expanded',alpha=.5,role='candidate')
        self.assertTrue(feature.assess(rows,config,[666,0,1])['eligible'])
        self.assertFalse(feature.assess(rows,dict(config,variant='expanded_rewired'),[666,0,1])['eligible'])
        rows[-1]['ap']-=1e-10
        self.assertFalse(feature.assess(rows,config,[666,0,1])['eligible'])

    def test_outer_gate_precedes_loading_any_reference_arrays(self):
        with mock.patch.object(experiment,'selected',side_effect=ValueError('ineligible')):
            with mock.patch.object(experiment.accepted,'stage_data') as loader:
                with self.assertRaises(ValueError):
                    experiment.stage_data('/unused',{'tasks':['MiDrug']},'MiDrug','outer')
                loader.assert_not_called()

    def test_comparator_uses_saved_accepted_predictor_not_older_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);path='predictions/inner/MiDrug/s666/scores.npz'
            (out/path).parent.mkdir(parents=True)
            a=dict(y_val=np.array([1,0]),pairs_val=np.array([[0,1],[0,1]]))
            strong=np.array([.9,.1]);weak=np.array([.6,.4])
            np.savez_compressed(out/path,tiered_0p5000=strong,baseline=weak,**a)
            base.write_json(out/'VALIDATION_inner.json',dict(protocol_sha256='reference',
                artifacts={path:base.digest(out/path)}))
            p=dict(reference_dir=folder,reference_protocol_sha256='reference',reference_recipe=dict(id='tiered_0p5000'))
            comparator,graph=experiment.reference_scores(p,'MiDrug','inner',666,a)
            np.testing.assert_array_equal(comparator,strong)
            np.testing.assert_array_equal(graph,weak)
            np.savez_compressed(out/path,tiered_0p5000=weak,baseline=weak,**a)
            with self.assertRaises(ValueError):
                experiment.reference_scores(p,'MiDrug','inner',666,a)


if __name__=='__main__':
    unittest.main()
