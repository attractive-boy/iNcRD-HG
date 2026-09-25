import unittest
from unittest import mock

import numpy as np
from scipy import sparse

from improve.pathway_context_data import (CONTEXT_NAMES, GeneLookup, contextual_features,
                                         lnctard_tier, mti_tier, numeric_id,
                                         program_projection, rank_units)
from improve.tabicl_context_program_modules import ContextPrograms, TIERS, names, assess
from improve import tabicl_context_program as experiment


class CuratedEvidenceTests(unittest.TestCase):
    def lookup(self):
        rows = []
        for i,symbol in enumerate(('A','B','C'),1):
            rows.append(dict(status='Approved',ensembl_gene_id='ENSG00000'+str(i),entrez_id=str(i),
                             symbol=symbol,prev_symbol='OLD'+symbol,alias_symbol='AMBIG' if i<3 else '',
                             locus_group='protein-coding gene' if i<3 else 'non-coding RNA'))
        return GeneLookup(rows)

    def test_identifier_conflicts_fail_instead_of_using_name(self):
        lookup = self.lookup()
        self.assertEqual(lookup.resolve('ENSG000001','2','A')[0],'')
        self.assertEqual(lookup.resolve(symbol='AMBIG')[0],'')
        self.assertEqual(lookup.resolve(entrez='1.0',symbol='obsolete')[0],'ENSG000001')
        self.assertEqual(lookup.resolve(symbol='OLDA')[0],'ENSG000001')
        self.assertEqual(lookup.resolve(entrez='3',protein=True)[0],'')
        self.assertEqual(numeric_id('1.5'),'1.5')

    def test_human_filename_is_not_a_species_filter(self):
        self.assertEqual(mti_tier('kshv','hsa','Functional MTI','reporter')[0],'')
        self.assertEqual(mti_tier('hsa','mmu','Functional MTI','reporter')[0],'')
        self.assertEqual(mti_tier('hsa','hsa','Non-Functional MTI','reporter')[0],'')
        self.assertEqual(mti_tier('hsa','hsa','Non-Functional MTI (Weak)','CLIP-seq')[0],'')

    def test_functional_binding_and_expression_evidence_are_distinct(self):
        self.assertEqual(mti_tier('hsa','hsa','Functional MTI','reporter')[0],'functional')
        self.assertEqual(mti_tier('hsa','hsa','Functional MTI (Weak)','PAR-CLIP')[0],'binding')
        self.assertEqual(mti_tier('hsa','hsa','Functional MTI (Weak)','Microarray')[0],'association')
        with self.assertRaises(ValueError):
            mti_tier('hsa','hsa','unknown evidence','reporter')

    def test_lnctard_e_suffix_does_not_mean_weak_expression_association(self):
        row = dict(SearchregulatoryMechanism='ceRNA or sponge',RegulationDiretion='positively-E',Drugs='DRUG-X')
        self.assertEqual(lnctard_tier(row),'functional')
        row.update(Drugs='DRUG-Y',RegulationDiretion='negatively-F')
        self.assertEqual(lnctard_tier(row),'functional')
        row['SearchregulatoryMechanism']='expression association'
        self.assertEqual(lnctard_tier(row),'association')


class ExpressionProgramTests(unittest.TestCase):
    def inputs(self):
        genes = np.array([[1,9,3,0],[2,7,2,0],[4,5,4,0],[6,3,8,0],[8,2,7,0],[9,1,9,0]],float)
        rna = np.column_stack([genes[:,0],genes[:,1],np.ones(6)])
        groups = np.array(['a','a','a','b','b','b'])
        gp = sparse.csr_matrix([[1,0],[1,1],[0,1],[0,0]],dtype=float)
        dg = sparse.csr_matrix([[1,0,0,0],[0,1,1,0],[0,0,0,1],[0,0,0,0]],dtype=float)
        rg = sparse.csr_matrix([[1,0,0,0],[0,1,0,0],[0,0,0,0]],dtype=float)
        return rna,genes,groups,gp,dg,rg

    def test_rank_features_ignore_monotone_measurement_scale(self):
        values = self.inputs()[1]
        np.testing.assert_allclose(rank_units(values),rank_units(np.log2(values+1)),atol=1e-12)

    def test_lineage_centering_removes_lineage_only_association(self):
        values = np.array([[1],[1],[1],[8],[8],[8]],float)
        groups = self.inputs()[2]
        self.assertGreater(np.linalg.norm(rank_units(values)),0)
        np.testing.assert_array_equal(rank_units(values,groups),0)

    def test_constant_rna_missing_drug_and_unmeasured_gene_are_finite(self):
        rna,gene,groups,gp,dg,rg = self.inputs()
        table,_ = contextual_features(rna,gene,groups,dg,program_projection(gp),rg)
        self.assertEqual(table.shape,(3,4,len(CONTEXT_NAMES)))
        self.assertTrue(np.isfinite(table).all())
        np.testing.assert_array_equal(table[2,:,[0,*range(2,len(CONTEXT_NAMES))]],0)
        np.testing.assert_array_equal(table[:,3,1:],0)
        np.testing.assert_array_equal(table[:,2,1:],0)

    def test_matched_cell_and_gene_permutations_do_not_change_features(self):
        rna,gene,groups,gp,dg,rg = self.inputs()
        original,_ = contextual_features(rna,gene,groups,dg,program_projection(gp),rg)
        cell=[5,3,1,0,4,2]
        col=[2,3,0,1]
        permuted,_ = contextual_features(rna[cell],gene[cell][:,col],groups[cell],dg[:,col],
                                        program_projection(gp[col][:,[1,0]]),rg[:,col])
        np.testing.assert_allclose(original,permuted,atol=1e-7)

    def model(self):
        rna,gene,groups,gp,dg,rg = self.inputs()
        table,_ = contextual_features(rna,gene,groups,dg,program_projection(gp),rg)
        layers = dict(functional=rg,binding=rg[:,[1,0,2,3]],association=sparse.csr_matrix(rg.shape))
        return ContextPrograms(gp,dg,{'MiDrug':layers,'LncDrug':layers},{'MiDrug':table,'LncDrug':table})

    def test_same_evidence_context_rule_for_both_task_names(self):
        model = self.model()
        for variant in ('annotation','tiered','context','context_permuted'):
            np.testing.assert_array_equal(model.table('MiDrug',variant),model.table('LncDrug',variant))
            self.assertEqual(model.table('MiDrug',variant).shape[-1],len(names(variant)))

    def test_context_control_preserves_annotation_block_and_context_values(self):
        model = self.model()
        normal,control = model.table('MiDrug','context'),model.table('MiDrug','context_permuted')
        np.testing.assert_array_equal(normal[:,:,:26],control[:,:,:26])
        np.testing.assert_array_equal(np.sort(normal[:,:,26:].ravel()),np.sort(control[:,:,26:].ravel()))

    def test_query_batch_order_does_not_change_adapter(self):
        model = self.model()
        pairs=np.array([[0,1,2,0],[1,0,3,1]])
        expected=model.transform('MiDrug',pairs,'context')
        np.testing.assert_array_equal(expected[::-1],model.transform('MiDrug',pairs[:,::-1],'context'))
        np.testing.assert_array_equal(expected[0],expected[-1])
        with self.assertRaises(ValueError):
            model.transform('MiDrug',np.array([[-1],[0]]),'context')


class AcceptanceTests(unittest.TestCase):
    def rows(self):
        return [dict(task=t,seed=s,recipe=r,auc=.8+(.01 if t=='MiDrug' and r=='c' else 0),
                     ap=.7+(.01 if t=='MiDrug' and r=='c' else 0),macro_drug_ap=.5)
                for t in ('MiDrug','LncDrug') for s in (666,0,1) for r in ('baseline','c')]

    def test_no_seed_can_hide_a_lnc_drop_and_control_cannot_be_selected(self):
        rows=self.rows();config=dict(id='c',variant='context',alpha=.1)
        self.assertTrue(assess(rows,config,[666,0,1])['eligible'])
        rows[-1]['auc']-=1e-10
        rows[-3]['auc']+=.001
        self.assertFalse(assess(rows,config,[666,0,1])['eligible'])
        self.assertFalse(assess(self.rows(),dict(config,variant='context_permuted'),[666,0,1])['eligible'])

    def test_outer_gate_precedes_baseline_array_loading(self):
        with mock.patch.object(experiment,'selected',side_effect=ValueError('ineligible')):
            with mock.patch.object(experiment.graph,'protocol') as loader:
                with self.assertRaises(ValueError):
                    experiment.stage_data('/unused',{'tasks':['MiDrug']},'MiDrug','outer')
                loader.assert_not_called()


if __name__=='__main__':
    unittest.main()
