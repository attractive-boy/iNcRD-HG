import tempfile
from pathlib import Path
import unittest
from unittest import mock

from improve import functional_evidence_data as data
from improve import tabular_foundation as base


class BindingEvidenceTests(unittest.TestCase):
    def valid(self):
        return dict(assay_type='B',target_tax_id=9606,standard_type='Ki',standard_units='nM',
                    standard_relation='=',standard_value='1000',standard_flag=1,potential_duplicate=0,
                    data_validity_comment=None,assay_variant_mutation=None)

    def test_activity_filters_do_not_treat_other_readouts_as_binding(self):
        row=self.valid()
        self.assertTrue(data.valid_activity(row))
        for field,value in (('assay_type','F'),('target_tax_id',10090),('standard_relation','<'),
                            ('standard_units','uM'),('standard_value','1001'),('standard_value','nan'),
                            ('standard_value','0'),('standard_type','EC50'),('standard_flag',0),
                            ('potential_duplicate',1),('data_validity_comment','Potential transcription error'),
                            ('assay_variant_mutation','T315I')):
            with self.subTest(field=field,value=value):
                self.assertFalse(data.valid_activity(dict(row,**{field:value})))

    def test_protein_mapping_rejects_ambiguity_complexes_and_nonhuman(self):
        row=dict(target_type='SINGLE PROTEIN',tax_id=9606,target_components=[dict(accession='P1')])
        self.assertEqual(data.resolve_target(row,{'P1':{'G1'}}),('G1',''))
        for changed,mapping in ((row,{'P1':{'G1','G2'}}),(row,{}),
                                (dict(row,target_type='PROTEIN COMPLEX'),{'P1':{'G1'}}),
                                (dict(row,tax_id=10090),{'P1':{'G1'}}),
                                (dict(row,target_components=[dict(accession='P1'),dict(accession='P2')]),{'P1':{'G1'}})):
            self.assertEqual(data.resolve_target(changed,mapping)[0],'')

    def test_partial_or_changing_pagination_cannot_be_used(self):
        for responses in ([dict(page_meta=dict(total_count=2,next=None),activities=[{}])],
                          [dict(page_meta=dict(total_count=2,next='url'),activities=[{}]),
                           dict(page_meta=dict(total_count=3,next=None),activities=[{}])]):
            with mock.patch.object(data,'read_or_fetch',side_effect=responses):
                with self.assertRaises(ValueError):
                    data.pages(Path('/unused'),'sample','activity',{},'activities')
        responses=[dict(page_meta=dict(total_count=2,next='url'),activities=[{'id':1}]),
                   dict(page_meta=dict(total_count=2,next=None),activities=[{'id':2}])]
        with mock.patch.object(data,'read_or_fetch',side_effect=responses) as fetch:
            self.assertEqual(len(data.pages(Path('/unused'),'sample','activity',{},'activities')),2)
            self.assertIn('offset=1',fetch.call_args.args[-1])

    def test_tampered_cache_and_query_are_rejected_before_build(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'sources').mkdir()
            base.write_json(out/'sources/a.json',dict(value=1))
            base.write_json(out/'sources/a.meta.json',dict(url='expected',sha256='wrong'))
            with self.assertRaises(ValueError):
                data.read_or_fetch(out,'a','expected')
            base.write_json(out/'retrieved.json',{})
            base.write_json(out/'QUERY_SPEC.json',{})
            base.write_json(out/'DOWNLOAD_COMPLETE.json',dict(complete=True,
                retrieved_sha256=base.digest(out/'retrieved.json'),query_spec_sha256='wrong'))
            with self.assertRaises(ValueError):
                data.verify_snapshot(out,Path('/unused'),Path('/unused'))


if __name__=='__main__':
    unittest.main()
