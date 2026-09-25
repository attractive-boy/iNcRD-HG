"""Guard the record/source boundary and avoid global label consensus leakage."""
import json
import unittest
import pandas as pd
from improve.context_source_eval import (source_components, assign_groups, aggregate,
                                        add_novelty, identity_set, semantic_flag, pmids)


def row(record, pmid, disease='A', effect='resistant', rna='r'):
    return dict(record_uid=record, pmid=pmid, source_group='pmid:'+pmid.split(';')[0],
                ncRNA=rna, cid='1', disease=disease, effect=effect,
                throughput='high', semantic_status='unverified')


class SourceTests(unittest.TestCase):
    def test_shared_and_multi_pmid_records_are_indivisible(self):
        records = [row('a','12;13'), row('b','13'), row('c','14'), row('d','')]
        groups = source_components(records)
        self.assertEqual(groups['a'],groups['b'])
        self.assertNotEqual(groups['b'],groups['c'])
        self.assertEqual(groups['d'],'')
        self.assertEqual(pmids('12;12, 13'), ['12','13'])
        with self.assertRaises(ValueError):
            pmids('12e6')

    def test_group_assignment_reproducible_and_label_free(self):
        sizes = {'p'+str(i):i+1 for i in range(24)}
        self.assertEqual(assign_groups(sizes),assign_groups(dict(reversed(list(sizes.items())))))
        self.assertEqual(set(assign_groups(sizes).values()),{'train','validation','holdout'})

    def test_expansion_does_not_duplicate_an_original_observation(self):
        frame = pd.DataFrame([row('a','12'),row('a','12'),row('a','12','B')])
        clean,_ = aggregate(frame)
        self.assertEqual(len(clean),2)
        self.assertEqual(clean.n_original_records.tolist(),[1,1])
        self.assertEqual(json.loads(clean.pmids.iloc[0]),['12'])

    def test_heldout_conflict_never_changes_training_consensus(self):
        train = pd.DataFrame([row('a','12')])
        heldout = pd.DataFrame([row('b','13',effect='sensitive')])
        train_clean,train_conflicts = aggregate(train)
        heldout_clean,_ = aggregate(heldout)
        self.assertEqual(train_clean.y.tolist(),[1])
        self.assertEqual(heldout_clean.y.tolist(),[0])
        self.assertEqual(len(train_conflicts),0)
        self.assertFalse(add_novelty(heldout_clean,identity_set(train)).novel_triple.iloc[0])
        _,global_conflicts = aggregate(pd.concat([train,heldout]))
        self.assertEqual(len(global_conflicts),1)

    def test_conflicted_training_identity_is_also_purged(self):
        train = pd.DataFrame([row('a','12'),row('b','12',effect='sensitive')])
        heldout,_ = aggregate(pd.DataFrame([row('c','13')]))
        self.assertEqual(len(aggregate(train)[0]),0)
        self.assertFalse(add_novelty(heldout,identity_set(train)).novel_triple.iloc[0])

    def test_semantic_flags_do_not_claim_intervention(self):
        r = dict(support_text='miRNA overexpression increases drug sensitivity',pmid='12',
                 detection_method='qRT-PCR',throughput='low')
        self.assertEqual(semantic_flag(r),'intervention_candidate_unverified')
        r.update(pmid='24145123',detection_method='Microarray')
        self.assertEqual(semantic_flag(r),'known_expression_comparison_24145123')


if __name__ == '__main__':
    unittest.main()
