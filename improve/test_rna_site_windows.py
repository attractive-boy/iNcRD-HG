import unittest
from improve.rna_site_windows import exact_window,fasta_records


class WindowTests(unittest.TestCase):
    def test_fasta_keeps_versions_and_rejects_error_pages_or_duplicates(self):
        self.assertEqual(fasta_records('>NM_1.2 description\nacg\ntta\n>NM_2.1\nCCAA\n'),
                         {'NM_1.2':'ACGTTA','NM_2.1':'CCAA'})
        for text in ('<error>not found</error>','>NM_1\nACGT\n>NM_1\nACGT\n','>NM_1\n'):
            with self.assertRaises(ValueError):
                fasta_records(text)

    def test_real_flanks_and_zero_based_coordinates(self):
        site='CGTACGTACGTA'
        sequence='A'*30+site+'C'*30
        window,reason=exact_window(sequence,site)
        self.assertEqual(reason,'')
        self.assertEqual(window['window_start_0'],11)
        self.assertEqual(window['site_start_0'],30)
        self.assertEqual(window['target_sequence'],sequence[11:61])
        self.assertEqual(len(window['target_sequence']),50)

    def test_ambiguous_sites_and_missing_flanks_are_not_padded(self):
        site='CGTACGTACGTA'
        for sequence,reason in ((site+'A'*60,'insufficient_real_flanks'),
            ('A'*30+site+'C'*30+site+'G'*30,'nonunique_site'),
            ('A'*100,'not_found_in_forward_transcript'),
            ('A'*29+'N'+site+'C'*30,'ambiguous_flank')):
            self.assertEqual(exact_window(sequence,site)[1],reason)


if __name__=='__main__':
    unittest.main()
