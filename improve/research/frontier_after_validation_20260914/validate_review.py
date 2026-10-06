"""Verify this review's provenance, arithmetic and authored document links."""
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    output = ROOT / 'COMPLETION.json'
    if output.exists():
        raise ValueError('Completion already archived; do not overwrite')
    bibliography = read('references.json')
    references = bibliography['references']
    assert len(references) == len({r['doi'].lower() for r in references}) == 20
    text_dois, snapshots = set(), 0
    for row in references:
        source = row['sources']
        cr_path = ROOT / source['crossref']['file']
        assert digest(cr_path) == source['crossref']['sha256']
        cr = json.loads(cr_path.read_text())['message']
        assert cr['DOI'].lower() == row['doi'].lower()
        text_dois.add(row['doi'].lower())
        assert digest(ROOT / source['reading_file']) == source['reading_file_sha256']
        if 'full_text_raw' in source:
            raw = ROOT / source['full_text_raw']
            assert digest(raw) == source['full_text_sha256']
            ids = [e.text.lower() for e in ET.parse(raw).getroot().findall('./front/article-meta/article-id')
                   if e.attrib.get('pub-id-type') == 'doi' and e.text]
            assert row['doi'].lower() in ids
        # At least one dated public record is on/before the search cutoff.
        candidate_dates = []
        dates = row['publication_dates']
        for key in ['crossref_online', 'crossref_posted', 'crossref_published']:
            if dates.get(key):
                parts = dates[key]['date-parts'][0]
                candidate_dates.append(tuple(parts))
        if dates.get('epmc_first_publication'):
            candidate_dates.append(tuple(map(int, dates['epmc_first_publication'].split('-'))))
        for d in dates.get('pubmed_article_dates', []):
            if d.get('type') == 'Electronic':
                candidate_dates.append(tuple(int(d[k]) for k in ['year', 'month', 'day']))
        assert any(d[0] < 2026 or (len(d) >= 2 and d[:2] < (2026, 9)) or
                   (len(d) == 3 and d <= (2026, 9, 14)) for d in candidate_dates), row['id']
        snapshots += 1
    assert bibliography['full_text_relevant_sections'] == 14
    assert bibliography['abstract_only_count'] == 6
    assert sum(r['publication_type'] == 'preprint' for r in references) == 3
    ris = (ROOT / 'references.ris').read_text()
    assert len(re.findall(r'^TY  - ', ris, re.M)) == 20
    assert len(re.findall(r'^ER  -', ris, re.M)) == 20
    assert {x.lower() for x in re.findall(r'^DO  - (.+)$', ris, re.M)} == text_dois
    local_links, doi_links = [], []
    for name in ['REPORT.md', 'READING_NOTES.md']:
        text = (ROOT / name).read_text()
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if target.startswith('https://doi.org/'):
                doi = target.removeprefix('https://doi.org/').lower()
                assert doi in text_dois, (name, doi)
                doi_links.append(target)
            elif not re.match(r'https?://|mailto:', target):
                path = (ROOT / target.split('#')[0]).resolve()
                assert path.exists(), (name, target)
                local_links.append(str(path))
    verified_inputs = set()
    for name in ['LOCAL_INFORMATION_AUDIT.json', 'ACTION_SCOPE_AUDIT.json']:
        audit = read(name)
        for item in audit['inputs']:
            path = Path(item['path'])
            assert digest(path) == item['sha256'], str(path)
            verified_inputs.add(str(path))
    info = read('LOCAL_INFORMATION_AUDIT.json')
    groups = info['site_windows']
    assert groups['all_cached_windows']['windows'] == 6975
    assert groups['source_filtered_windows']['windows'] == 6936
    assert groups['all_cached_windows']['windows'] - groups['source_filtered_windows']['windows'] == 39
    for g in groups.values():
        assert sum(g['support_classes'].values()) == g['windows']
        assert sum(g['raw_support_counts'].values()) == g['windows']
        for subset in g['coverage'].values():
            for coverage in subset.values():
                assert coverage['possible_pairs'] == 605 * 216
                assert abs(coverage['pair_fraction'] - coverage['pairs_with_at_least_one_exact_gene_match'] / 130680) < 1e-15
    assert info['source_filter']['excluded_sites_also_nonfunctional_only'] == 0
    assert sum(info['drug_actions']['actions_by_row'].values()) == 654
    assert sum(info['drug_actions']['target_types_by_row'].values()) == 654
    for name, item in info['outputs'].items():
        assert digest(ROOT / name) == item['sha256']
    readiness = read('DATA_READINESS_AUDIT.json')
    for item in readiness['feature_caches']:
        assert digest(Path(item['path'])) == item['sha256']
        verified_inputs.add(item['path'])
    assert digest(ROOT / 'audit_data_readiness.py') == readiness['script_sha256']
    for manifest_name in ['search_manifest.json', 'verification_manifest.json']:
        for record in read(manifest_name)['results']:
            if record.get('file'):
                assert digest(ROOT / record['file']) == record['sha256']
    for manifest_name in ['verification_retry_manifest.json', 'code_availability_manifest.json']:
        for record in read(manifest_name):
            if record.get('file'):
                assert digest(ROOT / record['file']) == record['sha256']
            if record.get('decoded_file'):
                assert digest(ROOT / record['decoded_file']) == record['decoded_sha256']
    inventory = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(ROOT.iterdir()) if p.is_file() and p != output}
    result = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(), as_of='2026-09-14',
        task='Literature reassessment and label-free feasibility audits after completed local experiments.',
        checks=dict(selected_dois_verified=snapshots, relevant_full_text_reading_records=14,
            abstract_only_reading_records=6, preprints_marked=3, ris_entries=20,
            authored_document_local_links_validated=len(local_links), authored_document_doi_links_validated=len(doi_links),
            unique_local_inputs_with_verified_hashes=len(verified_inputs), raw_download_hashes_verified=True,
            coverage_arithmetic_verified=True, date_cutoff_verified=True),
        work_boundary=dict(new_models_trained=False, new_prediction_scores_evaluated=False,
            historical_outer_arrays_read=False, original_association_test_arrays_read=False,
            source_holdout_outcome_labels_read=False, accepted_model_changed=False,
            auxiliary_pmid_metadata_used=True), files=inventory)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
