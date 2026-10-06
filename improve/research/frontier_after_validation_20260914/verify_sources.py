"""Resolve a selected reading set through Crossref and lawful PMC full texts."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import xml.etree.ElementTree as ET

from search import ROOT, fetch


SELECTED = {
    'm2umol': '10.1038/s41467-026-69302-6',
    'selformermm': '10.1093/bioinformatics/btag451',
    'fourierdrug': '10.1093/bioinformatics/btag276',
    'mirnapex': '10.1261/rna.080990.126',
    'lnc_rbp_occupancy': '10.3390/ijms27125593',
    'bin_pu': '10.1093/bioinformatics/btag067',
    'simple_controls': '10.1093/bioinformatics/btaf317',
    'slim': '10.64898/2026.08.07.743481',
    'mechanism_benchmark': '10.64898/2026.08.19.745729',
    'mdadgan': '10.1021/acs.jcim.6c00254',
    'rgdagmae': '10.1016/j.compbiolchem.2026.109236',
    'mphgnn': '10.1093/bib/bbag013',
    'mifndra': '10.1093/bib/bbag322',
    'mirbind2': '10.1093/bioinformatics/btag410',
    'parnet': '10.64898/2026.08.08.743506',
    'forge': '10.1038/s41467-026-73977-2',
    'dispa': '10.1093/bioinformatics/btag272',
    'mirbench': '10.1093/bioinformatics/btaf233',
    'pulsnar': '10.7717/peerj-cs.2451',
    'perturbation_benchmark': '10.1126/sciadv.aed3414',
}
FULL_TEXT = ('m2umol', 'selformermm', 'fourierdrug', 'mirnapex', 'lnc_rbp_occupancy', 'bin_pu', 'simple_controls')


def main():
    if (ROOT / 'verification_manifest.json').exists():
        raise ValueError('Already archived; do not overwrite')
    discovery = json.loads((ROOT / 'discovery_unique.json').read_text())
    by_doi = {(r.get('doi') or '').lower(): r for r in discovery if r.get('doi')}
    jobs = [('crossref_' + key, 'https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='/'))
            for key, doi in SELECTED.items()]
    for key in FULL_TEXT:
        row = by_doi[SELECTED[key]]
        if not row.get('pmcid'):
            raise ValueError('No verified PMC identifier for ' + key)
        jobs.append(('full_' + key, 'https://www.ebi.ac.uk/europepmc/webservices/rest/' + row['pmcid'] + '/fullTextXML'))
    ids = json.loads((ROOT / 'pubmed_direct.json').read_text())['esearchresult']['idlist']
    ids = sorted(set(ids) | {by_doi[d]['pmid'] for d in SELECTED.values() if d in by_doi and by_doi[d].get('pmid')})
    jobs.append(('pubmed_selected', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?' + urllib.parse.urlencode(
        dict(db='pubmed', id=','.join(ids), retmode='xml'))))
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, *job) for job in jobs]):
            row = future.result()
            if row.get('file'):
                path = ROOT / row['file']
                if row['label'].startswith('crossref_'):
                    metadata = json.loads(path.read_text())['message']
                    expected = SELECTED[row['label'].removeprefix('crossref_')]
                    if metadata['DOI'].lower() != expected.lower():
                        raise ValueError('Resolved DOI mismatch')
                    row['metadata'] = {name: metadata.get(name) for name in
                        ('DOI', 'title', 'author', 'container-title', 'type', 'published', 'published-online', 'posted', 'relation')}
                elif row['label'].startswith('full_'):
                    root = ET.fromstring(path.read_bytes())
                    actual_dois = [''.join(e.itertext()).lower() for e in root.findall('./front/article-meta/article-id') if e.attrib.get('pub-id-type') == 'doi']
                    expected = SELECTED[row['label'].removeprefix('full_')].lower()
                    if expected not in actual_dois:
                        raise ValueError('Wrong article full text: ' + expected)
                    lines = []
                    for section in (root.find('./front/article-meta'), root.find('./body')):
                        if section is not None:
                            for element in section.iter():
                                if element.tag in ('article-title', 'title', 'p', 'caption'):
                                    value = ' '.join(''.join(element.itertext()).split())
                                    if value:
                                        lines.append(value)
                    text_path = path.with_suffix('.txt')
                    text_path.write_text('\n'.join(f'{i+1:04d}: {line}' for i, line in enumerate(lines)) + '\n')
                    row.update(readable_file=text_path.name, doi_verified=True, status='open_access_full_text_xml',
                               si_status='not_downloaded', paragraph_count=len(lines))
            records.append(row)
            print(json.dumps({k: row.get(k) for k in ('label', 'http_status', 'doi_verified', 'paragraph_count', 'error')}, ensure_ascii=False), flush=True)
    (ROOT / 'verification_manifest.json').write_text(json.dumps(dict(as_of='2026-09-14',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), selected=SELECTED, results=records), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
