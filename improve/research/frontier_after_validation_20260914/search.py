"""Bounded public literature discovery after completed local ablations.

Read-only scientific API requests; retain original responses and retrieval
metadata. No training, application writes, private data or evaluation arrays.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parent
AS_OF = '2026-09-14'
QUERIES = {
    'direct_ncdrug': '(TITLE_ABS:"ncRNA-drug" OR TITLE_ABS:"miRNA-drug" OR TITLE_ABS:"lncRNA-drug") AND (TITLE_ABS:prediction OR TITLE_ABS:learning)',
    'multimodal_drug': 'TITLE_ABS:drug AND (TITLE_ABS:"missing modalities" OR TITLE_ABS:"multimodal pretraining" OR TITLE_ABS:"multimodal pre-training" OR TITLE_ABS:Madrigal)',
    'madrigal': 'Madrigal AND (TITLE_ABS:multimodal OR TITLE_ABS:representation OR TITLE_ABS:contrastive)',
    'signed_mechanism': '(TITLE_ABS:"signed graph" OR TITLE_ABS:"sign-aware" OR TITLE_ABS:"signed knowledge" OR TITLE_ABS:"directed signaling") AND (TITLE_ABS:drug OR TITLE_ABS:perturbation)',
    'pu_candidates': '(TITLE_ABS:"positive-unlabeled" OR TITLE_ABS:"positive unlabeled") AND (TITLE_ABS:drug OR TITLE_ABS:RNA OR TITLE_ABS:biological)',
    'source_robust': 'TITLE_ABS:"drug response" AND (TITLE_ABS:"domain generalization" OR TITLE_ABS:"distribution shift" OR TITLE_ABS:"label noise" OR TITLE_ABS:"invariant learning")',
    'rna_target_context': '(TITLE_ABS:microRNA OR TITLE_ABS:miRNA) AND (TITLE_ABS:"target repression" OR TITLE_ABS:"alternative polyadenylation" OR TITLE_ABS:"target site accessibility") AND (TITLE_ABS:predict OR TITLE_ABS:learning OR TITLE_ABS:cancer)',
    'rna_functional_local': '(TITLE_ABS:lncRNA OR TITLE_ABS:"RNA foundation") AND (TITLE_ABS:"localization prediction" OR TITLE_ABS:"functional representation" OR TITLE_ABS:"RNA-binding protein") AND (TITLE_ABS:model OR TITLE_ABS:learning)',
    'dependency_transfer': '(TITLE_ABS:"gene dependency" OR TITLE_ABS:"genetic dependency") AND (TITLE_ABS:"drug response" OR TITLE_ABS:"drug sensitivity") AND (TITLE_ABS:prediction OR TITLE_ABS:learning)',
    'perturbation_evaluation': 'TITLE_ABS:perturbation AND (TITLE_ABS:benchmark OR TITLE_ABS:generalization) AND (TITLE_ABS:"foundation model" OR TITLE_ABS:"linear model" OR TITLE_ABS:"cell types")',
}


def fetch(label, url):
    meta = dict(label=label, url=url, retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    req = urllib.request.Request(url, headers={'User-Agent': 'NcRNAEvidenceReview/1.1',
                                               'Accept': 'application/json,application/xml;q=0.9,*/*;q=0.1'})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read()
            meta.update(http_status=response.status, response_url=response.url, content_type=response.headers.get('Content-Type'))
        suffix = '.json' if raw.lstrip().startswith((b'{', b'[')) else '.xml'
        path = ROOT / (label + suffix)
        if path.exists():
            raise ValueError('Refuse to overwrite a prior source response')
        path.write_bytes(raw)
        meta.update(file=path.name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        if label.startswith('epmc_'):
            obj = json.loads(raw)
            meta.update(total_hits=obj.get('hitCount'), records=[dict(title=x.get('title'), doi=x.get('doi'),
                pmid=x.get('id') if x.get('source') == 'MED' else None, pmcid=x.get('pmcid'),
                source=x.get('source'), source_id=x.get('id'), date=x.get('firstPublicationDate'),
                authors=x.get('authorString'), journal=x.get('journalInfo', {}).get('journal', {}).get('title'),
                abstract=x.get('abstractText'), in_pmc=x.get('inPMC'), is_open_access=x.get('isOpenAccess'))
                for x in obj.get('resultList', {}).get('result', [])])
    except Exception as error:
        meta['error'] = str(error)
    return meta


def main():
    if (ROOT / 'search_manifest.json').exists():
        raise ValueError('Search already archived; use a new directory for a refresh')
    jobs = []
    for key, query in QUERIES.items():
        query = f'({query}) AND FIRST_PDATE:[2024-01-01 TO {AS_OF}]'
        params = dict(query=query, format='json', resultType='core', pageSize=30, sort_date='y')
        jobs.append(('epmc_' + key, 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode(params)))
    term = '("ncRNA-drug"[Title/Abstract] OR "miRNA-drug"[Title/Abstract] OR "lncRNA-drug"[Title/Abstract]) AND ("2025/01/01"[Date - Publication] : "2026/09/14"[Date - Publication])'
    jobs.append(('pubmed_direct', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?' + urllib.parse.urlencode(
        dict(db='pubmed', term=term, retmode='json', retmax=40, sort='pub_date'))))
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, *job) for job in jobs]):
            row = future.result()
            results.append(row)
            print(json.dumps({k: row.get(k) for k in ('label', 'http_status', 'total_hits', 'error')}, ensure_ascii=False), flush=True)
    # DOI first; normalize titles only as a fallback when DOI is absent.
    unique = {}
    for row in results:
        for record in row.get('records', []):
            key = (record.get('doi') or re.sub(r'[^\w]+', ' ', record['title'])).lower().strip()
            if key not in unique or (not unique[key].get('pmcid') and record.get('pmcid')):
                unique[key] = dict(record, discovery_query=row['label'])
    (ROOT / 'search_manifest.json').write_text(json.dumps(dict(as_of=AS_OF, queries=QUERIES,
        preflight=dict(pubmed='reachable', crossref='reachable', arxiv='10-second timeout'),
        limitation='Targeted bounded searches, not a systematic or exhaustive review; Europe PMC searches include indexed preprints.',
        results=results), ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'discovery_unique.json').write_text(json.dumps(list(unique.values()), ensure_ascii=False, indent=2) + '\n')
    print('DISCOVERY_UNIQUE', len(unique), 'FAILURES', sum('error' in r for r in results), flush=True)


if __name__ == '__main__':
    main()
