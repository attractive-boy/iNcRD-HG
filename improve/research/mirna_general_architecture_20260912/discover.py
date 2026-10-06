"""Bounded primary-source search for a general RNA/drug architecture.

Only public keywords are sent. Neither project observations nor private files
are uploaded. Results are discovery records, not automatically accepted claims.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from search_expression_consensus import arxiv, crossref, pubmed


DATES = ' AND ("2025/01/01"[Date - Publication] : "2026/09/12"[Date - Publication])'
ARXIV_DATES = ' AND submittedDate:[202501010000 TO 202609122359]'
JOBS = [
    ('PubMed', '(microRNA[tiab] OR miRNA[tiab]) AND ("deep learning"[tiab] OR "foundation model"[tiab] OR transformer[tiab]) AND (target[tiab] OR binding[tiab])'+DATES),
    ('PubMed', 'RNA[tiab] AND ("foundation model"[tiab] OR "language model"[tiab]) AND (motif[tiab] OR function[tiab] OR generalization[tiab])'+DATES),
    ('PubMed', '(microRNA[tiab] OR miRNA[tiab]) AND (Argonaute[tiab] OR seed[tiab] OR supplementary[tiab]) AND (prediction[tiab] OR "binding specificity"[tiab] OR "high-throughput"[tiab])'+DATES),
    ('PubMed', '("drug resistance"[tiab] OR "drug response"[tiab]) AND (ncRNA[tiab] OR miRNA[tiab] OR lncRNA[tiab]) AND (neural[tiab] OR prediction[tiab])'+DATES),
    ('PubMed', 'RNA[tiab] AND ("optimal transport"[tiab] OR "late interaction"[tiab] OR "multiple instance learning"[tiab])'+DATES),
    ('PubMed', '(perturbation[tiab] OR "regulatory program"[tiab]) AND (compositional[tiab] OR generalization[tiab]) AND (model[tiab] OR learning[tiab])'+DATES),
    ('CrossRef', 'miRBind2'),
    ('CrossRef', 'microRNA target prediction foundation models 2026'),
    ('CrossRef', 'RNA language model functional motifs 2025 2026'),
    ('arXiv', '(all:"RNA" OR all:"microRNA") AND (all:"language model" OR all:"foundation model")'+ARXIV_DATES),
    ('arXiv', 'all:"optimal transport" AND (all:"molecular" OR all:"biological" OR all:"compositional")'+ARXIV_DATES),
    ('arXiv', '(all:"late interaction" OR all:"ColBERT") AND (all:"molecular" OR all:"biological" OR all:"multimodal")'+ARXIV_DATES),
    ('arXiv', 'all:"graph" AND all:"in-context" AND (all:"structural" OR all:"relational")'+ARXIV_DATES),
]
LOCKS = {name:threading.Lock() for name in ('PubMed', 'arXiv')}


def query(source, text):
    fn = {'PubMed':pubmed, 'CrossRef':crossref, 'arXiv':arxiv}[source]
    if source in LOCKS:
        with LOCKS[source]:
            time.sleep(.4 if source == 'PubMed' else 3.)
            return fn(text)
    return fn(text)


def normalized_title(title):
    stop = {'a','an','the','in','of','for','on','to','and','with','by','et','al'}
    return set(re.sub(r'[^\w\s]', ' ', title.lower()).split())-stop


def deduplicate(responses):
    merged = []
    for response in responses:
        for record in response['records']:
            record['doi'] = (record.get('doi') or '').lower().removeprefix('https://doi.org/').strip()
            tokens = normalized_title(record['title'])
            first = (record.get('authors') or [''])[0].lower().split()
            surname = first[-1] if first else ''
            match = None
            for entry in merged:
                old = entry['record']
                other = normalized_title(old['title'])
                old_first = (old.get('authors') or [''])[0].lower().split()
                old_surname = old_first[-1] if old_first else ''
                same_doi = record['doi'] and record['doi'] == old.get('doi')
                fuzzy = surname == old_surname and bool(surname) and len(tokens & other)/max(1,len(tokens | other)) >= .9
                if same_doi or fuzzy:
                    match = entry
                    break
            if match is None:
                merged.append(dict(record=record, sources=[response['source']], queries=[response['query']]))
            else:
                match['sources'] = sorted(set(match['sources']+[response['source']]))
                match['queries'] = sorted(set(match['queries']+[response['query']]))
                if (bool(record['doi']), len(record.get('abstract',''))) > (bool(match['record'].get('doi')), len(match['record'].get('abstract',''))):
                    match['record'] = record
    return merged


def main():
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(query, source, text):(source,text) for source,text in JOBS}
        for future in as_completed(futures):
            source,text = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                value = dict(source=source,query=text,records=[],error=str(exc))
            results.append(value)
            print(source, len(value['records']), value.get('error',''), text, flush=True)
    merged = deduplicate(results)
    payload = dict(retrieved_at_utc=datetime.now(timezone.utc).isoformat(), mcp_available=False,
        scope='Bounded discovery; date window 2025-01-01 to 2026-09-12 except exact/metadata CrossRef queries. Verify selected dates before citation.',
        responses=results, deduplicated=merged)
    Path(__file__).with_name('primary_discovery.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    for item in merged:
        r=item['record']
        print(r['source'],r.get('year'),r['title'],r.get('doi') or r['url'],flush=True)


if __name__ == '__main__':
    main()
