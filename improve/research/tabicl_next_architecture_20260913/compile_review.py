"""Compile local retrieval evidence and explicit paper assessments; no network."""
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STOP = set('a an the in of for on to and with by et al'.split())


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


ASSESSMENTS = {
    'beta': {
        'publication_status': 'ICML 2025; verified against the PMLR proceedings',
        'primary_url': 'https://proceedings.mlr.press/v267/liu25cn.html',
        'read_scope': ['Accepted PDF: Section 3, Appendix B checkpoint, C.2, D.3', 'arXiv method and discussion', 'Official encoder and training forward source'],
        'priority': 'First: adapt the input-encoder mechanism to existing TabICLv2',
        'evidence': 'A shared input encoder is optimized while the entire TabPFN is frozen; the accepted Appendix D.3 ranks full TabPFNv2 and TabICL above original BETA.',
        'limitation': 'Experiments primarily use TabPFNv1 and accuracy. Neither superiority over TabICLv2 nor LncDrug AUC/AP protection is demonstrated.',
        'code_url': 'https://github.com/LAMDA-Tabular/BETA',
    },
    'tunetables': {
        'publication_status': 'NeurIPS 2024; exact Crossref DOI 10.52202/079017-2654 verified',
        'read_scope': ['Sections 3–4', 'Appendix F small-prompt excerpt', 'Official README'],
        'priority': 'Second: support-conditioned shared prototype generator',
        'evidence': 'Learned context embeddings can be used together with or instead of real context, and can optimize additional objectives.',
        'limitation': 'The paper does not establish that a dataset-specific prompt transfers across the two RNA tasks or protects both ranking metrics.',
        'code_url': 'https://github.com/penfever/TuneTables',
    },
    'gem': {
        'publication_status': 'NIPS 2017, as stated in the official arXiv record',
        'read_scope': ['Section 3, especially equations 6–8'],
        'priority': 'Protection mechanism supporting the new adapters',
        'evidence': 'Project an update onto half-spaces defined by gradients of retained training-memory losses.',
        'limitation': 'Local first-order constraints and representative finite memory are assumed; this is not an outer AUC/AP guarantee.',
        'code_url': 'https://github.com/facebookresearch/GradientEpisodicMemory',
    },
    'igem': {
        'publication_status': '2026 preprint',
        'read_scope': ['Sections II–III, adapter-space constraints and dual solver', 'AG News evaluation description and limitations'],
        'priority': 'Engineering precedent; use a simpler verifiable projection for few constraints',
        'evidence': 'Approximate GEM projection in the LoRA subspace, evaluated on a three-task GPT-2/AG News setup.',
        'limitation': 'The iterative solver is approximate and the evidence does not cover tabular AUC/AP or ncRNA–drug prediction.',
    },
    'puicl': {
        'publication_status': '2026 preprint',
        'read_scope': ['SCAR definition', 'Synthetic PU construction', 'Unlabeled token and unmasked row attention', 'Evaluation setup and limitations', 'Official model and inference source'],
        'priority': 'Longer-term observation-aware PU model, not a direct replacement',
        'evidence': 'Synthetic PU pretraining learns a distinct unlabeled token and joint inference over labeled positives and an unlabeled pool.',
        'limitation': 'SCAR, transductive query dependence, paper feature range up to 20 versus repository range 5–24, and unknown biological class prior.',
        'code_url': 'https://github.com/qinglong-tian/puicl',
        'code_note': 'Bundled latest.pt is listed in the repository tree (14,868,837 bytes), but was not downloaded or run. score_unlabeled returns negative/outlier probability.',
    },
    'knowstfm': {
        'publication_status': '2026 preprint',
        'read_scope': ['Sections 3.3–4', 'General-domain Table 3 and discussion', 'Limitations', 'Appendix E implementation description'],
        'priority': 'Counter-evidence against repeating knowledge-attention fine-tuning',
        'evidence': 'Knowledge graph feature-attention priors plus LoRA help some nano models; reported general-domain TabICLv2 balanced accuracy declines from 0.763 to 0.749.',
        'limitation': 'Feature columns must map meaningfully to the knowledge graph; current graph-statistic columns are not gene-expression columns.',
        'code_url': 'https://github.com/bkolosk1/knowstfm',
        'code_note': 'Author-linked repository search and minimal README checked; implementation was not audited or run.',
    },
    'crumb': {
        'publication_status': '2026 preprint',
        'read_scope': ['Section 4 and Algorithm 1', 'Section 6 and limitations'],
        'priority': 'Optional context diversity mechanism; not the primary improvement route',
        'evidence': 'Query clustering and MMD retrieval trade off relevance and redundancy at a fixed context budget.',
        'limitation': 'Needs the full query batch; the paper notes cached-centroid streaming may diminish distribution-matching advantages. No official repository was located in the performed search.',
    },
    'kernelicl': {
        'publication_status': '2026 preprint',
        'read_scope': ['Sections 3.2–3.4', 'Sections 4.2–4.3', 'Appendix A training details'],
        'priority': 'Interpretability and diagnostic alternative',
        'evidence': 'Reprocess support rows as queries to obtain symmetric embeddings and use an explicit kernel label readout, with synthetic-task fine-tuning.',
        'limitation': 'Competitive average accuracy is not strict non-degradation; additional fine-tuning and calibration overhead are required. No official repository was located in the performed search.',
    },
    'skilltfm': {
        'publication_status': '2026 preprint',
        'read_scope': ['Methodology including equations 2–11', 'Experimental setup'],
        'priority': 'Experiment-management precedent, not a new prediction backbone',
        'evidence': 'Validate external repairs, use deployment evidence for activation, and retain fallback and failed-route records.',
        'limitation': 'Tolerance and harm-rate criteria differ from the user’s strict two-metric rule; controlled boundary tasks and a limited electricity setting do not establish ncRNA–drug improvement.',
    },
}


def doi_normal(value):
    return re.sub(r'^https?://(?:dx\.)?doi.org/', '', value or '').strip().lower()


def words(value):
    return set(re.findall(r'\w+', html.unescape(value or '').lower())) - STOP


def surname(authors):
    first = (authors or [''])[0].strip()
    return (first.split(',')[0] if ',' in first else first.split(' ')[-1]).lower()


def main():
    queries, records, excluded_future = [], [], []
    for path in sorted(ROOT.glob('*.json')):
        if not path.name.startswith(('openalex_', 'arxiv_', 'pubmed_', 'crossref_')):
            continue
        result = json.loads(path.read_text())
        if not isinstance(result, dict) or 'query' not in result:
            continue
        queries.append({'file': path.name, 'source': result.get('source', path.name.split('_')[0]),
                        'query': result['query'], 'ok': result.get('ok'), 'error': result.get('error')})
        values = result.get('records', [])
        if path.name.startswith('crossref_'):
            values = result.get('response', {}).get('message', {}).get('items', [])
        for raw in values:
            title = raw.get('title', '')
            if isinstance(title, list):
                title = title[0] if title else ''
            authors = raw.get('authors') or [' '.join([a.get('given', ''), a.get('family', '')]).strip()
                                            for a in raw.get('author', [])]
            doi = raw.get('doi') or raw.get('DOI') or raw.get('identifiers', {}).get('doi')
            date = raw.get('publication_date') or raw.get('published', '')
            if isinstance(date, dict):
                parts = date.get('date-parts', [[]])[0]
                date = '-'.join(str(x).zfill(2) for x in parts)
            record = {'title': html.unescape(title), 'authors': authors, 'doi': doi_normal(doi),
                      'publication_date': date, 'source_files': [path.name],
                      'pmid': raw.get('pmid'), 'abstract': raw.get('abstract', '')}
            if date and len(date) >= 10 and date[:10] > '2026-09-13':
                excluded_future.append(record)
            else:
                records.append(record)
    merged = []
    for row in records:
        duplicate = None
        for existing in merged:
            if row['doi'] and row['doi'] in existing['doi_aliases']:
                duplicate = existing
                break
            left, right = words(row['title']), words(existing['title'])
            similarity = len(left & right) / max(len(left | right), 1)
            if surname(row['authors']) and surname(row['authors']) == surname(existing['authors']) and similarity >= .90:
                duplicate = existing
                break
        if duplicate is None:
            merged.append({**row, 'doi_aliases': [row['doi']] if row['doi'] else []})
        else:
            duplicate['source_files'] = sorted(set(duplicate['source_files'] + row['source_files']))
            if row['doi'] and row['doi'] not in duplicate['doi_aliases']:
                duplicate['doi_aliases'].append(row['doi'])
            if len(row['abstract']) > len(duplicate['abstract']):
                duplicate['abstract'] = row['abstract']
    write('deduplicated_discovery.json', merged)
    selected = []
    for slug, assessment in ASSESSMENTS.items():
        manifest = read(f'{slug}_fulltext_manifest.json')
        assert manifest.get('title_verified') and manifest['status'] in ('full_text_html_available', 'open_access_downloaded')
        meta = manifest['metadata']
        source = {**assessment, 'slug': slug, 'title': meta['citation_title'][0],
                  'authors': meta['citation_author'], 'first_posted': meta['citation_date'][0],
                  'arxiv_id': manifest['arxiv_id'],
                  'primary_url': assessment.get('primary_url', f"https://arxiv.org/abs/{manifest['arxiv_id']}"),
                  'read_depth': 'Selected full-text methods, results and/or limitations; not an independent reproduction',
                  'fulltext_file': manifest['fulltext_path'], 'fulltext_manifest': f'{slug}_fulltext_manifest.json'}
        if slug == 'beta':
            source['accepted_pdf_manifest'] = 'beta_icml_manifest.json'
            source['accepted_pdf_file'] = 'PDFs/Liu_2025_ICML_Beta.pdf'
        selected.append(source)
    write('selected_sources.json', selected)
    probe = read('input_adapter_probe.json')
    assert hashlib.sha256((ROOT/'probe_input_adapter.py').read_bytes()).hexdigest() == probe['probe_source_sha256']
    assert probe['probe_passed'] and probe['backbone_unchanged']
    summary = {
        'cutoff_date': '2026-09-13', 'systematic_review': False,
        'discovery_queries': queries, 'successful_discovery_queries': sum(bool(q['ok']) for q in queries),
        'failed_discovery_queries': sum(not q['ok'] for q in queries),
        'retrieved_discovery_records_before_date_filter': len(records) + len(excluded_future),
        'date_eligible_records_before_dedup': len(records), 'deduplicated_discovery_records': len(merged),
        'excluded_future_records': excluded_future,
        'fulltext_papers_with_selected_sections_reviewed': len(selected),
        'accepted_version_pdfs_downloaded': 1,
        'additional_metadata_and_abstract_sources': [
            {'name': 'StablePFN', 'doi': '10.1145/3770855.3818030', 'records': ['exact_crossref.json', 'stablepfn_openalex_detail.json']},
            {'name': 'MPHGNN', 'pmid': '41608986', 'doi': '10.1093/bib/bbag013', 'record': 'pubmed_mirna.json'},
            {'name': 'RGDAGMAE', 'pmid': '42470826', 'doi': '10.1016/j.compbiolchem.2026.109236', 'record': 'pubmed_mirna.json'},
            {'name': 'MVGSCA', 'pmid': '42678847', 'doi': '10.1109/TCBBIO.2026.3729865', 'record': 'pubmed_mirna.json'},
        ],
        'reference_resolution_corrections': [
            {'candidate_id': '2406.03441', 'status': 'Excluded: title mismatch; it is Cycles of Thought, not TuneTables',
             'correct_tunetables_id': '2402.11137', 'evidence_file': 'primary_abs_fetch.json'},
        ],
        'local_checks': {'fitting_only_input_probes': 2, 'adapter_parameters': 496,
                         'identity_errors': {task: row['identity_max_logit_error'] for task, row in probe['tasks'].items()},
                         'backbone_unchanged': True, 'optimizer_steps': 0,
                         'new_validation_scores': False, 'test_arrays_used': False},
        'active_reference': 'Existing graph-context TabICLv2',
        'user_confirmed_lnc_hard_metrics': ['auc', 'ap'], 'macro_drug_ap': 'report separately',
        'status': 'Literature-backed proposals plus an interface feasibility probe; no newly accepted predictor.',
        'limitations': ['Broad discovery contains many irrelevant results; counts are not counts of relevant papers.',
                        'OpenAlex discovery abstracts are truncated by the skill script; recommendations use primary full texts.',
                        'Historical development splits are reused; no independent generalization evidence was generated.'],
    }
    write('DISCOVERY_SUMMARY.json', summary)
    print(json.dumps({key: summary[key] for key in ('successful_discovery_queries', 'failed_discovery_queries',
                      'deduplicated_discovery_records', 'fulltext_papers_with_selected_sections_reviewed', 'local_checks')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
