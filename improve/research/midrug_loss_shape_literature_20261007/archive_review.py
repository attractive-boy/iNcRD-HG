"""Deduplicate all six public requests and archive the bounded primary readings."""
import datetime,json,hashlib
from pathlib import Path
from search import norm_title,surname
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    requests=json.loads((HERE/'search_results.json').read_text())+json.loads((HERE/'focused_search.json').read_text())
    assert len(requests)==6 and all(r['status']=='ok' for r in requests)
    merged=[]
    for r in requests:
        for p in r['papers']:
            found=None;doi=(p.get('doi') or '').lower();arxiv=(p.get('id') or '').rsplit('v',1)[0]
            for old in merged:
                od=(old.get('doi') or '').lower();oa=(old.get('id') or '').rsplit('v',1)[0]
                same=bool(doi and od and doi==od) or bool(arxiv and oa and arxiv==oa)
                if not same and not (doi and od):
                    a,b=set(norm_title(p['title']).split()),set(norm_title(old['title']).split())
                    same=bool(surname(p) and surname(p)==surname(old) and len(a&b)/max(1,len(a|b))>=.9)
                if same:found=old;break
            if found is None:merged.append(dict(p,discovery_queries=[r['query']]))
            else:found['discovery_queries'].append(r['query'])
    (HERE/'combined_deduplicated.json').write_text(json.dumps(merged,indent=2,ensure_ascii=False))
    dependencies=[HERE.parent/'midrug_literature_20261006/expression_followup_search.py',
        HERE.parent/'midrug_conjugate_blocks_20261007/archive_manifest.json',HERE.parent/'midrug_completion_20261006/REPORT.md',
        HERE.parent/'midrug_observed_context_20261006/REPORT.md',HERE.parent/'midrug_source_observed_20261006/REPORT.md',
        HERE.parent/'gradient_selection_next_20260916/STORM_ADAPTATION_V1.md',HERE.parent/'topology_selection_next_20260916/PULL_ADAPTATION_V1.md']
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['review_manifest.json','review_recheck.json','archive.log']]
    archive=dict(status='bounded_review_complete',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        raw_records=sum(len(r['papers']) for r in requests),deduplicated=len(merged),primary_documents_partially_read=3,
        no_new_training_data=True,artifact_hashes={str(p.relative_to(HERE)):digest(p) for p in sorted(files)},dependencies={str(p):digest(p) for p in dependencies})
    (HERE/'review_manifest.json').write_text(json.dumps(archive,indent=2));count=0
    for p,h in archive['artifact_hashes'].items():assert digest(HERE/p)==h;count+=1
    for p,h in archive['dependencies'].items():assert digest(p)==h;count+=1
    result=dict(status='passed',hash_checks=count,review_sha256=digest(HERE/'review_manifest.json'),raw_records=archive['raw_records'],deduplicated=len(merged))
    (HERE/'review_recheck.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
