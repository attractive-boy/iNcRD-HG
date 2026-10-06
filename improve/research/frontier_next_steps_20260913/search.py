"""Public-source literature refresh for the post-functional-pilot project state.

No training or changes to existing experimental artifacts. Dates are capped at
2026-09-13; raw responses, errors, and metadata are retained for review.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
AS_OF = "2026-09-13"
QUERIES = {
    "ncdrug_recent": '(TITLE_ABS:"ncRNA-drug" OR TITLE_ABS:"miRNA-drug" OR TITLE_ABS:"lncRNA-drug") AND (TITLE_ABS:prediction OR TITLE_ABS:learning)',
    "dependency_context": '(TITLE_ABS:"gene dependency" OR TITLE_ABS:"CRISPR") AND (TITLE_ABS:"drug response" OR TITLE_ABS:"drug resistance") AND (TITLE_ABS:prediction OR TITLE_ABS:learning)',
    "active_uncertainty": '(TITLE_ABS:"active learning" OR TITLE_ABS:"conformal prediction" OR TITLE_ABS:"uncertainty quantification") AND (TITLE_ABS:perturbation OR TITLE_ABS:"drug response" OR TITLE_ABS:"drug resistance")',
    "local_rna_function": '(TITLE_ABS:"RNA representation" OR TITLE_ABS:"RNA foundation" OR TITLE_ABS:"miRNA target") AND (TITLE_ABS:functional OR TITLE_ABS:regulatory OR TITLE_ABS:bias)',
    "source_shift": '(TITLE_ABS:"drug response" OR TITLE_ABS:"drug-target" OR TITLE_ABS:"ncRNA-drug") AND (TITLE_ABS:"distribution shift" OR TITLE_ABS:"selection bias" OR TITLE_ABS:"positive-unlabeled")',
}
DOIS = {
    "incrd": "10.1093/bioinformatics/btag029",
    "mifndra": "10.1093/bib/bbag322",
    "mphgnn": "10.1093/bib/bbag013",
    "hdbi": "10.1021/acs.jcim.6c01116",
    "mvgsca": "10.1109/TCBBIO.2026.3729865",
    "ncrdllm": "10.1021/acs.jcim.5c03011",
    "nucleicbert": "10.1038/s42256-026-01295-9",
    "mirbind2": "10.1093/bioinformatics/btag410",
    "parnet": "10.64898/2026.08.08.743506",
    "forge": "10.1038/s41467-026-73977-2",
    "dispa": "10.1093/bioinformatics/btag272",
    "scradar": "10.1371/journal.pcbi.1014392",
    "omicstransformer": "10.1093/bioinformatics/btag468",
    "mirbench": "10.1093/bioinformatics/btaf233",
    "pulsnar": "10.7717/peerj-cs.2451",
    "perturbation_benchmark": "10.1126/sciadv.aed3414",
    "mirbase": "10.1093/nar/gkae1072",
    "lnctard": "10.1093/nar/gkac984",
}


def fetch(label, url):
    req = urllib.request.Request(url, headers={"User-Agent": "NcRNALiteratureAssessment/1.0", "Accept": "application/json,application/xml;q=0.9,*/*;q=0.5"})
    record = {"label": label, "url": url, "retrieved_at": datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read()
            record.update(status=response.status, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        suffix = ".json" if raw.lstrip().startswith((b"{", b"[")) else ".xml"
        (ROOT / (label + suffix)).write_bytes(raw)
        record["path"] = label + suffix
        if label.startswith("crossref_"):
            m = json.loads(raw)["message"]
            record["metadata"] = {"title": m.get("title"), "doi": m.get("DOI"), "journal": m.get("container-title"), "published": m.get("published"), "online": m.get("published-online"), "type": m.get("type"), "relation": m.get("relation")}
        elif label.startswith("epmc_"):
            data = json.loads(raw)
            record["total_hits"] = data.get("hitCount")
            record["records"] = [{"title": x.get("title"), "doi": x.get("doi"), "id": x.get("id"), "source": x.get("source"), "pmcid": x.get("pmcid"), "date": x.get("firstPublicationDate"), "journal": x.get("journalInfo", {}).get("journal", {}).get("title"), "abstract": x.get("abstractText")} for x in data.get("resultList", {}).get("result", [])]
        elif label.startswith("pubmed_"):
            record["search_result"] = json.loads(raw).get("esearchresult")
    except Exception as exc:
        record["error"] = str(exc)
    return record


def main():
    jobs = []
    for key, query in QUERIES.items():
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode({"query": f"({query}) AND FIRST_PDATE:[2025-01-01 TO {AS_OF}]", "format": "json", "resultType": "core", "pageSize": 30, "sort_date": "y"})
        jobs.append(("epmc_" + key, url))
    for key, doi in DOIS.items():
        jobs.append(("crossref_" + key, "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")))
    term = '("ncRNA-drug"[Title/Abstract] OR "miRNA-drug"[Title/Abstract] OR "lncRNA-drug"[Title/Abstract]) AND ("2025/01/01"[Date - Publication] : "2026/09/13"[Date - Publication])'
    jobs.append(("pubmed_ncdrug", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode({"db": "pubmed", "term": term, "retmode": "json", "retmax": 40, "sort": "pub_date"})))
    jobs.append(("arxiv_uncertainty", "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": '(all:"perturbation" AND (ti:"active learning" OR ti:"uncertainty" OR ti:"conformal")) AND submittedDate:[202501010000 TO 202609132359]', "max_results": 10, "sortBy": "submittedDate", "sortOrder": "descending"})))
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, *j) for j in jobs]
        for future in as_completed(futures):
            record = future.result()
            results.append(record)
            summary = {k: v for k, v in record.items() if k in {"label", "status", "error", "total_hits", "metadata"}}
            if "records" in record:
                summary["records"] = [{k: v for k, v in r.items() if k != "abstract"} for r in record["records"][:12]]
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    manifest = {"as_of": AS_OF, "queries": QUERIES, "results": results}
    (ROOT / "search_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    unique = {}
    for r in results:
        for x in r.get("records", []):
            key = (x.get("doi") or x["title"]).lower().strip()
            if key not in unique:
                unique[key] = x
    (ROOT / "discovery_unique.json").write_text(json.dumps(list(unique.values()), ensure_ascii=False, indent=2))
    print(json.dumps({"queries": len(QUERIES), "returned_records": sum(len(r.get("records", [])) for r in results), "unique_records": len(unique), "failures": sum("error" in r for r in results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
