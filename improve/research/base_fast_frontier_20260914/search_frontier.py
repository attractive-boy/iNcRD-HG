"""Bounded public literature search for the original GNN baseline review."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
AS_OF = "2026-09-14"
DOIS = {
    "mifndra": "10.1093/bib/bbag322",
    "mphgnn": "10.1093/bib/bbag013",
    "agclnda": "10.1109/JBHI.2025.3531112",
    "omics_transformer": "10.1093/bioinformatics/btag468",
    "mirbench": "10.1093/bioinformatics/btaf233",
    "pulsnar": "10.7717/peerj-cs.2451",
    "cpu": "10.1109/TNNLS.2026.3706351",
    "mv gsca": "10.1109/TCBBIO.2026.3729865",
    "plmf_mda": "10.1371/journal.pcbi.1013968",
    "m2umol": "10.1038/s41467-026-69302-6",
    "dispa": "10.1093/bioinformatics/btag272",
}


def fetch(label, url):
    record = {"label": label, "url": url,
              "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "NcRDBaseLiteratureReview/1.0",
                                                       "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=22) as response:
            raw = response.read()
            record.update(status=response.status, response_url=response.url)
        obj = json.loads(raw)
        path = ROOT / (label + ".json")
        path.write_bytes(raw)
        record.update(file=path.name, sha256=hashlib.sha256(raw).hexdigest())
        if label.startswith("crossref_"):
            message = obj["message"]
            rows = message.get("items", [message])
            record["records"] = [{"title": r.get("title", [""])[0], "doi": r.get("DOI"),
                "published": r.get("published"), "published_online": r.get("published-online"),
                "journal": r.get("container-title"), "abstract": r.get("abstract"),
                "links": r.get("link"), "type": r.get("type")} for r in rows]
        elif label.startswith("epmc_"):
            record["total_hits"] = obj.get("hitCount")
            record["records"] = [{"title": r.get("title"), "doi": r.get("doi"),
                "pmid": r.get("id"), "pmcid": r.get("pmcid"),
                "date": r.get("firstPublicationDate"), "authors": r.get("authorString"),
                "journal": r.get("journalInfo", {}).get("journal", {}).get("title"),
                "abstract": r.get("abstractText")} for r in obj.get("resultList", {}).get("result", [])]
        elif label == "pubmed_direct":
            record["result"] = obj.get("esearchresult")
    except Exception as error:
        record["error"] = str(error)
    return record


def main():
    jobs = [("crossref_doi_" + key.replace(" ", ""),
             "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""))
            for key, doi in DOIS.items()]
    for key, query in {
        "target_edge": "link prediction target edges distribution shift masking",
        "spot_target": "SpotTarget Rethinking Effect Target Edges Link Prediction",
        "fake_edge": "FakeEdge Alleviate Dataset Shift Link Prediction",
    }.items():
        params = {"query.bibliographic": query, "rows": 6,
                  "filter": "from-pub-date:2022-01-01,until-pub-date:" + AS_OF}
        jobs.append(("crossref_query_" + key, "https://api.crossref.org/works?" + urllib.parse.urlencode(params)))
    for key, query in {
        "direct": '(TITLE_ABS:"ncRNA-drug" OR TITLE_ABS:"miRNA-drug" OR TITLE_ABS:"lncRNA-drug") AND (TITLE_ABS:prediction OR TITLE_ABS:learning)',
        "pu": '(TITLE_ABS:"positive unlabeled" OR TITLE_ABS:"positive-unlabeled") AND (TITLE_ABS:drug OR TITLE_ABS:miRNA)',
        "expression": 'TITLE_ABS:OmicsTransformer',
    }.items():
        query = "(" + query + ") AND FIRST_PDATE:[2025-01-01 TO " + AS_OF + "]"
        jobs.append(("epmc_" + key, "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" +
                     urllib.parse.urlencode({"query": query, "format": "json", "resultType": "core", "pageSize": 20, "sort_date": "y"})))
    term = '("ncRNA-drug"[Title/Abstract] OR "miRNA-drug"[Title/Abstract]) AND ("2025/01/01"[Date - Publication] : "2026/09/14"[Date - Publication])'
    jobs.append(("pubmed_direct", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" +
                 urllib.parse.urlencode({"db": "pubmed", "term": term, "retmode": "json", "retmax": 30, "sort": "pub_date"})))
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch, *j) for j in jobs]):
            result = future.result()
            results.append(result)
            print(json.dumps({"label": result["label"], "status": result.get("status"), "error": result.get("error"),
                              "titles": [r["title"] for r in result.get("records", [])]}, ensure_ascii=False), flush=True)
    (ROOT / "search_manifest.json").write_text(json.dumps({"as_of": AS_OF, "results": results,
        "scope": "Targeted review; not an exhaustive systematic review. arXiv API preflight timed out; Crossref, PubMed, Europe PMC and OpenAlex used."}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
