"""Bounded public-API literature search; no model training or data changes."""
import concurrent.futures
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
AS_OF = "2026-09-13"
DATE = f" FIRST_PDATE:[2024-01-01 TO {AS_OF}]"
QUERIES = {
    "ncdrug": '(TITLE_ABS:"drug resistance" AND (TITLE_ABS:ncRNA OR TITLE_ABS:"non-coding RNA" OR TITLE_ABS:miRNA OR TITLE_ABS:lncRNA) AND (TITLE_ABS:graph OR TITLE_ABS:contrastive OR TITLE_ABS:multimodal OR TITLE_ABS:"deep learning"))',
    "mirna_drug": '(TITLE_ABS:"miRNA-drug" OR TITLE_ABS:"microRNA-drug") AND (TITLE_ABS:predict OR TITLE_ABS:prediction)',
    "rna_foundation": '(TITLE_ABS:"RNA language" OR TITLE_ABS:"RNA foundation" OR TITLE_ABS:RNAErnie OR TITLE_ABS:RiNALMo OR TITLE_ABS:RhoFold)',
    "perturbation_models": '(TITLE_ABS:"drug response" OR TITLE_ABS:"chemical perturbation") AND (TITLE_ABS:"foundation model" OR TITLE_ABS:multimodal OR TITLE_ABS:pathway) AND (TITLE_ABS:predict OR TITLE_ABS:prediction)',
    "foundation_benchmark": '(TITLE_ABS:"foundation models" OR TITLE_ABS:"deep learning") AND TITLE_ABS:perturbation AND (TITLE_ABS:benchmark OR TITLE_ABS:benchmarking OR TITLE_ABS:evaluation)',
    "pu_association": '(TITLE_ABS:"positive unlabeled" OR TITLE_ABS:"positive-unlabeled" OR TITLE_ABS:"selected not at random") AND (TITLE_ABS:drug OR TITLE_ABS:RNA OR TITLE_ABS:selection OR TITLE_ABS:bias)',
    "missing_omics": '(TITLE_ABS:"multi-omics" OR TITLE_ABS:multimodal) AND (TITLE_ABS:"missing modalities" OR TITLE_ABS:"incomplete modalities" OR TITLE_ABS:"uncertainty-aware") AND (TITLE_ABS:cancer OR TITLE_ABS:drug)',
}
CROSSREF = {
    "rinalmo": "RiNALMo general-purpose RNA language models",
    "rnaernie": "RNAErnie RNA language modelling",
    "madrigal": "Madrigal multimodal drug interaction",
    "perturbnet": "PerturbNet predicting cellular responses unseen perturbations",
    "foundation_evaluation": "systematic evaluation single-cell foundation models perturbation",
}

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "LiteratureReview/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read())

def epmc(key, q):
    params = {"query": f"({q}) AND{DATE}", "format": "json", "resultType": "core", "pageSize": 25}
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(params)
    return "epmc_" + key, url, fetch(url)

def crossref(key, q):
    params = {"query.bibliographic": q, "rows": 8, "filter": f"from-pub-date:2024-01-01,until-pub-date:{AS_OF}"}
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    return "crossref_" + key, url, fetch(url)

def summarize(kind, data):
    if kind.startswith("epmc_"):
        items = data.get("resultList", {}).get("result", [])
        return [{"title": x.get("title"), "doi": x.get("doi"), "pmid": x.get("id"), "pmcid": x.get("pmcid"), "date": x.get("firstPublicationDate"), "journal": x.get("journalInfo", {}).get("journal", {}).get("title"), "abstract": x.get("abstractText")} for x in items]
    return [{"title": (x.get("title") or [""])[0], "doi": x.get("DOI"), "date": x.get("published"), "journal": x.get("container-title"), "abstract": x.get("abstract")} for x in data.get("message", {}).get("items", [])]

def main():
    manifest = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(epmc, k, q): "epmc_" + k for k, q in QUERIES.items()}
        futures.update({pool.submit(crossref, k, q): "crossref_" + k for k, q in CROSSREF.items()})
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                key, url, data = future.result()
                (ROOT / (key + ".json")).write_text(json.dumps({"as_of": AS_OF, "url": url, "response": data}, ensure_ascii=False, indent=2))
                items = summarize(key, data)
                manifest.append({"key": key, "url": url, "count": len(items), "results": items})
                print(json.dumps({"key": key, "count": len(items), "top": [{k:v for k,v in x.items() if k != "abstract"} for x in items[:10]]}, ensure_ascii=False), flush=True)
            except Exception as exc:
                manifest.append({"key": key, "error": str(exc)})
                print(json.dumps({"key": key, "error": str(exc)}), flush=True)
    (ROOT / "discovery_summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
