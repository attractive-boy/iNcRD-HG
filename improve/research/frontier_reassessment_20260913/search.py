"""Bounded public literature discovery and DOI verification, as of 2026-09-13."""
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
DISCOVERY = {
    "recent_ncdrug": '(TITLE_ABS:"ncRNA-drug" OR TITLE_ABS:"miRNA-drug" OR TITLE_ABS:"lncRNA-drug" OR TITLE_ABS:"circRNA-drug")',
    "lncrna_function": '(TITLE_ABS:lncRNA OR TITLE_ABS:"long noncoding RNA") AND (TITLE_ABS:"language model" OR TITLE_ABS:"foundation model" OR TITLE_ABS:"structure-aware")',
    "rna_local_function": '(TITLE_ABS:RNA) AND (TITLE_ABS:"binding site" OR TITLE_ABS:"functional repression" OR TITLE_ABS:"subcellular localization") AND (TITLE_ABS:"pre-trained" OR TITLE_ABS:"pretrained" OR TITLE_ABS:transformer)',
    "drug_missing_modalities": '(TITLE_ABS:drug) AND (TITLE_ABS:multimodal OR TITLE_ABS:"multi-view") AND (TITLE_ABS:"missing" OR TITLE_ABS:"perturbation" OR TITLE_ABS:"domain adaptation")',
    "context_perturbation": '(TITLE_ABS:"drug response" OR TITLE_ABS:perturbation) AND (TITLE_ABS:"cross-cell" OR TITLE_ABS:"unseen cell" OR TITLE_ABS:"cross-context" OR TITLE_ABS:"compositional generalization")',
    "signed_mechanism": '(TITLE_ABS:"signed graph" OR TITLE_ABS:"signed knowledge" OR TITLE_ABS:"directed knowledge") AND (TITLE_ABS:drug OR TITLE_ABS:RNA OR TITLE_ABS:omics OR TITLE_ABS:biological)',
    "pu_bias": '(TITLE_ABS:"positive-unlabeled" OR TITLE_ABS:"positive unlabeled" OR TITLE_ABS:"frequency class bias" OR TITLE_ABS:"selection bias") AND (TITLE_ABS:drug OR TITLE_ABS:RNA) AND (TITLE_ABS:prediction OR TITLE_ABS:learning)',
    "multitask": '(TITLE_ABS:"multi-task" OR TITLE_ABS:multitask) AND (TITLE_ABS:"drug response" OR TITLE_ABS:"ncRNA-drug" OR TITLE_ABS:"drug resistance") AND (TITLE_ABS:learning OR TITLE_ABS:prediction)',
}
VERIFY = {
    "mifndra": "10.1093/bib/bbag322",
    "mphgnn": "10.1093/bib/bbag013",
    "plmf_mda": "10.1371/journal.pcbi.1013968",
    "mvgsca": "10.1109/TCBBIO.2026.3729865",
    "nucleicbert": "10.1038/s42256-026-01295-9",
    "mirbind2": "10.1093/bioinformatics/btag410",
    "dispa": "10.1093/bioinformatics/btag272",
    "scradar": "10.1371/journal.pcbi.1014392",
    "omics_transformer": "10.1093/bioinformatics/btag468",
    "mirbench": "10.1093/bioinformatics/btaf233",
    "pulsnar": "10.7717/peerj-cs.2451",
    "perturbation_benchmark": "10.1126/sciadv.aed3414",
    "dc_metamg": "10.1109/JBHI.2026.3717803",
    "sigma_kg_flash": "10.64898/2026.04.29.721775",
}


def request(url):
    req = urllib.request.Request(url, headers={"User-Agent": "NcRNAResearchReview/1.0", "Accept": "application/json,application/xml;q=0.9,*/*;q=0.5"})
    with urllib.request.urlopen(req, timeout=25) as response:
        raw = response.read()
        return raw, {"url": url, "http_status": response.status,
                     "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def epmc_record(item):
    return {"title": item.get("title"), "doi": item.get("doi"),
            "pmid": item.get("id"), "pmcid": item.get("pmcid"),
            "date": item.get("firstPublicationDate"),
            "journal": item.get("journalInfo", {}).get("journal", {}).get("title"),
            "authors": item.get("authorString"), "abstract": item.get("abstractText"),
            "publication_types": item.get("pubTypeList"), "fulltext_urls": item.get("fullTextUrlList")}


def crossref_record(item):
    return {"title": (item.get("title") or [""])[0], "doi": item.get("DOI"),
            "journal": (item.get("container-title") or [""])[0],
            "authors": [" ".join(filter(None, [x.get("given"), x.get("family")])) for x in item.get("author", [])],
            "published": item.get("published"), "published_online": item.get("published-online"),
            "type": item.get("type"), "abstract": item.get("abstract"),
            "relation": item.get("relation"), "updates": item.get("update-to")}


def run_one(source, key, query):
    if source == "epmc":
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode({
            "query": f"({query}) AND FIRST_PDATE:[2025-01-01 TO {AS_OF}]",
            "format": "json", "resultType": "core", "pageSize": 25, "sort_date": "y"})
    elif source == "crossref":
        url = "https://api.crossref.org/works/" + urllib.parse.quote(query, safe="/")
    else:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
            "search_query": '(ti:RNA OR ti:drug) AND (all:"signed graph" OR all:"missing modalities" OR all:"multitask") AND submittedDate:[202501010000 TO 202609132359]',
            "max_results": 10, "sortBy": "submittedDate", "sortOrder": "descending"})
    label = source + "_" + key
    try:
        raw, info = request(url)
        suffix = ".xml" if source == "arxiv" else ".json"
        (ROOT / (label + suffix)).write_bytes(raw)
        if source == "epmc":
            data = json.loads(raw)
            items = [epmc_record(x) for x in data.get("resultList", {}).get("result", [])]
            info["total_hits"] = data.get("hitCount")
        elif source == "crossref":
            items = [crossref_record(json.loads(raw)["message"])]
        else:
            tree = ET.fromstring(raw)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            items = [{"title": " ".join(x.findtext("a:title", default="", namespaces=ns).split()),
                      "url": x.findtext("a:id", namespaces=ns),
                      "date": x.findtext("a:published", namespaces=ns),
                      "abstract": x.findtext("a:summary", namespaces=ns)} for x in tree.findall("a:entry", ns)]
        result = {"key": key, "source": source, "query": query, **info, "records": items}
    except Exception as exc:
        result = {"key": key, "source": source, "query": query, "url": url, "error": str(exc)}
    print(json.dumps({"key": key, "source": source, "error": result.get("error"),
                      "total_hits": result.get("total_hits"),
                      "records": [{k: v for k, v in x.items() if k in {"title", "doi", "date", "journal", "published_online", "type", "url"}}
                                  for x in result.get("records", [])[:12]]}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    jobs = [("epmc", key, query) for key, query in DISCOVERY.items()]
    jobs += [("crossref", key, doi) for key, doi in VERIFY.items()]
    jobs += [("arxiv", "frontier", "RNA/drug signed graph, missing modalities, or multitask; 2025–2026")]
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    manifest = {"as_of": AS_OF, "retrieved_at": datetime.now(timezone.utc).isoformat(), "results": results}
    (ROOT / "search_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
