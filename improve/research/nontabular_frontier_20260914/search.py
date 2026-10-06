"""Public literature discovery; preserve original metadata and failed requests."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
QUERIES = {
    "graph_diffusion": "graph diffusion collaborative filtering recommendation",
    "diffrec": "diffusion recommender model",
    "product_kernel": "Kronecker kernel link prediction",
    "graph_gp": "Gaussian processes graph link prediction",
    "spectral": "spectral graph filtering collaborative recommendation",
    "pu_matrix": "positive unlabeled matrix completion drug miRNA",
    "graph_autoencoder": "masked graph autoencoder collaborative filtering recommendation",
    "ultra": "Towards Foundation Models for Knowledge Graph Reasoning",
}


def fetch(name, query, source):
    params = ({"query.bibliographic": query, "rows": 6,
               "filter": "from-pub-date:2023-01-01,until-pub-date:2026-09-14"}
              if source == "crossref" else
              {"search": query, "per-page": 6,
               "filter": "from_publication_date:2023-01-01,to_publication_date:2026-09-14"})
    url = ("https://api.crossref.org/works?" if source == "crossref" else
           "https://api.openalex.org/works?") + urllib.parse.urlencode(params)
    entry = dict(query=query, source=source, url=url,
                 retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NcRDGraphResearch/1.0"})
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read()
            entry["status"] = response.status
        path = ROOT / (source + "_" + name + ".json")
        path.write_bytes(raw)
        entry.update(file=path.name, sha256=hashlib.sha256(raw).hexdigest())
        obj = json.loads(raw)
        rows = obj["message"]["items"] if source == "crossref" else obj["results"]
        entry["records"] = [dict(
            title=r.get("display_name") or (r.get("title") or [""])[0],
            doi=r.get("doi") or r.get("DOI"),
            date=r.get("publication_date") or r.get("published"),
            abstract=r.get("abstract") or r.get("abstract_inverted_index"),
            authors=r.get("author") or r.get("authorships"),
            location=r.get("primary_location") or r.get("container-title")) for r in rows]
    except Exception as exc:
        entry["error"] = repr(exc)
    return entry


def main():
    records = []
    # CrossRef (T1) first. The no-MCP OpenAlex fallback supplements missing CS metadata.
    for source in ("crossref", "openalex"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch, n, q, source) for n, q in QUERIES.items()]
            for future in as_completed(futures):
                entry = future.result()
                records.append(entry)
                print(json.dumps({"source": source, "query": entry["query"],
                                  "error": entry.get("error"),
                                  "titles": [(r["title"], r["doi"]) for r in entry.get("records", [])]},
                                 ensure_ascii=False), flush=True)
    (ROOT / "manifest.json").write_text(json.dumps(dict(
        as_of="2026-09-14", scope="Discovery metadata; full-text verification recorded separately.",
        records=records), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
