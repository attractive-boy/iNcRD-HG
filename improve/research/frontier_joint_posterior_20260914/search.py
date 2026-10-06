"""Bounded public-API literature discovery, with raw responses and provenance."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
QUERIES = {
    "tabdpt": "TabDPT Scaling Tabular Foundation Models",
    "mitra": "Mitra Mixed Synthetic Priors Tabular Foundation Models",
    "graph_gp": "Gaussian process link prediction graph uncertainty",
    "dyadic": "drug interaction Kronecker kernel conditional prediction",
    "diffusion": "diffusion models link prediction graph 2025",
    "ultra": "Towards Foundation Models for Knowledge Graph Reasoning",
    "puicl": "PU-ICL Positive Unlabeled In Context Learning",
}


def fetch(label, url, source):
    row = dict(label=label, url=url, source=source,
               retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NcRDResearch/1.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = resp.read()
            row.update(status=resp.status, response_url=resp.url)
        file = ROOT / (label + (".xml" if source == "arxiv" else ".json"))
        file.write_bytes(data)
        row.update(file=file.name, sha256=hashlib.sha256(data).hexdigest())
        if source == "crossref":
            obj = json.loads(data)
            row["records"] = [dict(title=r.get("title", [""])[0], doi=r.get("DOI"),
                published=r.get("published"), journal=r.get("container-title"),
                abstract=r.get("abstract"), authors=r.get("author"),
                type=r.get("type"), url=r.get("URL")) for r in obj["message"].get("items", [])]
        else:
            root = ET.fromstring(data)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            row["records"] = [dict(title=" ".join(e.findtext("a:title", "", ns).split()),
                id=e.findtext("a:id", "", ns), published=e.findtext("a:published", "", ns),
                abstract=e.findtext("a:summary", "", ns),
                authors=[a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)])
                for e in root.findall("a:entry", ns)]
    except Exception as exc:
        row["error"] = repr(exc)
    return row


def main():
    jobs = []
    for name, query in QUERIES.items():
        args = {"query.bibliographic": query, "rows": 5,
                "filter": "from-pub-date:2024-01-01,until-pub-date:2026-09-14"}
        jobs.append(("crossref_" + name, "https://api.crossref.org/works?" +
                     urllib.parse.urlencode(args), "crossref"))
    for name, query in {"tabdpt": "ti:TabDPT", "mitra": "ti:Mitra AND all:tabular",
                        "puicl": "all:PU-ICL", "diffusion": 'ti:diffusion AND ti:"link prediction"',
                        "graph_gp": 'all:"Gaussian process" AND all:"link prediction"'}.items():
        args = {"search_query": query, "max_results": 6, "sortBy": "submittedDate", "sortOrder": "descending"}
        jobs.append(("arxiv_" + name, "https://export.arxiv.org/api/query?" +
                     urllib.parse.urlencode(args), "arxiv"))
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch, *job) for job in jobs]):
            row = future.result()
            rows.append(row)
            print(json.dumps({"label": row["label"], "status": row.get("status"),
                "error": row.get("error"), "titles": [r["title"] for r in row.get("records", [])]},
                ensure_ascii=False), flush=True)
    (ROOT / "search_manifest.json").write_text(json.dumps(dict(as_of="2026-09-14", results=rows,
        scope="Targeted discovery; search hits are not all read or relevant."), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
