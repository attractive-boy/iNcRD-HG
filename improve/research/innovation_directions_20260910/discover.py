"""Bounded discovery using public scholarly sources; no project data is sent."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import unescape
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from improve.search_expression_consensus import pubmed


def clean(value):
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", value)).split())


def fetch(url):
    req = Request(url, headers={"User-Agent": "academic-method-discovery/1.0"})
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def arxiv_search(query):
    url = "https://arxiv.org/search/?" + urlencode({
        "query": query, "searchtype": "all", "abstracts": "show",
        "order": "-announced_date_first", "size": 50,
    })
    body = fetch(url)
    records = []
    for block in re.findall(r'<li class="arxiv-result">(.*?)</li>', body, re.S):
        link = re.search(r'href="(https://arxiv.org/abs/[^"]+)"', block)
        title = re.search(r'<p class="title[^"]*">(.*?)</p>', block, re.S)
        abstract = re.search(r'<span class="abstract-full[^"]*"[^>]*>(.*?)</p>', block, re.S)
        date = re.search(r'<p class="is-size-7[^"]*">(.*?)</p>', block, re.S)
        if link and title:
            records.append({"source": "arXiv", "query": query, "title": clean(title.group(1)),
                            "url": link.group(1), "date_text": clean(date.group(1)) if date else "",
                            "abstract": clean(abstract.group(1)).replace("△ Less", "").strip() if abstract else ""})
    return {"source": "arXiv", "query": query, "url": url, "records": records}


def crossref_recent(query):
    url = "https://api.crossref.org/works?" + urlencode({
        "query.bibliographic": query, "rows": 8,
        "filter": "from-pub-date:2025-01-01,until-pub-date:2026-09-10",
    })
    items = json.loads(fetch(url))["message"]["items"]
    records = [{"source": "CrossRef", "query": query, "title": " ".join(x.get("title", [])),
                "doi": x.get("DOI"), "url": x.get("URL"),
                "date": x.get("published", x.get("issued", {})).get("date-parts"),
                "venue": " ".join(x.get("container-title", [])),
                "abstract": clean(x.get("abstract", ""))} for x in items]
    return {"source": "CrossRef", "query": query, "url": url, "records": records}


JOBS = [
    (arxiv_search, '"link prediction" AND "diffusion"'),
    (arxiv_search, '"link prediction" AND "flow matching"'),
    (arxiv_search, '"positive-unlabeled" AND "graph"'),
    (arxiv_search, '"foundation" AND "link prediction"'),
    (arxiv_search, '"conformal" AND "link prediction"'),
    (arxiv_search, '"sheaf" AND "biological"'),
    (crossref_recent, 'positive unlabeled graph link prediction missing not at random'),
    (crossref_recent, 'ncRNA drug resistance diffusion flow matching'),
    (pubmed, '(ncRNA OR lncRNA OR miRNA) AND "drug resistance" AND ("deep learning" OR "graph neural" OR "prediction") AND ("2025/01/01"[Date - Publication] : "2026/09/10"[Date - Publication])'),
    (pubmed, '("positive unlabeled" OR "positive-unlabeled") AND (drug OR RNA OR graph) AND ("2025/01/01"[Date - Publication] : "2026/09/10"[Date - Publication])'),
]


def main():
    output = ROOT / "discovery.json"
    if output.exists():
        raise FileExistsError(output)
    responses = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fn, query): (fn.__name__, query) for fn, query in JOBS}
        for future in as_completed(pending):
            name, query = pending[future]
            try:
                response = future.result()
            except Exception as exc:
                response = {"source": name, "query": query, "error": str(exc), "records": []}
            responses.append(response)
            print(name, len(response["records"]), response.get("error", ""), query, flush=True)
    merged = {}
    for response in responses:
        for record in response["records"]:
            key = (record.get("doi") or record.get("url") or re.sub(r"\W+", "", record["title"])).lower()
            if key not in merged or len(record.get("abstract", "")) > len(merged[key].get("abstract", "")):
                merged[key] = record
    output.write_text(json.dumps({
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Bounded discovery; candidate records require relevance/date and primary-source checks.",
        "mcp_available": False, "fallback": "Official arXiv website and public CrossRef/PubMed APIs",
        "responses": responses, "deduplicated": list(merged.values()),
    }, ensure_ascii=False, indent=2) + "\n")
    for record in merged.values():
        print(json.dumps({k: record.get(k) for k in ("source", "title", "date_text", "date", "year", "url", "doi")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
