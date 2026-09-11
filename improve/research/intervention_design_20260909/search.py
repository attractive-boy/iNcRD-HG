"""Bounded public literature check for the proposed research objective.

Discovery/abstract review only; no completeness or first-in-field claim.
Does not touch model code, data splits, or training artifacts.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from search_expression_consensus import arxiv, crossref, pubmed


QUERIES = [
    ("CrossRef", "Predicting transcriptional outcomes of novel multigene perturbations with GEARS"),
    ("CrossRef", "Predicting cellular responses to complex perturbations in high-throughput screens"),
    ("CrossRef", "Learning single-cell perturbation responses using neural optimal transport"),
    ("CrossRef", "Time-staggered inhibition of multiple oncogenic pathways to enhance therapeutic efficacy"),
    ("CrossRef", "Endogenous miRNA and target concentrations determine susceptibility to potential ceRNA competition"),
    ("PubMed", '(microRNA[Title/Abstract] OR miRNA[Title/Abstract] OR lncRNA[Title/Abstract]) AND (drug[Title/Abstract] OR chemotherapy[Title/Abstract]) AND (perturbation[Title/Abstract] OR intervention[Title/Abstract] OR combination[Title/Abstract]) AND ("deep learning"[Title/Abstract] OR "machine learning"[Title/Abstract] OR "graph neural"[Title/Abstract])'),
    ("PubMed", '(microRNA[Title/Abstract] OR miRNA[Title/Abstract]) AND ("drug sensitivity"[Title/Abstract] OR chemoresistance[Title/Abstract] OR chemosensitivity[Title/Abstract]) AND ("high-throughput"[Title/Abstract] OR "functional screen"[Title/Abstract])'),
    ("PubMed", '(perturbation[Title/Abstract]) AND (prediction[Title/Abstract] OR predicting[Title/Abstract]) AND (benchmark[Title/Abstract] OR "State"[Title/Abstract]) AND 2025:2026[dp]'),
    ("PubMed", '(microRNA[Title/Abstract] OR miRNA[Title/Abstract]) AND ("target competition"[Title/Abstract] OR "target switching"[Title/Abstract] OR "target abundance"[Title/Abstract]) AND (drug[Title/Abstract] OR resist*[Title/Abstract])'),
    ("PubMed", '("drug treatment order"[Title/Abstract] OR "drug sequencing"[Title/Abstract] OR "sequential drug"[Title/Abstract] OR "order-dependent"[Title/Abstract]) AND (cancer[Title/Abstract] OR tumor[Title/Abstract]) AND (predict*[Title/Abstract] OR learn*[Title/Abstract])'),
    ("PubMed", '24145123[uid] OR 21289630[uid] OR 19074899[uid] OR 28487996[uid] OR 24755562[uid] OR 25356050[uid] OR 30864705[uid]'),
    ("arXiv", 'all:"perturbation" AND (ti:"State" OR ti:"combination" OR ti:"benchmark")'),
]

PUBMED_LOCK = threading.Lock()


def run_query(source, query):
    fn = {"CrossRef": crossref, "PubMed": pubmed, "arXiv": arxiv}[source]
    if source == "PubMed":
        with PUBMED_LOCK:
            time.sleep(0.4)
            return fn(query)
    return fn(query)


def key(record):
    doi = (record.get("doi") or "").strip().lower()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    if doi:
        return "doi:" + doi
    title = re.sub(r"\W+", "", record["title"]).lower()
    author = (record.get("authors") or [""])[0].lower()
    return "title:" + title + "|" + author


def main():
    responses = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_query, source, query): (source, query)
                   for source, query in QUERIES}
        for future in as_completed(futures):
            source, query = futures[future]
            try:
                response = future.result()
            except Exception as error:
                response = dict(source=source, query=query, error=str(error), records=[])
            responses.append(response)
            print(source, len(response["records"]), response.get("error", ""), query, flush=True)
    merged = {}
    for response in responses:
        for record in response["records"]:
            identifier = key(record)
            entry = merged.setdefault(identifier, {"record": record, "sources": [], "queries": []})
            if record["source"] not in entry["sources"]:
                entry["sources"].append(record["source"])
            if record["query"] not in entry["queries"]:
                entry["queries"].append(record["query"])
            previous = entry["record"]
            if (bool(record.get("doi")), len(record.get("abstract", ""))) > (bool(previous.get("doi")), len(previous.get("abstract", ""))):
                entry["record"] = record
    payload = dict(
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="Targeted metadata and abstract review; finite result caps; not a systematic review or novelty certification.",
        preflight="PubMed, CrossRef, arXiv reachable before search",
        mcp_available=False,
        fallback="Direct public PubMed, CrossRef, arXiv APIs",
        responses=responses,
        deduplicated=list(merged.values()),
    )
    path = Path(__file__).with_name("search_results.json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("Saved", path, "records", len(merged), flush=True)


if __name__ == "__main__":
    main()
