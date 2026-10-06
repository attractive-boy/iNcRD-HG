"""Audit entity sequences without reading association labels or changing inputs.

The mature miRNA snapshot must already have been downloaded from miRBase.
Only case-insensitive exact identifiers are accepted. Candidate arm names are
reported for manual curation, never substituted for the supplied identifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def read_fasta(path: Path) -> list[dict]:
    records = []
    header, chunks = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append({"header": header, "sequence": "".join(chunks)})
            header, chunks = line[1:].strip(), []
        elif line.strip():
            if header is None:
                raise ValueError(f"Sequence before FASTA header: {path}")
            chunks.append(line.strip().upper().replace("T", "U"))
    if header is not None:
        records.append({"header": header, "sequence": "".join(chunks)})
    return records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=here.parents[3] / "incrd-run")
    parser.add_argument("--mature-fasta", type=Path,
                        default=here / "mirbase_mature.fa")
    parser.add_argument("--out-dir", type=Path, default=here)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    node_path = args.data_root / "2.construct_network/miRNA.csv"
    with node_path.open(newline="") as handle:
        names = [row["x"].strip() for row in csv.DictReader(handle)]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate input miRNA node identifiers")

    reference = read_fasta(args.mature_fasta)
    by_name = defaultdict(list)
    for rec in reference:
        parts = rec["header"].split()
        if parts[0].casefold().startswith("hsa-"):
            by_name[parts[0].casefold()].append({
                "name": parts[0], "accession": parts[1], **rec,
            })

    rows, resolved = [], {}
    for name in names:
        exact = by_name.get(name.casefold(), [])
        row = {"node_name": name, "status": "unresolved", "mirbase_name": "",
               "accession": "", "sequence": "", "length_nt": "",
               "unassigned_candidates": ""}
        if len(exact) == 1:
            rec = exact[0]
            row.update(status="exact_case_insensitive", mirbase_name=rec["name"],
                       accession=rec["accession"], sequence=rec["sequence"],
                       length_nt=len(rec["sequence"]))
            resolved[name] = rec
        else:
            if exact:
                row["status"] = "ambiguous_exact_identifier"
                candidates = exact
            else:
                candidates = [candidate for arm in ["-5p", "-3p"]
                              for candidate in by_name.get(name.casefold() + arm, [])]
            row["unassigned_candidates"] = "|".join(
                f"{r['name']}:{r['accession']}" for r in candidates)
        rows.append(row)

    mapping_path = args.out_dir / "mature_alignment.csv"
    with mapping_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mature_path = args.out_dir / "mature_exact.fasta"
    mature_path.write_text("".join(
        f">{name} {rec['accession']} source_name={rec['name']}\n{rec['sequence']}\n"
        for name, rec in resolved.items()))

    lnc_stats, source_files = {}, [Path(__file__).resolve(), node_path, args.mature_fasta]
    for path in sorted((args.data_root / "11.seq_data/sequences").glob("lncRNA_seq*.fasta")):
        records = read_fasta(path)
        lengths = [len(r["sequence"]) for r in records]
        lnc_stats[path.name] = {
            "records": len(records), "minimum_nt": min(lengths),
            "median_nt": median(lengths), "maximum_nt": max(lengths),
            "at_1022_nt": lengths.count(1022),
            "longer_than_1022_nt": sum(x > 1022 for x in lengths),
            "sha256": sha256(path),
        }
        source_files.append(path)

    prior_path = here / "input_audit.json"
    prior = json.loads(prior_path.read_text())
    arm_checks = []
    for pair in prior["cross_arm_identical_sequence_pairs"]:
        if not pair["same_stem"]:
            continue
        first, second = pair["first"], pair["second"]
        a, b = resolved.get(first), resolved.get(second)
        arm_checks.append({
            "first": first, "second": second,
            "both_exactly_resolved": a is not None and b is not None,
            "mature_sequences_differ": a["sequence"] != b["sequence"] if a and b else None,
        })
    source_files.append(prior_path)
    lengths = [len(r["sequence"]) for r in resolved.values()]
    result = {
        "description": "Read-only input audit; no association labels, splits, or predictions opened.",
        "snapshot_date": "2026-09-12",
        "official_mature_source": "https://www.mirbase.org/download/mature.fa",
        "matching_policy": "Case-insensitive exact full identifier; preserve arm, locus, and species. No inferred aliases or precursor cropping.",
        "nodes": len(names), "human_mature_reference_records": sum(map(len, by_name.values())),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "resolved_fraction": len(resolved) / len(names),
        "resolved_length_nt": {"min": min(lengths), "median": median(lengths), "max": max(lengths)},
        "resolved_unique_sequences": len(set(r["sequence"] for r in resolved.values())),
        "unresolved_nodes": [r for r in rows if r["status"] != "exact_case_insensitive"],
        "previous_same_stem_cross_arm_pairs": arm_checks,
        "lnc_sequence_files": lnc_stats,
        "source_sha256": {str(p.resolve()): sha256(p) for p in source_files},
        "output_sha256": {p.name: sha256(p) for p in [mapping_path, mature_path]},
        "limits": [
            "This public snapshot has not been checked against original experimental accessions or historical miRBase versions.",
            "Exact reference matching does not establish sample-specific isomiR abundance or verify the source association labels.",
            "All checked lncRNA FASTA alternatives are already truncated; they cannot restore missing full-length sequence.",
            "These sequence channels are not inputs to the existing 36-feature TabICL baseline.",
        ],
    }
    (args.out_dir / "mature_input_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "nodes": result["nodes"], "status_counts": result["status_counts"],
        "resolved_length_nt": result["resolved_length_nt"],
        "previous_same_stem_cross_arm_pairs": len(arm_checks),
        "resolved_cross_arm_pairs_now_distinct": sum(r["mature_sequences_differ"] is True for r in arm_checks),
        "unresolved_count": len(result["unresolved_nodes"]),
        "unresolved_examples": [r["node_name"] for r in result["unresolved_nodes"][:10]],
        "lnc_sequence_files": lnc_stats,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
