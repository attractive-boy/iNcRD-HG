"""
Turn `run_bench.py` output into the tables that go in the paper.

    python analysis/summarize_bench.py --results ./9.bench/results_8seed.csv
"""

import argparse

import pandas as pd

SPLIT_ORDER = ["random", "matched", "cold_rna", "cold_drug", "cold_pair"]
METHOD_ORDER = ["degree", "iNcRD-HG", "GAT", "GraphSAGE", "ASAP", "GeneralGNN", "GraphConv", "GCN"]
METRICS = {
    "auc": "global AUC (pooled over drugs)",
    "auc_per_drug": "per-drug macro AUC",
    "rank_auc": "full-ranking AUC",
    "hits@30": "Hits@30",
}


def table(frame, metric):
    wide = frame.pivot_table(index="method", columns=["task", "split"], values=metric)
    wide = wide.reindex([m for m in METHOD_ORDER if m in wide.index])
    columns = [(t, s) for t in ("LncDrug", "MiDrug") for s in SPLIT_ORDER]
    return wide.reindex(columns=pd.MultiIndex.from_tuples(columns))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="./9.bench/results.csv")
    parser.add_argument("--csv-out", default=None, help="optional path for a tidy combined CSV")
    args = parser.parse_args()

    frame = pd.read_csv(args.results)

    for metric, caption in METRICS.items():
        print(f"\n### {caption}  [{metric}]")
        print(table(frame, metric).round(3).to_string())

    # How much of each method's headline number survives debiasing.
    print("\n### drop from the released protocol to the hardest honest split")
    print("    (per-drug macro AUC; 'random' is the protocol the paper reports)")
    rows = []
    for task in ("LncDrug", "MiDrug"):
        sub = frame[frame.task == task]
        for method in METHOD_ORDER:
            m = sub[sub.method == method].set_index("split")["auc_per_drug"]
            if m.empty:
                continue
            debiased = m.reindex([s for s in SPLIT_ORDER if s != "random"]).dropna()
            rows.append({
                "task": task, "method": method,
                "random": m.get("random", float("nan")),
                "debiased_mean": debiased.mean(),
                "debiased_worst": debiased.min(),
            })
    summary = pd.DataFrame(rows)
    summary["delta"] = summary["debiased_mean"] - summary["random"]
    print(summary.round(3).to_string(index=False))

    # Seed stability: a method whose spread swamps the reported margins is not
    # separable from its baselines no matter which way the mean falls.
    print("\n### seed spread on the debiased splits (per-drug macro AUC std)")
    spread = frame[frame.split != "random"].pivot_table(
        index="method", columns=["task", "split"], values="auc_per_drug_std"
    ).reindex([m for m in METHOD_ORDER if m != "degree"])
    print(spread.round(3).to_string())

    if args.csv_out:
        frame.to_csv(args.csv_out, index=False)
        print(f"\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
