# -*- coding: utf-8 -*-
"""Summarise the official-DeepLDA reproduction results and compare to the paper."""
import csv
import glob
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))

# paper values: (AUC, AUPR, F1, MCC)
PAPER = {
    ("1", 1): (0.9583, 0.8601, 0.8625, 0.7628),   # balanced NoncoRNA
    ("1", 0): (0.9536, 0.8511, 0.8612, 0.7562),   # unbalanced NoncoRNA
    ("2", 1): (0.9819, 0.8687, 0.9163, 0.7885),   # balanced ncDR
    ("2", 0): (0.9728, 0.8572, 0.8876, 0.7792),   # unbalanced ncDR
}
# paper GCN/GAT "average metric" (= mean of AUC/AUPR/F1/MCC), from Table 3
PAPER_AVG = {
    ("1", 1): {"gat": 0.8289, "gcn": 0.7808},
    ("1", 0): {"gat": 0.7820, "gcn": 0.7541},
    ("2", 1): {"gat": 0.7949, "gcn": 0.7769},
    ("2", 0): {"gat": 0.7740, "gcn": 0.7466},
}


def load():
    rows = []
    for fp in glob.glob(os.path.join(HERE, "results_*.csv")):
        for r in csv.DictReader(open(fp)):
            if "balanced" not in r:
                r["balanced"] = "0"
            rows.append({k: (float(v) if k in ("auc", "aupr", "f1", "mcc") else v)
                         for k, v in r.items()})
    return rows


def main():
    rows = load()
    groups = {}
    for r in rows:
        key = (str(r["dataset"]), int(r["balanced"]), r["variant"])
        groups.setdefault(key, []).append(r)

    order = [("1", 0), ("1", 1), ("2", 0), ("2", 1)]
    dsname = {"1": "NoncoRNA (Dataset1)", "2": "ncDR (Dataset2)"}
    lines = ["# DeepLDA official-code reproduction summary", ""]
    for ds, bal in order:
        tag = "balanced" if bal else "unbalanced"
        lines.append(f"## {dsname[ds]} — {tag}")
        pa, pp, pf, pm = PAPER[(ds, bal)]
        lines.append(f"paper DeepLDA: AUC {pa:.4f} AUPR {pp:.4f} F1 {pf:.4f} MCC {pm:.4f}")
        lines.append("")
        lines.append("| variant | n | AUC | AUPR | F1 | MCC | paper-avg |")
        lines.append("|---|---|---|---|---|---|---|")
        for v in ("deeplda", "gcn", "gat"):
            g = groups.get((ds, bal, v))
            if not g:
                lines.append(f"| {v} | 0 | - | - | - | - | - |")
                continue
            def m(k):
                x = [r[k] for r in g]
                return f"{st.mean(x):.4f}±{st.pstdev(x):.4f}"
            pavg = PAPER_AVG.get((ds, bal), {}).get(v)
            pavg = f"{pavg:.4f}" if pavg else "-"
            lines.append(f"| {v} | {len(g)} | {m('auc')} | {m('aupr')} | {m('f1')} | {m('mcc')} | {pavg} |")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "SUMMARY_deeplda_official.md"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
