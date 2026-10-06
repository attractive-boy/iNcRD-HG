# New external information channels: two tested, neither clears the gate

Date 2026-09-17. Scope: label-free diagnostics plus a full multi-seed / multi-split
evaluation. No default model replaced; no held-out test source opened.

## What was tried

Two external data sources, neither used anywhere in the project before:

1. **ncRNA–disease × drug–disease bridge.** RNADisease v4.0 (human miRNA and lncRNA
   experimental associations) + HMDD v4.0 + CTD chemicals–diseases
   (`therapeutic`/`marker/mechanism` only). For a pair (r, d): shared disease count,
   IDF-weighted shared count, IDF-weighted RNA-side count, RNA disease breadth, drug
   disease breadth. Coverage: 605/605 miRNA, 836/1322 lncRNA, 159/216 drugs.
2. **Drug-perturbation × miRNA-target connectivity.** Enrichr `Old_CMAP` and
   `Drug_Perturbations_from_GEO` up/down gene sets (82/216 drugs) intersected with
   miRTarBase target sets (416/605 miRNA). Signed and unsigned overlap features.

## Label-free signal (inner MiDrug / LncDrug validation)

| feature | MiDrug AUC / AP | LncDrug AUC / AP |
|---|---|---|
| disease GBDT (5 columns) | 0.9064 / 0.8629 | 0.9863 / 0.9681 |
| perturbation connectivity (best single) | 0.6367 / 0.6291 | — |

Both carry real standalone signal. Neither is orthogonal to the accepted reference:
blending the disease GBDT into `logit(reference)` moves **MiDrug AP by +0.0004 to
+0.0007 in all three inference seeds but leaves AUC flat or −0.0001**, and LncDrug is
flat. The perturbation channel is weaker and adds nothing.

## Full gate (accepted shared-anchor recipe, members hgb+et+mlp+lr, bank = context76/tiered62)

`disease_anchor_verify_v1.py` appends the disease columns to the accepted bank and
re-runs the exact shared-anchor gates (inner seeds 666/0/1; resplits 42/2026/31415),
with a **permuted-drug-map control** that preserves feature marginals.

| recipe | w | inner Mi ΔAUC / ΔAP | resplit Mi ΔAUC / ΔAP | joint gate |
|---|---|---|---|---|
| `bank` (accepted anchor) | 0.30 | +0.000156 / +0.000703 | +0.000318 / +0.000746 | pass |
| **`bank+disease`** | 0.30 | **−0.000079 / +0.000566** | +0.000607 / +0.001500 | **fail (Mi AUC < 0)** |
| `bank+control` (permuted) | 0.30 | +0.000225 / +0.001262 | +0.000423 / +0.001178 | pass |

Two readings:

1. **The real disease channel is not an improvement.** It raises MiDrug AP but its
   mean inner MiDrug AUC is negative at every weight; the single-seed +0.0004 seen
   before averaging was partition noise. This reproduces, on a brand-new data source,
   the exact pattern the project already documented for 87 graph mechanisms.
2. **The permuted control also passes the joint gate.** Adding five arbitrary columns
   to the bank and letting the four members refit is enough to move both metrics past
   the gate. The gate is therefore not a sufficient information test for
   bank-augmentation style changes; a permutation control has to travel with them, and
   here it falsifies the disease recipe.

## Why (consistent with earlier findings)

The global metric is 99.7% cross-drug: it asks whether a positive on a popular
drug/ncRNA outranks a negative on a rare one. External disease and perturbation
annotations are proxies for exactly the same popularity (study counts) and are
downstream of the RNA→target and expression channels the accepted bank already
carries, so they cannot supply orthogonal cross-drug information.

## Artifacts

- `build_channel_v1.py`, `CHANNEL_DIAGNOSTIC_V1.json` — disease bridge + diagnostic
- `perturb_connectivity_diag_v1.py` — perturbation connectivity diagnostic
- `disease_anchor_verify_v1.py`, `disease_anchor_metrics_v1.csv`,
  `disease_anchor_summary_v1.csv`, `DISEASE_ANCHOR_ASSESSMENT_V1.json` — full gate + control
- Raw data: `/data1/licheng/incrd-hg/data_external/ncrna_disease_20260917/`
