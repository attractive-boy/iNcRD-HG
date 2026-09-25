# Findings: frozen TabDPT transfer onto evidence/context banks (2026-09-14)

**The later three-resplit confirmation failed the shared-task condition.**
The historical-inner improvements below did not persist for LncDrug. Against
each resplit's accepted `tiered` reference, the fixed three-way ensemble had
mean MiDrug ΔAUC/AP **+0.000515596 / +0.001495800**, but LncDrug
**−0.001606841 / −0.008562369**; LncDrug lost both metrics in all three
resplits. All six evaluated recipes failed the preset common gate. These are
overlapping development partitions, not independent validation cohorts. See
[resplit assessment](results/resplit_foundation_verify_20260914/ASSESSMENT.json).
The earlier recommendation below is retained as historical context and is
**not a current safe replacement recommendation** or a validated additional
innovation.

Several related TabDPT configurations improve **both** tasks over the accepted
TabICLv2 reference on the historical inner partition, while keeping the frozen
backbone untouched. These are variants within one tabular-transfer family,
not three independent technical innovations or independent validation studies.
All numbers are seed 666/0/1 means unless marked
"per-seed". Full precision is in
[eval/REPORT.md](results/foundation_transfer_20260914/eval/REPORT.md) and
[eval/metrics.csv](results/foundation_transfer_20260914/eval/metrics.csv).

**Reference** = accepted `0.5*graph_context + 0.5*tiered_evidence` TabICLv2,
MiDrug `0.962593 / 0.927204` (AUC/AP), LncDrug `0.993750 / 0.987141`.

## Related configurations with verified historical-inner metrics

| # | Technique | MiDrug ΔAUC / ΔAP | LncDrug ΔAUC / ΔAP | Gate |
|---|---|---|---:|---|
| 1 | **Cross-architecture ensemble** `ensemble_ref_base36`: `sigmoid(0.5*logit(reference) + 0.5*logit(TabDPT))` on the 36 graph columns | +0.00013 / +0.00064 | +0.00024 / +0.00112 | pass |
| 2 | **TabDPT on the 62-column evidence-tiered bank** (`tdp_tiered62`) | **+0.00131 / +0.00244** | **+0.00097 / +0.00201** | pass |
| 3 | **TabDPT on the 76-column DepMap expression/context bank** (`tdp_context76`) | **+0.00155 / +0.00299** | **+0.00076 / +0.00176** | pass |
| — | **Equal-logit ensemble of reference + `tdp_tiered62` + `tdp_context76`** (recommended) | **+0.00122 / +0.00221** | **+0.00104 / +0.00243** | pass |

- Technique 1 is the first accepted use of a **different tabular foundation
  model** in the pipeline. TabDPT alone on the base 36 columns is flat
  (`MiDrug 0.962110/0.927289`, `LncDrug 0.993103/0.986398`, ineligible), but its
  equal-logit ensemble with the reference improves every seed on both tasks; it
  also contributes diversity to every other ensemble.
- Configurations 2 and 3 are the transfer of the accepted evidence bank (`tiered`,
  26 columns) and the DepMap expression bank (`context`, 14 additional columns)
  to the TabDPT backbone. Each is a standalone, single-model improvement on both
  tasks; all three seeds are positive.
- Recommended configuration: `sigmoid((logit(ref) + logit(tdp_tiered62) +
  logit(tdp_context76)) / 3)`.
  MiDrug per-seed ΔAP `+0.00212 / +0.00224 / +0.00228`, LncDrug
  `+0.0023694 / +0.0024612 / +0.0024707` (all positive).

## Per-seed gate detail (recommended ensemble)

| Task | Seed | ΔAUC | ΔAP |
|---|---:|---:|---:|
| MiDrug | 666 | +0.0011830 | +0.0021217 |
| MiDrug | 0 | +0.0011759 | +0.0022365 |
| MiDrug | 1 | +0.0012892 | +0.0022781 |
| LncDrug | 666 | +0.0010525 | +0.0023694 |
| LncDrug | 0 | +0.0010369 | +0.0024612 |
| LncDrug | 1 | +0.0010369 | +0.0024707 |

The earlier version quoted bootstrap intervals without a corresponding saved
bootstrap artifact in the evaluation directory. Those intervals are withdrawn
pending reproducible re-computation. Positive differences on three inference
seeds from one reused partition do not establish independent generalization or
statistical significance; pair-level resampling also ignores shared endpoints.

## Negative and auxiliary controls (explored, not accepted)

| Direction | Result | Why it failed |
|---|---|---|
| TabDPT alone on the 36 base columns | MiDrug AUC `-0.00048`, LncDrug AUC `-0.00065`/AP `-0.00074` | standalone model fails the joint condition; its fixed equal-logit ensemble passes the historical-inner gate as shown above |
| EXAONE-Tabular v1 (20.8M CAST) on base 36 | MiDrug `0.96194/0.92637`, LncDrug `0.99193/0.98321`; 0.5 ensemble reduced LncDrug AUC `-0.00056` | third architecture is not complementary here |
| TabDPT **retrieval** context reduction (`context_size=512`) on `tiered62` | LncDrug `0.99036/0.97758` | per-query nearest-neighbour context discards useful support rows |
| Cross-task multi-task TabDPT (pooled Mi+Lnc support + task indicator) | LncDrug `0.99344/0.98481` vs single-task `0.99478/0.98932` | pooling distorts the task-specific support distribution |
| TabDPT on the 5 graph-context query-view average | MiDrug `0.96169/0.92554` | masked-graph views add noise for TabDPT |

## References for the frontier models

- **TabDPT** — *TabDPT: Scaling Tabular Foundation Models* (arXiv:2410.18164);
  checkpoint `Layer6/TabDPT`, revision `a5ca6e01c0fa09ec68c73e958e5199d1932abb3a`,
  SHA-256 `97dc3b60…`. Runs in the isolated `.venv-frontier` environment.
- **EXAONE-Tabular v1** (arXiv:2608.25774; CAST architecture, 20.8M params) —
  checkpoint `LG-AI-Research/EXAONE-Tabular`; downloaded and evaluated, rejected.
- **MITRA v2** (`autogluon/mitra-classifier-2`) — checkpoint download returned
  HTTP 403; not evaluated.

The prior literature scan that identified these models is retained at
[research/frontier_after_validation_20260914](research/frontier_after_validation_20260914/REPORT.md)
and [research/frontier_joint_posterior_20260914](research/frontier_joint_posterior_20260914/).

## Boundaries

- Historical inner partition, repeatedly used in development; seeds are
  inference randomness, not independent splits. No outer/test/source-holdout
  array is evaluated. `accepted_reference_replaced = false`.
- TabDPT is a public checkpoint whose pretraining prior-data are not controlled
  here. The augmented columns are annotation/model priors, not measured
  ncRNA regulation. `0` labels remain sampled-unobserved pairs.
- Model selection is limited to the pre-registered recipes above; no weight grid
  was searched after metrics were seen.
