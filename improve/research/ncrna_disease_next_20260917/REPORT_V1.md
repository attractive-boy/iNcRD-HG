# New external information channels for iNcRD-HG: results and blocker

Date 2026-09-17. Independent re-verification plus two new external data families,
tested on both the improve/ frozen reference and the released author protocol.
No default model replaced; no reserved test source opened.

## Deliverable checklist (goal: >=3 new, innovative, verified techniques; MiDrug up, LncDrug not down)

| candidate | surface | MiDrug | LncDrug | control test | status |
|---|---|---|---|---|---|
| ncRNA-disease x drug-disease bridge | author S_test | +0.0011 AUC / +0.0010 AUPR | up | above 10-ctrl mean, above max 3/4 cells | **marginal, not robust** |
| ncRNA-disease x drug-disease bridge | improve/ inner | dAUC < 0 | flat | — | fail |
| drug-target-gene-disease bridge | author | +0.0011 AUC / +0.0011 AUPR | — | below ctrl max 0/4 | fail |
| drug-perturbation x miRNA-target | author | +0.0008 AUC / +0.0008 AUPR | — | below ctrl max 0/4 | fail |
| disease-similarity (shared-drug profiles) | author | +0.0011 AUC / +0.0009 AUPR | — | above ctrl max 2/4 | fail |
| disease + mechanism combined | author | +0.0008 AUC / 0.0000 AUPR | — | below ctrl max 0/4 | fail |

The goal requires three robust techniques; none is robust, and only the first is
even a candidate.

## What was newly established

1. **Independent confirmation of the ceiling.** On the frozen MiDrug validation
   set the metric is 99.7% cross-drug; a label-free feature scan gives residual
   correlation ~0 against the reference; per-drug recalibration destroys AUC
   (0.96 -> 0.54). The metric is a sampling-propensity / drug-prior metric.
2. **Two new external data families were acquired and audited** (RNADisease v4,
   HMDD v4, CTD chemicals-diseases and genes-diseases 129.7M rows, Enrichr
   LINCS/CMAP perturbation gene sets), with an RNA identity crosswalk for
   miRNA/lncRNA/drug.
3. **A permutation-control null was established for bank-augmentation changes.**
   Adding four to nine *permuted* annotation columns to the non-GNN member and
   letting it refit lifts the author-protocol MiDrug blend to 0.9590-0.9594 test
   AUC (~+0.001 over the accepted blend) regardless of information. Most real
   channels fall inside this null. This is the same "generic residual gain"
   pattern the project already measured for 87 graph mechanisms on the improve/
   reference.
4. **One marginal candidate**: the disease bridge exceeds the 10-control mean by
   0.5-3.4 sd and the control maximum in 3/4 cells (AUC val is a 0.0001 tie).
   With a pair-specific-only variant it drops below the control max on test.

## Why this is a blocker, not just difficulty

The requested end state is three techniques that robustly raise MiDrug global
AUC/AP while preserving LncDrug, in directions other than the adopted ones. The
global metric is dominated by the sampling propensity (drug/ncRNA popularity);
every external annotation tested is a proxy for the same popularity or is
downstream of the already-used target/expression channels, so its information
component is at most a sub-thousandth, inside the ensemble-perturbation null.
Reaching a robust gain needs one of:

- relaxing constraint 3 to allow a stronger propensity estimator or the
  project's robust directions (stronger predictor, new evidence columns,
  fusion/shrinkage); or
- a new independent evaluation cohort / intervention data where the metric is
  not a bare propensity measure.

Neither is available inside the current data and constraints.

## Artifacts

- `FINDINGS_V1.md`, `CHANNEL_DIAGNOSTIC_V1.json` — improve/ reference diagnostics
- `AUTHOR_DISEASE_FINDINGS_V1.md`, `AUTHOR_DISEASE_ASSESSMENT_V1.json`
- `author_disease_gbdt_v1.py`, `author_external_gbdt_v1.py`,
  `external_channels_v1.py`, `diseasesim_channel_v1.py`,
  `build_gene_disease_v1.py`, `gene_disease_v1.json`
- Raw data: `/data1/licheng/incrd-hg/data_external/ncrna_disease_20260917/`
