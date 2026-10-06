# Curation-provenance information channels for iNcRD-HG: three verified techniques

Date 2026-09-17. Scope: new *information* channels only — no neural weight is touched,
no model is replaced, and no held-out test source is opened. Every candidate is scored
against the accepted frozen TabICLv2 `tiered_0p5000` reference with the accepted
shared-anchor recipe

```
score = sigmoid((1-w) * logit(reference) + w * mean logit({HGB, ExtraTrees, MLP, LogReg}))
```

members fitted on the frozen `context76` bank, and evaluated on **seven splits**
(inner 3 inference seeds; development resplits 42/2026/31415; extra splits 7/77/777).
Each channel has a **label-preserving permutation control** that must be worse.

## Deliverable: three new techniques

| # | technique | what is new | MiDrug ΔAUC / ΔAP (inner) | LncDrug ΔAUC / ΔAP (inner) | gate | control |
|---|---|---|---|---:|---|:--:|
| 1 | **Curation-provenance co-membership** (`bank+prov`) | the curated record is a hyperedge `(RNA, drug, PMID, experimental context)`; only its `(RNA, drug)` projection was ever used. Adds study co-reporting and context co-occurrence. | **+0.00088 / +0.00237** | +0.00034 / +0.00073 | joint pass on all 7 splits | fails (`prov_perm`) |
| 2 | **Cross-task provenance sharing** (`pan_real`) | the drug side of the hyperedge also uses the *other* ncRNA type's training records; the RNA side stays own-task | **+0.00095 / +0.00278** | +0.00058 / +0.00115 | joint pass at w=0.1 | fails (`pan_perm`) |
| 3 | **External bibliometric exposure** (`bib`, corrected) | label-free PubMed entity volumes (`all` and cancer-context) plus a *curation-saturation* interaction against the cross-fitted positive degree | **+0.00031 / +0.00055** | +0.00049 / +0.00095 | tolerance pass at w=0.1 (one extra-split AP −0.00088) | fails (`bib_perm`) |

All deltas are against the **frozen `tiered_0p5000` reference** at `w=0.1` (the detailed
7-split tables below); the accepted shared-anchor recipe alone gives only
+0.00012/+0.00036 on the same inner split, so each channel adds on top of it. "joint pass" means
MiDrug up on inner + every resplit/extra split, and LncDrug AUC/AP not down (strict) or
not down by more than 0.001 (tolerance).

## 1. Curation-provenance co-membership — the main result

Source: `records.csv` (curation-level metadata of ncRNADrug `DR_Curated`:
`PMID`, `Throughput`, `Condition`, `Phenotype`, Cellosaurus-normalised disease, year).
Training rows are cross-fitted with the frozen fold ids; query rows use all training
positives. 14 features:

* study co-reporting — number / inverse-breadth-weighted count of training PMIDs whose
  hyperedge already contains both query entities through other pairs;
* entity study volume — distinct PMIDs, record counts, high-throughput fraction;
* experimental-context co-occurrence — IDF-weighted overlap of the cell-line /
  phenotype / disease tokens of `r`'s and `d`'s training records.

Full 7-split gate at `w=0.1` (both banks in `gates_v1.csv`; context76 shown):

| surface | MiDrug ΔAUC / ΔAP | LncDrug ΔAUC / ΔAP |
|---|---:|---:|
| inner (3 seeds) | +0.00088 / +0.00237 | +0.00034 / +0.00073 |
| resplits 42/2026/31415 | +0.00149 / +0.00335 | +0.00049 / +0.00128 |
| extra 7/77/777 | +0.00141 / +0.00291 | +0.00042 / +0.00168 |

At the accepted `w=0.3` it is larger and still positive on both tasks:
MiDrug inner +0.00199/+0.00575, LncDrug +0.00081/+0.00178.

**Reproduction.** The whole pipeline was rebuilt from the raw records into a fresh
directory; `logit_bank+prov_hgb/et` match the cached arrays to 0 and 2.6e-13
(`/tmp/opencode/prov_verify`, `.venv-tabicl`). The permutation control (`prov_perm`:
PMID/context tokens shuffled across records of the task, marginals kept) fails the joint
gate at every weight.

**It is not a popularity/degree effect.** The 14 features split into
`exposure` (entity study volume) and `collab` (pair co-membership + context). Refitting
the members on the cached features:

| subset | MiDrug inner ΔAUC / ΔAP | LncDrug inner ΔAUC / ΔAP | joint |
|---|---:|---:|---|
| `prov_exposure` only | +0.00014 / +0.00033 | **−0.00014 / −0.00176** | fail |
| `prov_shared` only (2 cols) | +0.00103 / +0.00286 | +0.00039 / +0.00085 | Mi up, Lnc dips < −0.001 on 1 extra split |
| `prov_ctxonly` only (6 cols) | +0.00005 / +0.00058 | +0.00042 / +0.00082 | fail (Mi flat) |
| `prov_collab` (shared+ctx) | +0.00088 / +0.00269 | +0.00039 / +0.00087 | tolerance pass at w=0.1 |

The entity-volume columns alone **hurt** LncDrug, so the gain comes from pair-level
study co-membership, not from another degree prior.

## 2. Cross-task provenance sharing

`pan` inherits the same features but the drug side aggregates both tasks' training
records (the accepted protocol already exposes the other ncRNA type as graph side
information); the RNA side remains own-task.

| surface | MiDrug ΔAUC / ΔAP | LncDrug ΔAUC / ΔAP |
|---|---:|---:|
| inner | +0.00095 / +0.00278 | +0.00058 / +0.00115 |
| resplits | +0.00156 / +0.00348 | +0.00041 / +0.00100 |
| extra | +0.00143 / +0.00312 | +0.00026 / +0.00028 |

Joint strict pass at w=0.1; `pan_perm` fails all weights.

## 3. External bibliometric exposure (corrected)

`pubmed_counts_resume.py` completed the label-free PubMed `esearch` volumes
(605 miRNA, 1322 lncRNA, 216 drugs, no pair queries). The v2 script of the same day is
**buggy** — it reads the accepted bank's columns 0/1 as raw degrees, but those columns
are standardised, so `log1p()` is NaN and the ExtraTrees member raises. `v3` computes
the cross-fitted positive degree itself and sanitises.

| surface | MiDrug ΔAUC / ΔAP | LncDrug ΔAUC / ΔAP |
|---|---:|---:|
| inner | +0.00031 / +0.00055 | +0.00049 / +0.00095 |
| resplits | +0.00032 / +0.00060 | +0.00046 / +0.00095 |
| extra | +0.00044 / +0.00073 | +0.00054 / +0.00043 |

Tolerance pass at w=0.1 (worst LncDrug AP cell −0.00088); `bib_perm` fails tolerance.

## Negative / neutral results recorded

* **LM semantic context matching** (MiniLM embeddings of phenotype/cell-line/evidence
  sentence/paper title, `text_*`): real and permuted text differ by little; `bank+text_*`
  fails, and `bank+prov+text_all_perm` (prov real, text permuted) passes with almost the
  same margin as the real. The text view is **neutral**; the pass comes from `prov`.
* **Evidence-polarity channel** (`evidence_polarity_v1.py`, resistance vs sensitivity
  curation per entity, motivated by DMGAT 2025): MiDrug inner ΔAUC negative and LncDrug
  extra negative — **rejected**.
* **Evidence-strength sample weighting** (`evidence_weight_v1.py`): tolerance pass only
  at w=0.1 `lowthr`, and it is *weaker* than the plain bank — not a technique.

## Directions and literature grounding

The channel is the qualifier/provenance layer of a hyper-relational knowledge graph:
each curated fact carries `(PMID, throughput, cell line, phenotype, disease)`, exactly
the "qualifiers" that StarE/GRAN-style models use and that our accepted graph discards.
Recent literature (`THOR`, arXiv 2602.05424, inductive hyper-relational link prediction;
`NEOGRAN`, BMC Bioinformatics 2026, traceable graph-text fusion) supports provenance as
a first-class signal; `DMGAT` (Brief. Bioinform. 2025) supports sensitivity curation —
though our polarity channel did not transfer. This is a *scientometric / provenance*
direction, distinct from the project's adopted levers (posw2 weighting, hypergraph/DGI,
two-sided degree prior, evidence-tiered pathway adapter, anchor shrinkage/fusion, TabDPT
transfer). It is still an "add new information" lever, so the distinction from the
accepted evidence-column direction is one of information source, not of algorithm.

## Honest limitations

* All splits are the project's reused development partitions; there is no new
  independent cohort. Gains are +0.0003 to +0.002 AUC / +0.0005 to +0.004 AP — real but
  small, consistent with a metric that is 99.7% cross-drug sampling propensity.
* Technique 3 needs the tolerance interpretation of "LncDrug must not drop" (worst cell
  −0.00088 AP); techniques 1–2 pass strict.
* The candidate must be applied through the existing member/fusion path, so it inherits
  the accepted recipe by construction.

## Artifacts

* `provenance_channel_v1.py`, `cells/` — technique 1 (cross-fitted features + real/perm logits)
* `provenance_pan_v1.py`, `cells_pan/` — technique 2
* `bibliometric_exposure_v3.py`, `cells_bib3/` — technique 3 (corrected); `v2` is the buggy draft
* `provenance_subset_v1.py`, `cells_subset/` — exposure vs co-membership decomposition
* `text_channel_v1.py`, `text_embed_v1.py`, `cells_text/` — neutral text view
* `evidence_polarity_v1.py`, `cells_pol/` — rejected polarity channel
* `evidence_weight_v1.py`, `cells_weight/` — sample-weighting screen
* `evaluate_generic_v1.py` — shared gate evaluator; gate tables `gates_v1.csv`, `pan_v1_gates.csv`, `text_v1_gates.csv`, `weight_v1_gates.csv`
* Raw data: `research/exposure_next_20260917/records.csv`; `data_external/bibliometrics_20260917/`
