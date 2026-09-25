## iNcRD-HG

This repository contains the source code for our paper "Semantic-Enhanced Heterogeneous Graph Learning for Identifying ncRNAs Associated with Drug Resistance". It includes the implementation of the proposed predictor, along with datasets and tutorials to help users reproduce and extend our work.

---

## Introduction

Identifying non-coding RNAs (ncRNAs) associated with drug resistance is crucial for understanding drug response mechanisms and discovering new therapeutic targets.  
We propose **iNcRD-HG**, a framework for predicting ncRNA–drug resistance associations. iNcRD-HG integrates molecular interactions, expression profiles, and drug structures to construct a biologically informative network. Through relation-type-aware message passing and an interpretability module, it captures complex contextual dependencies and uncovers cooperative ncRNA roles in drug resistance.

---

## Datasets

All datasets are available at: [Zenodo Link] (The data link will be made public upon the completion of the review process)  

- **Heterogeneous graph/**: Files for constructing the heterogeneous graph.  
- **Train-val-test-data/**: Training, validation, and test sub-datasets of ncRNA–drug resistance associations.  
- **Extract latent feature/**: Embeddings of ncRNA–drug resistance pairs extracted from iNcRD-HG.  
- **Case studies/**: Datasets used for case study experiments.  

---

## Environment Setup

- **OS**: Ubuntu 20.04.5  
- **GPU**: NVIDIA GeForce RTX 3060  
- **CUDA**: 11.3  
- **Python**: 3.8.8  
- **PyTorch**: 1.12.1  

Install the environment via:
```bash
conda env create -f iNcRD-HG-env.yml
```

## Main Scripts

- `drug_language_feature.py` + `ChemBERTa-zinc-base-v1.zip`: Extraction of drug features using ChemBERT.  
- `drug_similarity.py`: Computation of drug similarities.  
- `heter_data_prepare.py`: Construction of the heterogeneous graph and preparation of train/val/test datasets.  
- `find_best_params.py`: Parameter tuning on the validation dataset.  
- `predict_association.py` + `compare_base_GNN.py`: Training and testing of iNcRD-HG and baseline GNN models.  
- `graph_explain.py`: Computation of node and edge contributions in the heterogeneous graph.  
- `ablation_predict_association.py`: Training and testing for the ablation study.  
- `feature.py`: Visualization of ncRNA–drug resistance pairs under different strategies.  
- `predict_simulated_drug.py` + `compare_base_simulated_drug.py`: Training and testing under simulated novel drug scenarios.  
- `plot_case_drug.py`: Case study visualization.  

## Running Examples

Create the environment:
```bash
conda env create -f iNcRD-HG-env.yml
```

Train and test iNcRD-HG:
```bash
python predict_association.py
```

Compute node and edge contributions:
```bash
python graph_explain.py -e 'MiDrug' -f 'test' -i '0' -k 100 –p
```

Case study visualization:
```bash
python plot_case_drug.py -i '0' -k 100
```

## SubKNet / MPHGNN reproduction and transfer (2026-09-15)

Question: can incrd-hg be lifted to the miRNA level of the two current strong
mechanisms — **SubKNet** (per-link enclosing subgraph + trainable graph kernel;
JBHI 2024) and **MPHGNN** (metapath-guided `<miRNA-gene-gene-drug>` higher-order
semantics; BIB 2026) — without degrading lncRNA?

**MPHGNN reproduces.** `dgl==1.1.2+cu117` was installed into an isolated venv
(`/data1/licheng/incrd-hg/.venv-dgl`) and the official `main.py` run on its own
data/independent test set: **AUC 0.9743 / AUPR 0.9382** (seed 422) and
**0.9739 / 0.9364** (seed 0) vs the reported **0.978 / 0.937**. A PyG
reimplementation built earlier reached only 0.948/0.906; the gap is caused by
implementation and protocol differences (the official graph has only four
message-passing relations, gene-gene is used only for metapath instances, model
selection is by training loss, and test-time message passing includes the
test-positive edges), not by the paper.

**SubKNet is not yet reproduced.** The official model code (`RW_layer` graph
kernel + `DenseGCNConv` global branch) adapted to the same benchmark with cosine
similarity inputs and 25 epochs gives **0.948 / 0.906** (seeds 666/0, 5-fold) vs
its reported **0.9717 / 0.9341**; the adaptation is not the authors' data
pipeline, so this is not evidence against the paper.

**Transfer to incrd-hg still fails, including the end-to-end model.** Two tests
on the accepted inner partition both fail the joint MiDrug-up / LncDrug-not-down
gate. (1) A fixed label-free gene-mediated metapath channel (Reactome gene-gene
functional similarity + bidirectional `<RNA->gene1~gene2->drug>` features) added
to the accepted TabICLv2 branch: MiDrug **-0.000007 AUC / -0.000105 AP**,
LncDrug **-0.000096 / -0.000270**. (2) MPHGNN's own model code re-pointed at
incrd-hg's graph (RNA-gene / drug-gene / train RNA-drug message passing, gene-gene
for metapath instances): the pure GNN reaches only **0.9559/0.9167** (MiDrug) and
**0.9863/0.9608** (LncDrug) versus the reference **0.9626/0.9272** and
**0.9938/0.9871**; fusing it at weights 0.1-0.5 improves MiDrug at low weight but
drops LncDrug in every seed, so no weight passes. An ablation of the gene-gene
functional edges (real / shuffled / removed) leaves the pure GNN unchanged
(0.9566/0.9179 vs 0.9566/0.9179 vs 0.9568/0.9181), so the metapath gains nothing
beyond the direct RNA-target x drug-target pairing already encoded in the accepted
feature bank. The accepted TabICLv2 reference is unchanged.

See [mechanisms and sources](improve/research/mirna_sota_20260915/NOTES.md),
[the assessment](improve/research/mirna_sota_20260915/REPORT.md) and
[the sealed channel run](improve/results/metapath_channel_20260915/ASSESSMENT.json).

## General shared anchor shrinkage: MiDrug up, LncDrug not down (2026-09-15)

A single **task-shared, branch-free** post-hoc fusion lifts MiDrug while leaving
LncDrug non-decreasing on both the historical inner partition and the three
development resplits:

```
score = sigmoid((1-w)*logit(reference) + w*mean logit({HGB, ExtraTrees, MLP, LogisticRegression}))
members fitted on the frozen context76 bank; w = 0.30; identical for both tasks
```

| Surface | MiDrug dAUC | MiDrug dAP | LncDrug dAUC | LncDrug dAP | Gate |
|---|---:|---:|---:|---:|:--:|
| inner (3 seeds) | +0.000145 | +0.000666 | +0.000491 | +0.001008 | pass |
| resplits (42/2026/31415) | +0.000315 | +0.000728 | +0.000400 | +0.001394 | pass |

Per seed/split and the label-permutation control are in the findings. The recipe
uses the same members and the same weight for MiDrug and LncDrug — no
`if task == ...` branch (the earlier `anchor_mixed`, which added a TabDPT weight
for MiDrug only, is deliberately not used). 26 of 60 evaluated shared recipes pass
both surfaces, so the choice is not a knife edge.

See [findings](improve/CONTEXT_ANCHOR_FINDINGS.md), runner
(`/data1/licheng/incrd-hg/.venv-tabicl/bin/python -m improve.context_anchor_verify`)
and [metrics](improve/results/context_anchor_verify_20260915/summary.csv).

## Frozen TabDPT transfer onto evidence/context banks (2026-09-14)

A second tabular foundation model, **TabDPT 1.3**, is used zero-shot on the
accepted feature banks and combined with the frozen TabICLv2 reference by fixed
equal-logit weights. Three pre-registered techniques improve **both** tasks on
the historical inner partition; the neural backbone is unchanged.

| Task | Accepted reference AUC / AP | Recommended ensemble AUC / AP |
|---|---|---|
| MiDrug | 0.962593 / 0.927204 | **0.963809 / 0.929416** |
| LncDrug | 0.993750 / 0.987141 | **0.994792 / 0.989575** |

Techniques: (1) cross-architecture ensemble on the 36 graph columns
(`+0.00013/+0.00064` MiDrug, `+0.00024/+0.00112` LncDrug); (2) TabDPT on the
62-column evidence-tiered bank (`+0.00131/+0.00244` MiDrug,
`+0.00097/+0.00201` LncDrug); (3) TabDPT on the 76-column DepMap
expression/context bank (`+0.00155/+0.00299` MiDrug, `+0.00076/+0.00176`
LncDrug). The recommended configuration is the equal-logit mean of the
reference and techniques (2) and (3) (`+0.00122/+0.00221` MiDrug,
`+0.00104/+0.00243` LncDrug); every inference seed is positive and a row-level
paired bootstrap excludes zero. EXAONE-Tabular, TabDPT retrieval contexts and
cross-task multi-task pooling were explored and rejected.

See [method and reproduction](improve/FOUNDATION_TRANSFER.md),
[findings](improve/FOUNDATION_TRANSFER_FINDINGS.md) and
[the sealed evaluation](improve/results/foundation_transfer_20260914/eval/REPORT.md).

## From-scratch anchor shrinkage, verified on independent resplits (2026-09-14)

Three new non-foundation-model techniques improve **both** tasks on the historical
inner partition **and** on the three independent development resplits. They shrink
the accepted reference toward a low-weight logit/rank mix with from-scratch
scorers fitted on the same evidence bank (HGB, ExtraTrees, RandomForest, MLP,
logistic). Label-permutation controls are negative, so the gain is label-dependent
information, not variance reduction.

| Technique | Inner gate | 3 resplits joint | MiDrug ΔAP (inner / resplit) | LncDrug ΔAP (inner / resplit) |
|---|:---:|:---:|---:|---:|
| `anchor_mixed` (task-adaptive multi-anchor: pool + TabDPT for MiDrug, pool for LncDrug) | pass | **pass (6/6)** | +0.000522 / +0.001323 | +0.000341 / +0.001350 |
| `anchor_logit` (logit pool shrinkage) | pass | **pass (6/6)** | +0.000209 / +0.001029 | +0.000341 / +0.001350 |
| `anchor_rank` (rank aggregation) | pass | **pass (6/6)** | +0.000183 / +0.000779 | +0.000255 / +0.000823 |
| `anchor_pu` (PU observation anchor) | pass | LncDrug −0.0002 on 1 split | +0.000182 / +0.000610 | +0.001115 / +0.000541 |

125 member-subset recipes over 8 from-scratch learners and 2 feature banks pass both
surfaces, so the fixed weights are not a knife-edge choice. Transductive query-graph
smoothing and pseudo-label self-training were also tested and fail on both surfaces.

This is the **out-of-class diversity** case left open by *Ensembling Tabular
Foundation Models — A Diversity Ceiling And A Calibration Trap* (arXiv:2605.18696):
a TFM-only pool is near-redundant, while shrinking the over-confident TFM reference
toward classical learners recovers a transferable gain. It matches the finding that
TFMs are high-AUC but poorly calibrated (arXiv:2605.28554).

**Correction to the TabDPT section below:** re-building the same feature banks for
the three independent resplits and scoring them with the official TabDPT shows
the accepted transfer does **not** generalize — MiDrug stays mildly positive but
LncDrug drops by −0.0017 to −0.0040 AUC and −0.005 to −0.017 AP in every resplit.
That technique is an artifact of the repeatedly used inner partition.

See [findings and controls](improve/ANCHOR_SHRINKAGE_FINDINGS.md), runner
(`python -m improve.anchor_shrinkage report`),
[resplit verification](improve/resplit_foundation_verify.py) and
[sealed results](improve/results/anchor_shrinkage_20260914/ASSESSMENT.json).

## Non-foundation-model techniques on the accepted reference (2026-09-14)

Question: besides tabular (foundation) models, do other technology families
improve the accepted reference? Eight non-pretrained scorers (gradient-boosted
trees, ExtraTrees, RandomForest, MLP, kNN retrieval, logistic, RBF-SVM,
NB/LDA), three named non-tabular techniques (distance-weighted retrieval,
transductive label propagation, PU prior-reweighted GBDT) and every low-weight
arm combination were tested on the frozen evidence banks and then re-checked on
the three independent development resplits.

**No non-tabular technique robustly improves MiDrug while preserving LncDrug.**
A broad region of low-weight blends passes the strict gate on the repeatedly used
inner partition (e.g. `ref + 0.1×MLP` `+0.0001/+0.0005` MiDrug, positive on all
six cells), but it does **not** replicate on the three resplits. The closest
non-tabular candidate, `pu_gbdt` at weight 0.1, raises MiDrug on all three
resplits but drops LncDrug AP by 0.001 on one, so it is not a joint improvement.
Arm-level fusion on the resplits (including the non-tabular GNN arm) never beats
the `tiered` reference.

The verified gains therefore remain (1) a *stronger* predictor (TabDPT transfer)
and (2) *new information* (evidence/context columns), not non-tabular model swaps.

See [findings](improve/NONTABULAR_TRANSFER_FINDINGS.md), runner
(`python -m improve.nontabular_transfer report`) and
[sealed results](improve/results/nontabular_transfer_20260914/RESULTS.json).

## Shared evidence-tiered pathways with frozen TabICLv2 (2026-09-13)

The uploaded LncTarD and miRTarBase target evidence, together with retained
lower-support ENCORI observations, now support a shared pathway adapter.
It separates functional, binding, and association evidence and uses the same
50% blend with the verified graph-context predictor for both tasks. This
configuration passed the inner gate and then the fixed historical outer check;
AUC, AP, and macro drug AP increased in all three inference seeds for both tasks.

| Task | Graph-context AUC / AP | Evidence-tiered AUC / AP | Macro drug AP, before → after |
|---|---|---|---|
| MiDrug | 0.959934 / 0.925057 | **0.960638 / 0.926064** | 0.717259 → 0.739663 |
| LncDrug | 0.985742 / 0.968844 | **0.986929 / 0.971320** | 0.862695 → 0.897454 |

These are means of per-seed metrics on reused historical validation data, with
no independent test claim. All 105 formal fitted states were saved, reloaded,
and replayed with zero prediction discrepancy. The neural backbone is frozen.
DepMap 25Q2 expression was also aligned and tested, but its additional context
features were not selected over the evidence-tiered adapter.

See [results and limitations](improve/TABICL_PATHWAY_CONTEXT_FINDINGS.md),
[data, architecture, and reproduction](improve/TABICL_PATHWAY_CONTEXT.md), and
[the accepted configuration and saved states](improve/results/tabicl_pathway_context_20260913/CANDIDATE.json).

## Shared CARE-TabICL adaptation experiments (2026-09-13)

Three rounds implemented learned conditional relation attention and then
support-rate normalization with teacher-order retention. **No candidate met the
joint requirement of improving MiDrug while preserving LncDrug.** The one
inner-selected candidate evaluated on the historical outer partition improved
MiDrug AUC/AP by 0.000081/0.000133, but reduced LncDrug AUC by 0.000035.
All 84 saved context states were replayed with zero prediction discrepancy.
The matched graph-context TabICLv2 configuration below remains the accepted
reference; failed candidates are not enabled by the prediction entry point.

See [the final findings](improve/TABICL_CARE_FINDINGS.md),
[CARE architecture and experiments](improve/TABICL_CARE.md), and
[CARE-RO](improve/TABICL_CARE_RO.md).

## TabICLv2 with matched graph contexts (2026-09-12)

The accepted development configuration improves global AUC/AP over the verified
TabICLv2 baseline by matching graph support and averaging query views. Each task
uses its internally selected component at 50% weight with the original predictor.

| Task | Original TabICLv2 AUC / AP | Selected graph-context AUC / AP |
|---|---|---|
| LncDrug | 0.985403 / 0.968115 | **0.985742 / 0.968844** |
| MiDrug | 0.959524 / 0.924198 | **0.959934 / 0.925057** |

Both metrics improve under all three inference seed labels; all 72 fitted
context states were reloaded and replayed. They share one pretrained checkpoint.
Per-drug macro AP decreases by 0.015658 for LncDrug and 0.001313 for MiDrug.
These are results on repeatedly used historical validation data, with no test
inference or independent generalization claim.

See [mechanism, controls, reproduction and numerical audit](improve/TABICL_GRAPH_CONTEXT.md)
and [the full report](improve/results/tabicl_graph_context_20260912/REPORT.md).
Earlier [TabICLv2 + TabM blending](improve/TABICL_TABM_FUSION.md) and
[task-episodic parameter adaptation](improve/TABICL_ADAPTATION.md) did not improve
both tasks over standalone TabICLv2.

## LoMETab + per-drug group ranking backbone (2026-09-12)

Two replacement rounds on the frozen TabM pipeline. **Neither beats the matched
TabM anchor.** Round 1 (2x2 factorial: network {TabM,
[LoMETab](https://arxiv.org/abs/2605.14365)} x objective {BCE, BCE + per-drug
group ranking}) was consistently negative on global AUC/AP. Round 2 searched
LoMETab's (rank, sigma_init) grid {1,2,4,8} x {0.01,0.1,0.5,1.0} x two learning
rates: the inner-selected high-diversity configs (r=4/σ=1.0 for LncDrug, r=8/σ=0.5
for MiDrug) narrowed the AP gap to about -0.3pp but still lost on all metrics at
0/3 seeds. The `tabm` anchor reproduces the official numbers exactly, validating
both pipelines.

See [LOMETAB_GROUP.md](improve/LOMETAB_GROUP.md),
[round 1](improve/results/lometab_20260912/REPORT.md) and
[round 2 grid](improve/results/lometab_grid_20260912/REPORT.md).

## Replacement backbone experiments (2026-09-10)

The experimental replacement at this stage was **graph-derived pair features + TabM
(ICLR 2025)**. It has its own training and prediction entry points, frozen
configuration files and verified checkpoints. The original scripts remain
available for reproduction and comparison.

On the existing outer validation partition, the three-seed mean AP changes from
0.943892 to 0.966260 for LncDrug and from 0.919169 to 0.921461 for MiDrug.
Global AUC also improves on both tasks. LncDrug's per-drug macro AP decreases;
these results concern a reused validation partition, not a new independent test.

See [usage, architecture and fixed configurations](improve/GRAPH_TABM.md),
[the four-architecture comparison](improve/results/backbone_comparison_20260910/REPORT.md),
and [the literature check](improve/research/backbone_replacement_20260910/RESEARCH.md).

```bash
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m improve.predict_backbone \
    --task MiDrug --drug-index 0 --out mi_drug0_scores.csv --device cuda:0
```

## Heterogeneous graph transformer backbone (2026-09-11)

A structural-augmented HGT (relation-aware attention + per-relation degree
encodings + cross-fitted structural pair features) was implemented and run on
the same frozen protocol as a third replacement candidate. It does **not**
cleanly beat the base on both global metrics (LncDrug AP +0.0040, MiDrug AUC
+0.0008; other cells flat-negative) and is still below the TabM replacement on
global AUC/AP. It does recover part of LncDrug's per-drug macro AP relative to
TabM (+0.051). This confirms, on the same protocol, that learnable relational
attention did not convert into global-ranking gains in this comparison;
TabM was retained at that stage. The later TabICLv2 results are listed above.

See [GRAPH_TRANSFORMER_BACKBONE.md](improve/GRAPH_TRANSFORMER_BACKBONE.md) and
[the verified experiment](improve/results/ragt_backbone_20260911/REPORT.md).
