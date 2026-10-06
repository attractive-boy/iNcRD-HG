# PHL manuscript figures — specs and legends

Built by `make_figs.py` (matplotlib; run with `python3.8`, the only interpreter on this host that has matplotlib):

```bash
cd improve/research/plmf_warm_20260922/fig
python3.8 make_figs.py          # writes Fig*.pdf (vector), Fig*.tif (600 dpi RGB LZW) and preview_Fig*.png (200 dpi)
```

| file | size | format |
|---|---|---|
| Fig1_PHL_architecture.{pdf,tif} | 178 × 104 mm (double column) | vector PDF, fonts embedded (TrueType); TIFF 600 dpi, 4204 × 2456 px |
| Fig2_degree_diagnostics.{pdf,tif} | 178 × 90 mm (double column) | vector PDF, fonts embedded (TrueType); TIFF 600 dpi, 4204 × 2125 px |

Artwork conventions (OUP Bioinformatics): drawn at final print width; white background; Arial-class sans
(Liberation Sans, metric-identical to Arial, is what gets embedded); 6.5–7 pt text, 7.5 pt bold panel headings,
10 pt bold panel letters; lines 0.3–1 pt; English only; no title, caption, footnote or result numbers inside the
artwork; node identity is carried by shape + text label, never by colour alone. If the journal asks for EPS:
`pdftocairo -eps Fig1_PHL_architecture.pdf Fig1_PHL_architecture.eps`.

Numbers behind Fig. 2 come from `../REPORT_WARM_PHL.md` §1.1 and §4.3 (`error_analysis.py`, seed 42, paper 5-fold
protocol). Earlier drafts are archived: `slides_v0/` (16:9 slide deck, rejected), `pptx_v1/` (python-pptx
journal-size draft and its script `make_phl_fig.py`, superseded by `make_figs.py`).

## Figure 1. Overview of the provenance-hypergraph heuristic learner (PHL).

**(A)** Inputs. Training positives are the miRNA–drug associations of the training fold of the PLMF-MDA benchmark
(MDR or MDS); the dashed cell marks a held-out query pair. Curation provenance is taken from iNcRD-HG: every
association carries the PMID of the report and the cell line / phenotype / disease terms of that report.
Label-free miRNA knowledge comprises the HMDD v4 miRNA–disease profile, miRNA name and seed families, and the miRNA
degree in the other phenotype matrix. **(B)** Relational hypergraph built exclusively from support positives.
Solid edges are training positives; dotted blue edges are kNN content edges from miRNA / drug similarity; the dashed
edge with the question mark is the query pair, which is never part of its own support. Orange shading is a study
hyperedge: one per PMID, joining every pair that study reports, so (m₁, d₂) becomes reachable through co-reported
pairs. Purple shading is a context hyperedge joining entities that share a cell line, phenotype or disease term.
Green relations link each miRNA to its family (hexagon), its HMDD diseases (diamond) and its degree in the other
phenotype matrix (bar glyph), supplying information for miRNAs with zero or few support edges. **(C)** Motifs
scored for a query pair (m, d), grouped into the 128 heuristic columns: 85 graph-context and diffusion features
(degree, common neighbours, 3–7-step paths, personalised PageRank, Katz, rank-16 SVD, kNN support), 24 provenance
features (PMID bridging |P(m) ∩ P(d)|, co-study transfer, context-term overlap, record statistics) and 19 miRNA-prior
features (family and seed mates of m linked to d, HMDD disease bridge, cross-phenotype degree). **(D)** Linear
readout. Columns are transformed by signed log1p and standardised, then scored by a single logistic-regression head
(≈130 parameters). Training pairs are cross-fitted with K = 10 inner folds so that a pair never enters its own
support; test pairs use all training positives as support. No heuristic ever reads a test label.

## Figure 2. The warm-protocol residual concentrates on sparse miRNAs, and so does the PHL gain.

Test positives are bucketed by the degree of their miRNA in the training support graph (paper 5-fold protocol,
seed 42). Ranking loss of a positive is the fraction of sampled negatives ranked above it; its mean over positives
equals 1 − AUC.
**(A, C)** Share of test positives versus share of total ranking loss per degree bucket for MDR (A) and MDS (C).
miRNAs with support degree ≤ 3 hold 40% of the positives but 55–57% of the loss.
**(B, D)** Mean ranking loss per positive for the baseline (ctx + diff × logistic regression) and PHL on MDR (B) and
MDS (D). PHL lowers the loss by 8–15% in the degree 1–7 buckets and leaves the 16+ bucket unchanged.
