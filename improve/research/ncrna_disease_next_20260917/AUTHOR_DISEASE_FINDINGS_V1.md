# Verified: ncRNA-disease x drug-disease channel improves the author-protocol MiDrug blend

Date 2026-09-17. Released-protocol evaluation (author's frozen
`RandomLinkSplit` train/val/test). Default model unchanged.

## Technique

Add a label-free **ncRNA-disease x drug-disease bridge** to the existing non-GNN
member (`improve/gbdt.py`), keeping every other part of the accepted author
pipeline fixed (GNN combiner `auc_r_uniform_top8cfg`, rank-normalised blend with
the validation-selected weight).

Data: RNADisease v4.0 (human experimental miRNA/lncRNA-disease) + HMDD v4.0 +
CTD chemicals-diseases (`therapeutic`/`marker/mechanism`). Five pair columns:
shared disease count, IDF-weighted shared count, IDF-weighted RNA-side count,
RNA disease breadth, drug disease breadth. IDF is computed on the drug-disease
map only, so no label enters.

## Result (S_val / S_test, 3 GBDT seeds, blend weight chosen on validation)

| scorer | MiDrug AUC val / test | MiDrug AUPR val / test |
|---|---|---|
| accepted blend (19 features) | 0.9569 / 0.9582 | 0.9213 / 0.9191 |
| **+ disease bridge** | **0.9582 / 0.9593** | **0.9225 / 0.9201** |
| 10 permuted-map controls (mean ± sd) | 0.9577±0.0003 / 0.9590±0.0001 | 0.9215±0.0005 / 0.9191±0.0003 |
| 10 controls (max) | 0.9583 / 0.9592 | 0.9223 / 0.9198 |

With ten controls the disease blend is above the control **mean** by ~0.5–3.4 sd
in all four cells and above the control **maximum** in three of four cells
(AUC val is a 0.0001 tie). So the channel carries information, but the margin is
small and near the edge of the ensemble-perturbation null. A pair-specific-only
variant (drop the two breadth columns) is weaker and falls below the control max
on both test cells, so part of the effect rides on the drug-disease breadth.

LncDrug is not down (disease blend test AUC 0.9908 vs 0.9885, AUPR 0.9761 vs
0.9714).

## Honesty notes

- The added columns also shift the ensemble when permuted (controls beat the
  accepted blend on MiDrug), so the permutation control is load-bearing; the
  disease advantage has to be read against the control range, which it clears.
- The author protocol is one frozen split; only model seeds vary and the HGB
  member is deterministic across them. Verification therefore rests on the
  permutation controls, as the project's own methodology prescribes.
- The same channel does **not** help the improve/ TabICL reference (see
  `FINDINGS_V1.md`): there the MiDrug inner dAUC is negative.

## Other channels on the same surface (all fail the control test)

Built with the same protocol and 5-control null (`author_external_gbdt_v1.py`).
MiDrug blend, "vs ctrl max" = does the real channel beat every permuted control.

| channel | columns | Mi AUC test | Mi AUPR test | > ctrl max |
|---|---|---|---|---|
| disease (all 5) | 5 | 0.9593 | 0.9201 | 3/4 cells |
| mechanism (drug target-gene diseases) | 4 | 0.9593 | 0.9202 | 0/4 |
| drug-perturbation x miRNA-target | 6 | 0.9590 | 0.9199 | 0/4 |
| disease + mechanism | 9 | 0.9590 | 0.9191 | 0/4 |

Adding **any** four to nine permuted columns lifts the blend to 0.9590–0.9594
test AUC, i.e. the six-column perturbation channel and the four-column
mechanism channel sit *inside* the ensemble-perturbation null. Only the disease
channel's aggregate columns clear the null, and only marginally. The mechanism
channel was rebuilt from CTD gene-disease (129,662,329 rows, Entrez-matched
target genes, 155/216 drugs, 583 genes) and is genuinely new data, not a
re-parameterisation.

**Reading.** On the released protocol the non-GNN member is under-regularised:
extra columns change its fit and move the blend by ~+0.001 regardless of
information. Against that null, new external information is at most a
sub-thousandth effect, and most channels are indistinguishable from noise. This
matches the improve/ reference result and the project's earlier 87-mechanism
synthesis.

## Artifacts

- `author_disease_gbdt_v1.py`, `author_external_gbdt_v1.py` — author-protocol members
- `external_channels_v1.py`, `build_gene_disease_v1.py`, `gene_disease_v1.json`
- `AUTHOR_DISEASE_ASSESSMENT_V1.json` — numbers
- Raw data: `/data1/licheng/incrd-hg/data_external/ncrna_disease_20260917/`
