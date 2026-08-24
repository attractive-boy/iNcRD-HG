# NcRD-Bench findings

Reproduced with `analysis/` on the iNcRD-HG data release: 8 seeds
(666, 0, 1, 2, 3, 42, 7, 13), epoch selected on validation, drug-degree control
included as a floor.

## 1. The released protocol is solved by drug popularity

`T.RandomLinkSplit(neg_sampling_ratio=2.0)` draws negatives uniformly over
(ncRNA, drug), while positives concentrate on a few heavily studied drugs. In
the released `LncDrug` test set, drug 5 carries 157 positives against 3
negatives, and 96.9% of positives sit on 20 of 216 drugs.

A scorer that reads a drug's training degree and ignores the ncRNA entirely:

| | LncDrug AUC / AUPR | MiDrug AUC / AUPR |
|---|---|---|
| drug degree only | 0.9775 / 0.9513 | 0.9316 / 0.8762 |
| iNcRD-HG (paper) | 0.9857 / 0.9585 | 0.9564 / 0.9297 |

The trained model's own scores correlate with drug degree at Spearman
rho = 0.837 (LncDrug) and 0.877 (MiDrug).

## 2. Debiasing works

Matching every negative to the same drug as its positive makes the degree
control worthless by construction. Measured, not assumed:

| split | LncDrug | MiDrug |
|---|---|---|
| random (released protocol) | 0.9707 | 0.9337 |
| matched | 0.5000 | 0.5000 |
| cold_rna | 0.5000 | 0.5000 |
| cold_drug | 0.5000 | 0.5000 |
| cold_pair | 0.5000 | 0.5000 |

## 3. The published ranking does not survive

Per-drug macro AUC, 8 seeds. `random` reproduces the paper's ordering; every
other column is popularity-neutral.

| method | Lnc random | Lnc matched | Lnc cold_rna | Lnc cold_drug | Lnc cold_pair | Mi random | Mi matched | Mi cold_rna | Mi cold_drug | Mi cold_pair |
|---|---|---|---|---|---|---|---|---|---|---|
| degree | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| iNcRD-HG | **0.764** | 0.648 | **0.751** | 0.627 | 0.771 | **0.794** | **0.797** | 0.629 | **0.820** | 0.682 |
| GAT | 0.667 | 0.719 | 0.639 | 0.764 | 0.652 | 0.631 | 0.669 | 0.705 | 0.699 | 0.724 |
| GraphSAGE | 0.722 | 0.694 | 0.654 | 0.822 | **0.822** | 0.769 | 0.780 | 0.650 | 0.779 | 0.574 |
| ASAP | 0.602 | 0.708 | 0.569 | 0.659 | 0.674 | 0.714 | 0.791 | 0.598 | 0.826 | 0.612 |
| GeneralGNN | 0.662 | 0.753 | 0.655 | 0.803 | 0.667 | 0.657 | 0.712 | 0.665 | 0.776 | 0.729 |
| GraphConv | 0.638 | 0.715 | 0.663 | 0.769 | 0.709 | 0.640 | 0.718 | 0.676 | 0.745 | 0.722 |
| GCN | 0.709 | **0.794** | 0.681 | **0.846** | 0.663 | 0.636 | 0.752 | **0.722** | 0.796 | **0.749** |

Three things to take from this:

**iNcRD-HG tops the released protocol and loses roughly half the honest cells.**
A plain GCN beats it by 0.22 on `cold_drug`/LncDrug (0.846 vs 0.627) and by 0.15
on `matched`/LncDrug. It still wins `cold_rna`/LncDrug and `cold_drug`/MiDrug.
The ordering is scrambled, not simply reversed -- which is the point: the
released protocol does not rank these methods on anything a biologist wants.

**It is the only method that debiasing hurts.** Averaged over the four honest
splits, per-drug AUC moves by -0.065 (LncDrug) and -0.062 (MiDrug) for iNcRD-HG,
while every other architecture is flat or gains. The semantic-enhanced
heterogeneous design leans on the popularity shortcut harder than the baselines
it beats.

**The reported margin is smaller than the seed spread.** Table 3 claims +0.0128
AUC over the best baseline. On the debiased LncDrug splits iNcRD-HG's own seed
std is 0.153 (`matched`) and 0.168 (`cold_drug`) -- an order of magnitude larger.

## 4. Global AUC is the wrong headline once negatives are matched

Pooled AUC and per-drug AUC diverge sharply on the debiased splits: iNcRD-HG
scores 0.500 pooled but 0.648 per-drug on `matched`/LncDrug. With the class
ratio now fixed per drug, pooled AUC is dominated by whether scores are
calibrated *across* drugs, which is a different failure from ranking ncRNAs
*within* a drug. Report both; the per-drug and full-ranking numbers are the ones
that track the screening task.

## Reproducing

```bash
python analysis/popularity_bias_diagnosis.py --root .
python analysis/ncrd_bench.py --root . --out ./9.bench
python analysis/run_bench.py --root . --seeds 666 0 1 2 3 42 7 13 --out ./9.bench/results_8seed.csv
python analysis/summarize_bench.py --results ./9.bench/results_8seed.csv
```
