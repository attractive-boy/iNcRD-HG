# Improving iNcRD-HG on the released protocol

Everything here runs on the **author's own protocol**: the frozen
`RandomLinkSplit(num_val=0.2, num_test=0.2, neg_sampling_ratio=2.0)` folds that
`heter_data_prepare.py` writes, scored with global AUC / AUPR. Nothing is
debiased. Branch is cut from the pristine author code (`main`, e2a33e5); the
only edit to a released file is a two-line guard in `heter_data_prepare.py` so
data prep does not crash on the `pea`/`spe` similarity matrices, which the data
release does not ship and the paper does not use.

## Reproduction

`base` is the author's architecture verbatim.

| | paper | `base`, 10 seeds (best epoch on test) |
|---|---|---|
| LncDrug AUC | 0.9857 | **0.9866 ± 0.0005** (max 0.9873) |
| LncDrug AUPR | 0.9585 | 0.9624 |
| MiDrug AUC | 0.9564 | **0.9559 ± 0.0005** (max 0.9563) |
| MiDrug AUPR | 0.9165 | 0.9155 |

The reproduction is exact. Three selection rules are recorded per run —
`final` (last epoch), `best_test` (best epoch measured on test, which is what
`predict_association.py:180` prints) and `sel_val` (epoch chosen on validation).
They agree to ~0.0005 here, so the choice does not drive any conclusion below.

## Result: four proposed changes, none of them helps

Paired by seed against `base`, 10 seeds, `sel_val_auc`.

| arm | change | LncDrug Δ | p | MiDrug Δ | p |
|---|---|---|---|---|---|
| `neg` | fresh negatives each epoch | −0.0017 | 0.028 | **+0.0007** (9/10) | 0.001 |
| `featnorm` | row-L2 on node features | −0.0008 | 0.008 | **+0.0005** (8/10) | 0.049 |
| `deg` | relation-wise degree encoding | −0.0002 | 0.36 | −0.0001 | 0.55 |
| `dec` | Hadamard + \|diff\| decoder | **−0.0599** | <1e-4 | **−0.0173** | <1e-4 |
| `all` | all four | −0.0174 | <1e-4 | −0.0092 | <1e-4 |

Seed ensembling, logits averaged over 10 inits at the val-selected epoch:
**+0.0002 AUC (LncDrug), +0.0003 (MiDrug)**; LncDrug AUPR *drops* 0.0019 at
n=10. Also nothing.

`dec` was re-run at lr 1e-4 and 2e-4 with 200 epochs in case it was being
judged on hyperparameters tuned for the base decoder. It gets **worse**, and
the collapse (peaks near 0.95, ends near 0.62) reproduces at every lr — so it
is a property of the change, not of the tuning.

## Why the model-side changes have no headroom

Two results in the table explain each other and are worth more than the deltas.

**The degree encoding is *exactly* neutral** — −0.0002 and −0.0001, p=0.36 and
0.55, the only arm that lands on zero from both sides. Handing the model each
node's per-relation degree explicitly changes nothing, which means the two-layer
GeneralConv already recovers it perfectly. There is no degree information left
on the table.

**The richer decoder is the largest effect in the table, and it is negative.**
Feeding `z_rna * z_drug` and `|z_rna - z_drug|` forces the decoder to model a
genuine ncRNA×drug interaction, and that costs 0.06 AUC. On this split it is
not an improvement to stop relying on the drug prior — it is a regression.

Together: a scorer that reads only a drug's training degree already reaches
**0.9775** on LncDrug, against `base` at 0.9866. The 0.009 that separates them
is what is actually available to a model, and `base`'s own seed std is 0.0005.
Anything that makes the model more relational moves away from the metric;
anything that makes it better at reading degree is redundant, because it
already does. What *does* buy a gain is a better estimate of the sampling
propensity itself — see "The one thing that does improve" below.

## What is left if the number has to go up

- **Hyperparameter search on validation** (`find_best_params.py`) is untried
  here and is the only honest lever remaining. Expect single thousandths.
- **Reporting the max over seeds** gives LncDrug 0.9873 / AUPR 0.9669, above the
  published 0.9857 / 0.9585. This is seed selection, not an improvement, and the
  spread it is drawn from is 0.0005 wide. It should not be reported as a gain.

## The ceiling, measured three ways

Three measurements bound what is available. They do not say "zero" -- see the
next section for the +0.0012 that is real -- they say the budget is tiny and
say where it is.


**The drug prior is already fully absorbed** (`improve/headroom.py`). The
degree-only scorer and the model correlate at rho≈0.87 but not 1.0, so a blend
`z(model) + alpha * z(log drug degree)` with alpha selected on validation could
in principle beat both. It does not: LncDrug **−0.0002** (validation picks
alpha≈0.6, it does not transfer, 2/5 seeds gain), MiDrug **0.0000** (validation
picks alpha=0 on 4/5 seeds). There is no error the prior can fix that the model
has not already fixed.

**The residual error is not concentrated anywhere reachable.** Test AUC
recomputed inside quartiles of drug training degree, 5 seeds:

| task | q1 (low deg) | q2 | q3 | q4 (high deg) | global |
|---|---|---|---|---|---|
| LncDrug | 0.937 | 0.912 | 0.550 | — | 0.9860 |
| MiDrug | 0.857 | 0.851 | 0.837 | 0.791 | 0.9556 |

Conditioning on degree drops the model from 0.956 to 0.79–0.86 (MiDrug) — so
the global figure is overwhelmingly the *between*-degree ordering, and what is
left inside a degree stratum is the part the model genuinely cannot do.

**And the metric barely pays for fixing it** (`improve/metric_decomp.py`).
Global AUC is `P(score_pos > score_neg)` over every positive x negative pair in
the test fold. Split that pair set by whether the two edges share a drug:

| task | ranking pairs | same-drug (biology) | cross-drug (drug prior) |
|---|---|---|---|
| LncDrug | 354,482 | 1,266 (**0.36%**) | 353,216 (99.64%) |
| MiDrug | 2,324,168 | 7,235 (**0.31%**) | 2,316,933 (99.69%) |

**99.7% of the metric is "does a positive on a popular drug outrank a negative
on a rare one".** A model that became a *perfect* within-drug ranker overnight —
from 0.65 to 1.0 — would move global AUC by **0.0012**. That is the entire
budget available to any biologically real improvement, and it is twice the seed
noise. The protocol does not have a ceiling that better modelling can reach; it
has a metric that almost never asks the question the model is being improved at.

This is measured on the author's own frozen split with no debiasing applied, so
it stands independently of any benchmark redesign.

## The one thing that does improve on the released protocol

The pair decomposition above bounds what a better *within-drug* ranker can win.
It says nothing about the cross-drug pairs, which are 99.6% of the metric and
where the model sits at 0.988, not 1.0. Chasing that residual found a real
gain, and it is exactly the size the decomposition predicted.

`RandomLinkSplit` samples negatives uniformly over **(ncRNA, drug)**, so
positives concentrate on popular ncRNAs as well as popular drugs. Every degree
control in this repo had only ever used the drug side. Fitting a logistic
prior on **both** log degrees (`improve/priors.py`, fitted on the training
fold's labels):

| task | drug-degree prior | ncRNA-degree prior | both |
|---|---|---|---|
| LncDrug | 0.9775 | 0.4036 | 0.9781 |
| MiDrug | 0.9316 | 0.6802 | **0.9534** |

On MiDrug the ncRNA side is worth **+0.022** over the drug-only control that
this repo has been quoting. On LncDrug the lncRNA degree is *below chance*
(0.4036) and adds nothing.

Stacking the two things that worked — per-epoch negative resampling, plus
blending the two-sided prior into the model score with alpha chosen on
validation — against `base`, paired over 10 seeds:

| task | arm | test AUC | Δ vs base | seeds | p |
|---|---|---|---|---|---|
| MiDrug | base | 0.9558 | — | — | — |
| MiDrug | `neg` | 0.9565 | +0.00076 | 9/10 | 0.0005 |
| MiDrug | **`neg` + two-sided prior blend** | **0.9570** | **+0.00118** | **10/10** | **<1e-4** |
| LncDrug | base | 0.9866 | — | — | — |
| LncDrug | base + prior blend | 0.9865 | −0.00006 | 5/10 | 0.62 |

**MiDrug 0.9570 against the published 0.9564** — an improvement on the released
protocol, 10/10 seeds, p<1e-4. LncDrug does not move; its `base` is already at
0.9866 against a published 0.9857, which is reproduction spread, not a gain.

Two things have to be said next to that number. It is **+0.0012**, which is the
budget the pair decomposition predicted almost exactly, and it is **9% of the
+0.0128 margin the paper claims over its best baseline**. And it is not
biology: it comes from estimating the negative-sampling propensity better, on
both endpoints instead of one.

## The sharpest number in this file

The AUC decomposition, run on the model's own scores:

| task | same-drug AUC | cross-drug AUC | global AUC |
|---|---|---|---|
| LncDrug | **0.4490** | 0.9883 | 0.9864 |
| MiDrug | 0.7553 | 0.9572 | 0.9565 |

**On LncDrug the published model is below chance at ranking ncRNAs within a
drug, and reports 0.9864.** Not merely weak — worse than a coin flip on the
only comparison that corresponds to the screening task. It survives because
that comparison is 0.36% of the metric.

## Reproducing

Data prep and all runs use `--run-dir` = a directory holding
`2.construct_network/` and `3.heter_data/`:

```bash
python heter_data_prepare.py                      # rebuild the frozen folds
python improve/run.py --arms base neg dec deg featnorm neg+dec neg+dec+deg all \
    --tasks LncDrug --out 9.improve/lnc.csv
python improve/summarize.py 9.improve/lnc.csv 9.improve/mi.csv
python improve/ensemble.py --arm base --out 9.improve/ens.csv
```

Per-seed CSVs for every number above are in `improve/results/`.

## LncDrug: the residual is located, and nothing reaches it

The residual was localised before anything was designed against it
(`improve/rare_drug.py`, 10 seeds, per-positive win rate against all negatives):

| | positives on low-degree drugs | on high-degree drugs |
|---|---|---|
| per-positive win rate | **0.8892** | 0.9891 |
| share of all positives | **2.6%** | 97.4% |
| share of all error | **21.5%** | 78.5% |

Spearman(win rate, drug degree) = 0.82. So 2.6% of positives carry 21.5% of the
error, and they are exactly the positives on drugs that are rare in lncRNA
space. Fixing them completely is worth **0.0029 AUC** — 2.4x the MiDrug gain,
and precisely targeted.

Three sources of side information were tested against it. All fail:

| mechanism | prior alone | blended into the model |
|---|---|---|
| raw drug degree (control) | 0.9775 | −0.0002 |
| **chemical-similarity-smoothed degree** | 0.9703 | −0.0003 |
| **cross-task degree** (the drug's MiDrug degree) | 0.9771 | −0.0003 |
| **ChemBERTa → expected degree** (ridge) | 0.9771 | −0.0002 |
| all of them together | 0.9779 | +0.0002 |

The cross-task arm is the notable one: when LncDrug is the task, the full MiDrug
edge set sits in the graph untouched, so a drug that is rare in lncRNA space may
be well studied in miRNA space. That is the paper's own pan-ncRNA premise stated
as a propensity, and it scores **below** the drug's own degree.

Read together with the smoothing and ChemBERTa results: **nothing available —
not chemistry, not the other ncRNA task, not learned drug structure — predicts
which rare drugs have lncRNA associations.** Those positives are not merely
hard for this architecture; they carry no signal any of the released modalities
can supply. LncDrug's 0.9866 stands, and the 0.0029 above it is not reachable.

## LncDrug ledger: every mechanism tried

Paired against `base` (0.9866 ± 0.0005), 10 seeds, `sel_val_auc`.

| # | mechanism | class | Δ | seeds | p |
|---|---|---|---|---|---|
| 1 | per-epoch negative resampling | training | −0.0017 | 2/10 | 0.028 |
| 2 | feature row-normalisation | input | −0.0008 | 2/10 | 0.008 |
| 3 | relation-wise degree encoding | input | −0.0002 | 3/10 | 0.36 |
| 4 | Hadamard + \|diff\| decoder | architecture | −0.0599 | 0/10 | <1e-4 |
| 5 | all of 1–4 | — | −0.0174 | 0/10 | <1e-4 |
| 6 | 10-seed logit ensemble | ensemble | +0.0002 | — | ns |
| 7 | drug-degree prior blend | prior | −0.0002 | 2/5 | 0.36 |
| 8 | two-sided degree prior | prior | −0.0001 | 5/10 | 0.62 |
| 9 | chem-similarity-smoothed degree | prior | −0.0003 | — | ns |
| 10 | cross-task (MiDrug) degree | prior | −0.0003 | — | ns |
| 11 | ChemBERTa → expected degree | prior | −0.0002 | — | ns |
| 12 | all four priors together | prior | +0.0002 | — | ns |
| 13 | `aggr='sum'` | architecture | −0.0110 | 0/10 | <1e-4 |
| 14 | `aggr='mean'` | architecture | −0.0055 | 0/10 | 0.007 |
| 15 | `aggr='max'` | architecture | −0.0069 | 0/10 | <1e-4 |
| 16 | **learned relation attention (HAN-style)** | architecture | ≈−0.010 | 0/3 | — |
| 17 | joint MiDrug supervision (multi-task) | training | −0.0012 | 3/10 | 0.073 |
| 18 | joint MiDrug + negative resampling | training | −0.0014 | 1/10 | 0.018 |
| 19 | **ensemble + full prior** (6 ⊕ 12) | combination | +0.0001 | CI [−0.0005, +0.0007] | p(>0)=0.64 |

Nineteen arms across five mechanism classes — training, input, architecture,
prior, ensemble. Nothing is distinguishable from zero on the positive side.
Arm 19 is the strongest combination available and its bootstrap CI over test
edges spans zero; its AUPR is *worse* than a single model (0.9592 vs 0.9625).

### Two by-products worth keeping

**`min` is not an arbitrary choice.** The paper never justifies `aggr='min'`
for LncDrug, and it beats every alternative: mean −0.0055, max −0.0069, sum
−0.0110, and a learned HAN-style relation attention loses by about 0.010 even
at 400 epochs. The fixed reduction the paper does not explain outperforms the
learnable semantic attention that its own title implies. That is a publishable
ablation, though not an improvement.

**Cross-task transfer fails as both a prior and a loss.** The drug's MiDrug
degree scores below its own degree (arm 10), and joint MiDrug supervision --
5394 extra positives on shared drug nodes, 4x LncDrug's own 1267 -- costs
0.0012 (arm 17). The pan-ncRNA premise does not pay on this task in either
form.

## Is +0.01 available? An oracle says no.

Every propensity estimator here failed, which shows the estimators failed, not
that the information is absent. `improve/oracle.py` separates the two: it reads
the **test fold's own labels** to compute each drug's and each ncRNA's true
positive rate, then scores test edges with them. It cannot be built without the
answers, so it is not a method — it is an upper bound on every scorer whose
inputs are the two endpoints' identities.

The leave-one-out correction is what makes it a bound rather than a fiction.
An endpoint appearing once in the fold has an empirical rate equal to its own
label; including it, "oracle" AUC reaches 0.9983 and the apparent headroom is
+0.0114, all of which is label memorisation. Excluding each edge from its own
endpoint's rate:

| task | model | oracle drug | oracle ncRNA | oracle both | model ⊕ oracle | headroom |
|---|---|---|---|---|---|---|
| LncDrug | 0.9861 | 0.9672 | 0.4811 | 0.9731 | 0.9864 | **+0.0003** |
| MiDrug | 0.9557 | 0.9277 | 0.6265 | 0.9415 | 0.9563 | **+0.0007** |

The blend alpha is chosen **on test**, deliberately — this is meant to be
optimistic. A scorer that knows both endpoints' true positive rates in the test
fold, fused with the model at a test-optimal weight, is worth **+0.0003** on
LncDrug. The whole propensity family is closed.

### The arithmetic for +0.01

Global AUC = 0.0036 × AUC_same + 0.9964 × AUC_cross, and the model sits at
0.0036 × 0.449 + 0.9964 × 0.9883 = 0.9863.

Reaching 0.9963 requires AUC_cross ≥ **0.99987** even with a *perfect*
within-drug ranker — one error in every 7,700 cross-drug pairs, from a model
currently below chance at ranking within a drug. The two routes bound as:

| route | ceiling | how it was bounded |
|---|---|---|
| within-drug (biology) | +0.0020 | 0.36% pair share, 0.449 → 1.0 |
| propensity (cross-drug) | +0.0003 | leave-one-out oracle, test-optimal blend |
| **both, simultaneously, with oracles** | **≈ +0.0023** | |

**+0.01 is ruled out by measurement.** The maximum available on LncDrug is about
+0.002, and collecting it requires a perfect within-drug ranker *and* a
propensity oracle that reads the test labels.

## Every layer of the model, and what it was worth

| layer | what was changed | best result |
|---|---|---|
| input features | row-L2 norm, relation-wise degree encoding | −0.0002 (deg, ns) |
| graph structure | cross-task edges as prior and as loss | −0.0003 / −0.0012 |
| encoder | GeneralConv → HAN-style relation attention | ≈ −0.010 |
| relation fusion | min → sum / mean / max / learned | −0.0055 … −0.0110 (min wins) |
| decoder | concat → Hadamard + \|diff\| | −0.0599 |
| loss | fixed → per-epoch resampled negatives; multi-task | −0.0017 / −0.0012 |
| training | 10-seed logit ensemble | +0.0002 (ns) |
| post-hoc | 5 propensity priors, blended | +0.0002 (ns) |

All eight layers. The model is not under-engineered on this protocol; the
protocol has nothing left to pay for.

## Methods from the current literature

Searched OpenAlex for 2024-06 onward in this exact area (ncRNA/miRNA/circRNA–
drug association prediction). The mechanism classes that had not been tested
here are reliable-negative selection (DMGAT 2025, LSNSCDA 2024), LightGCN-style
propagation (GSLRDA 2022 and its descendants), adaptive-mask graph contrastive
learning (DMAGCL 2025) and metapath guidance (MPHGNN 2026).

Two were reimplemented and run on the frozen LncDrug fold, both chosen because
they fit what this metric rewards rather than because they are recent:

| arm | method | source | LncDrug Δ vs base | seeds | p |
|---|---|---|---|---|---|
| `lightgcn` | symmetric-normalised propagation over ID embeddings, no feature transform, no nonlinearity, layer-averaged, inner-product decoder | LightGCN / GSLRDA | **−0.0004** | 4/10 | 0.096 |
| `relneg` | negatives resampled each epoch as the pool candidates the current model scores most confidently negative | DMGAT / LSNSCDA | **−0.0056** | 1/5 | 0.10 |

LightGCN was given its own tuning (lr 1e-3 and 5e-3, 600 epochs, selected on
validation) because its embeddings are randomly initialised and the author's
100-epoch schedule undertrains them. Its best configuration reaches 0.9862
against base's 0.9866.

LightGCN was the strongest prior bet in the whole investigation — no feature
transform, no nonlinearity, pure normalised propagation, i.e. the closest thing
in the literature to a smoothed degree model, on a metric that is 99.7% a
degree comparison. It still loses, which is the same result as arm 3: the
degree structure is already fully extracted, so a method that extracts it more
directly has nothing to add.

### Why searching further does not help

The +0.0023 ceiling is a property of the **metric and the fold**, not of the
architecture. Any scorer is a function of (ncRNA, drug); the pair decomposition
splits its AUC into a 0.36% same-drug part and a 99.64% cross-drug part
regardless of how the score is computed. A new method can only exceed the
ceiling through genuine pair-level accuracy, and to gain +0.01 it needs
cross-drug AUC of 0.99987 — while the best within-drug AUC any method in this
repo's debiased benchmark reaches is 0.79 (GCN on `matched`/LncDrug), against
iNcRD-HG's 0.648 and the current model's 0.449 on the released fold.
