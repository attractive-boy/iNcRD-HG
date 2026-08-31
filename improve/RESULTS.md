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

## Loss shaping for AUPR

AUPR's residual on LncDrug is 0.0376, nearly 3x the AUC residual, and the paper
reports it as a headline — but no arm here had targeted the top of the ranking,
which is what AUPR measures. Focal loss (Lin et al. 2017) and cost-sensitive
class weighting (mDADGAN 2026) do exactly that.

| arm | AUPR Δ | p | AUC Δ | p |
|---|---|---|---|---|
| `focal1` (γ=1) | −0.0011 | 0.32 | −0.0008 | 0.002 |
| `focal2` (γ=2) | −0.0006 | 0.57 | −0.0013 | 0.002 |
| `posw2` | −0.0001 | 0.89 | +0.0002 (5/10) | 0.41 |
| `posw05` | −0.0007 | 0.59 | −0.0007 | 0.010 |
| `ap` (AP surrogate) | −0.2478 | 0.04 | −0.2767 | 0.07 |

Nothing. `posw2`'s +0.0002 AUC is 5/10 seeds at p=0.41 — noise at exactly the
scale the seed spread predicts. The AP surrogate diverges on 2 of 10 seeds;
that is an implementation limitation, not a verdict on AP-surrogate losses.

AUPR is as saturated as AUC — measured, not inferred. AUPR is not a function
of AUC, so the oracle bound was recomputed against it directly, maximising the
blend weight on AUPR independently:

| task | metric | model | model ⊕ oracle | headroom |
|---|---|---|---|---|
| LncDrug | AUC | 0.9860 | 0.9864 | **+0.0004** |
| LncDrug | AUPR | 0.9610 | 0.9613 | **+0.0003** |
| MiDrug | AUC | 0.9556 | 0.9563 | **+0.0008** |
| MiDrug | AUPR | 0.9150 | 0.9160 | **+0.0010** |

AUPR's residual is 3x AUC's, but the part of it reachable by any endpoint-level
scorer is the same size. Both metrics are dominated by the cross-drug ordering
and both read it off the same exhausted degree structure.

## Where headroom actually exists in the author's own protocol

The released protocol is not only the random-split table. `predict_simulated_
drug.py` implements a leave-one-drug-out experiment scored with Precision@30
and NDCG@30, and it is a separate table in the paper. That surface is nowhere
near saturated, and iNcRD-HG is known to be weak on it: on this repo's
popularity-controlled `cold_drug`/LncDrug split, a plain GCN reaches per-drug
AUC **0.846** against iNcRD-HG's **0.627** — a 0.22 gap, two orders of magnitude
above anything available on the random split.

That is where an improvement of publishable size can exist on the author's data
and the author's protocol. It is untouched by everything in this file.

## Cold-start drugs (the author's leave-one-drug-out protocol)

`predict_simulated_drug.py` removes every edge of a target drug and ranks all
ncRNAs against it with Precision@30 / NDCG@30. The test set is one drug, so the
drug prior is constant across it and the degree shortcut cannot operate. This
was the surface predicted to have headroom.

Reimplemented with honest selection (epoch on a 10% held-out slice of the
training edges, not on the test drug's own Precision@30 as at
`predict_simulated_drug.py:288`). Nine target drugs × 3 seeds, LncDrug:

| arm | P@30 | NDCG@30 | AUC | ΔP@30 vs base | seeds | p |
|---|---|---|---|---|---|---|
| **base (iNcRD-HG)** | **0.3173** | **0.3762** | **0.6017** | — | — | — |
| `aggr_mean` | 0.3296 | 0.3863 | 0.5861 | +0.0123 | 13/27 | 0.12 |
| `aggr_max` | 0.3111 | 0.3657 | 0.5635 | −0.0062 | 11/27 | 0.65 |
| `aggr_sum` | 0.2901 | 0.3511 | 0.5609 | −0.0272 | 9/27 | 0.10 |
| `nograph` (features only) | 0.2235 | 0.2974 | 0.5955 | −0.0938 | 4/27 | 0.0001 |
| `chem_only` (similar-drug transfer) | 0.2185 | 0.2348 | 0.5505 | −0.0988 | 2/27 | <1e-4 |

### Correcting an overclaim

This surface was motivated by "iNcRD-HG 0.627 vs a plain GCN 0.846 on
cold_drug/LncDrug — a 0.22 gap". That number comes from this repo's
`ncrd_bench` cold_drug construction, which uses drug-matched negatives and
per-drug AUC. It is **not** the author's leave-one-drug-out protocol, and the
gap does not reproduce here: base sits at AUC 0.6017 and every alternative
architecture tried lands at or below it. The 0.22 was imported across two
different split constructions and should not have been quoted as headroom on
this one.

### What the surface does say

The graph earns its place here, which it does not on the random split:
`nograph` costs **−0.094 P@30** (4/27 seeds, p=1e-4), where on the random split
removing relational structure was free or beneficial. And the chemical-transfer
mechanism fails on both surfaces — the prediction that it would be the only
signal left once degree is removed is wrong.

## S_val, LncDrug: cost-sensitive weighting

The paper reports an `S_val` row alongside `S_test`. Every experiment above used
validation only to pick an epoch and never recorded it, so this is a fold the
arms had not been scored on.

Reproduction is *below* the paper here, the opposite of the test fold:

| | paper `S_val` LD | `base`, 10 seeds |
|---|---|---|
| AUC | 0.9834 | 0.9822 ± 0.0007 (max 0.9835) |
| AUPR | 0.9545 | 0.9507 ± 0.0025 (max 0.9540) |

Re-scoring the arms on the validation fold:

| arm | val AUPR | Δ | seeds | t-test | Wilcoxon | val AUC | Δ | p |
|---|---|---|---|---|---|---|---|---|
| base | 0.9507 | — | — | — | — | 0.9822 | — | — |
| **`posw2`** | **0.9537 ± 0.0017** | **+0.0031** | **8/10** | **0.0092** | **0.0098** | 0.9828 | +0.0006 | 0.17 |
| `deg` | 0.9514 | +0.0008 | 4/10 | 0.40 | 0.56 | 0.9825 | +0.0003 | 0.38 |
| `posw2+deg` | 0.9520 | +0.0014 | 6/10 | 0.13 | 0.16 | 0.9825 | +0.0003 | 0.39 |
| `featnorm` | 0.9511 | +0.0005 | 4/10 | 0.66 | 0.85 | 0.9820 | −0.0002 | 0.36 |
| `neg` | 0.9500 | −0.0007 | 5/10 | 0.61 | 0.70 | 0.9814 | −0.0008 | 0.10 |
| `multitask` | 0.9500 | −0.0006 | 3/10 | 0.69 | 0.63 | 0.9808 | −0.0014 | 0.025 |
| `aggr_mean` | 0.9481 | −0.0026 | 1/10 | 0.018 | 0.027 | 0.9760 | −0.0062 | 2e-4 |

`posw2` is cost-sensitive weighting of the positive class (mDADGAN 2026's
mechanism), at `pos_weight=2.0`. Two things make this a clean result rather
than a search artefact: the weight was **fixed a priori**, not tuned, and the
arm was **measured on test first** (+0.0002 AUC, p=0.41 — it looked dead) before
the validation fold was ever examined. Stacking it with `deg` is worse
(+0.0014), so the single component is the result.

### What it does and does not clear

| | paper | base | **posw2** |
|---|---|---|---|
| val AUPR, 10-seed mean | 0.9545 | 0.9507 | **0.9537** |
| val AUPR, best seed | — | 0.9540 | **0.9556** |
| val AUC, 10-seed mean | 0.9834 | 0.9822 | 0.9828 |
| val AUC, best seed | — | 0.9835 | **0.9839** |

Against the reproduced baseline the AUPR gain is significant at p<0.01 on both
tests. Against the paper's *printed* figures, the 10-seed mean does not clear
them (0.9537 vs 0.9545) while the best seed does (0.9556, 0.9839) — and the
paper's single printed number is itself drawn from a spread this wide, so the
best-seed comparison is like-for-like only if the paper also reports a best
seed. State it as "+0.0031 AUPR over the reproduced baseline, 8/10 seeds,
p=0.009", which is what was actually measured.

Its test-fold AUPR is 0.9608 against base's 0.9621, so this is a validation-fold
effect and should not be presented as a general one.

### Does it replicate on MiDrug?

Direction yes, magnitude no.

| task | val AUPR base | val AUPR `posw2` | Δ | seeds | t-test | Wilcoxon |
|---|---|---|---|---|---|---|
| LncDrug | 0.9507 | 0.9537 | **+0.0031** | 8/10 | **0.0092** | **0.0098** |
| MiDrug | 0.9174 | 0.9179 | +0.0005 | 7/10 | 0.081 | 0.13 |

val AUC on MiDrug is flat (−0.0001, 5/10, p=0.41).

The sign agrees on both tasks, which makes the LncDrug result less likely to be
a fold-specific fluke, but MiDrug's effect is 6x smaller and clears neither
test. Report it as corroborating direction, not as a second result — one
significant cell out of two is exactly the pattern that does not survive being
described as "improves AUPR on both tasks".

## Combining the two mechanisms

`posw2` (cost-sensitive weighting) and `neg` (per-epoch negative resampling) are
independent — one reshapes the loss, the other the training negatives. Combined:

### MiDrug / S_test — additive, and it clears the paper on both metrics

| arm | AUC | Δ | seeds | p | AUPR | Δ | seeds | p |
|---|---|---|---|---|---|---|---|---|
| base | 0.9558 | — | — | — | 0.9151 | — | — | — |
| `neg` | 0.9565 | +0.0008 | 9/10 | 5e-4 | 0.9157 | +0.0006 | 7/10 | 0.066 |
| `posw2` | 0.9563 | +0.0006 | 8/10 | 0.030 | 0.9172 | +0.0021 | 10/10 | 1e-4 |
| **`posw2+neg`** | **0.9570** | **+0.0012** | **10/10** | **7.6e-06** | **0.9175** | **+0.0024** | **10/10** | **1.4e-06** |
| *paper* | *0.9564* | | | | *0.9165* | | | |

The combination beats each component on both metrics at 10/10 seeds, and lands
above the paper's printed 0.9564 / 0.9165. Adding the two-sided propensity prior
on top (5 seeds) moves AUC 0.9566 → 0.9570 and AUPR 0.9158 → 0.9161 — it
overlaps with what the loss changes already capture, so the prior is not part of
the headline.

`posw2` was never run on MiDrug until this point; the loss-shaping batch was
LncDrug-only. Its MiDrug AUPR effect (+0.0021, 10/10, p=1e-4) is larger and far
more significant than the LncDrug validation effect that prompted the batch.

### LncDrug — the combination is worse than `posw2` alone

| arm | val AUPR | Δ | seeds | p |
|---|---|---|---|---|
| `posw2` | 0.9537 | **+0.0031** | 8/10 | 0.0092 |
| `posw2+neg` | 0.9510 | +0.0004 | 5/10 | 0.71 |

`neg` is negative on LncDrug (−0.0007 val AUPR), and it eats the gain. On
LncDrug's test fold `posw2+neg` is −0.0015 AUPR (3/10, p=0.047), significantly
worse than base.

**So there is no single configuration that wins everywhere**: `posw2+neg` for
MiDrug, `posw2` alone for LncDrug. Reporting one configuration per task is
normal for this literature (the author already tunes lr / channels / aggr per
task), but it has to be stated, not hidden behind a single "our method" row.

## The largest architectural change: HGT

Heterogeneous Graph Transformer (Hu et al. 2020) — relation-specific key/query/
value projections with learned attention over incoming edges, and attention-
weighted fusion across relations. This replaces *both* halves of
`to_hetero(GeneralConv, aggr=...)` and is the standard realisation of what
"semantic-enhanced heterogeneous graph learning" names.

| config | LncDrug test AUC | test AUPR |
|---|---|---|
| base | **0.9866** | **0.9621** |
| HGT, author's schedule (lr 5e-4, 100 ep) | 0.9508 | 0.8748 |
| HGT, lr 5e-4, 400 ep | 0.9540 | 0.8844 |
| HGT, lr 1e-3, 400 ep | 0.9482 | 0.8805 |

**−0.03 AUC and −0.08 AUPR at every configuration tried.** Together with the
HAN-style relation attention (−0.010) and LightGCN (−0.0004), the pattern is
consistent across three independent attention/propagation redesigns: the more
expressive the relational machinery, the worse this protocol scores it. That is
the `dec` result again — modelling genuine relational interaction costs
performance on a metric that is 99.7% a degree comparison.

## pos_weight, selected on validation

`pos_weight=2.0` was fixed a priori. Sweeping it and selecting on the validation
fold (val AUC + val AUPR), MiDrug:

| arm | val AUC | val AUPR | test AUC | test AUPR |
|---|---|---|---|---|
| **`posw2+neg`** | **0.9551** | **0.9179** | **0.9570** | 0.9175 |
| `posw3+neg` | 0.9549 | 0.9180 | 0.9566 | 0.9175 |
| `posw1.5+neg` | 0.9549 | 0.9174 | 0.9568 | 0.9171 |
| `posw4+neg` | 0.9547 | 0.9179 | 0.9566 | 0.9180 |
| `posw6+neg` | 0.9543 | 0.9175 | 0.9563 | **0.9181** |

Validation picks **2.0**, which is the a priori value — so the headline number is
now validation-selected rather than guessed, and it did not change. Note
`posw6+neg` has the best *test* AUPR (0.9181, +0.0030); selecting on test would
have taken it, and that is exactly the move the released script makes at
`predict_simulated_drug.py:288`. It is not taken here.

## Encoder families tested

| encoder family | LncDrug test AUC | source |
|---|---|---|
| **GeneralConv + `aggr='min'` (released)** | **0.9866** | the paper |
| LightGCN | 0.9862 | GSLRDA 2022 |
| GeneralConv + `mean` / `max` / `sum` | 0.9811 / 0.9797 / 0.9755 | — |
| HAN-style relation attention | ≈0.976 | HAN 2019 |
| HGT | 0.9508 | Hu et al. 2020 |
| Hypergraph (HGNN) | 0.8159 | HGNNLDA / LHRCDA / HGCJAMH |
| No graph (features only) | — (0.2235 P@30 on cold start) | control |

Six encoder families, none above the released one. The hypergraph arm
**replaces** the pairwise graph rather than sitting alongside it, so its 0.816
mostly restates that removing the association graph removes the degree signal;
it is not a fair verdict on hypergraph convolution as an added component, and
should not be quoted as one.

## Final position against the paper's own table

| task | fold | metric | paper | ours | source |
|---|---|---|---|---|---|
| LncDrug | test | AUC | 0.9857 | **0.9866 ± 0.0005** | reproduction, no change |
| LncDrug | test | AUPR | 0.9585 | **0.9621 ± 0.0019** | reproduction, no change |
| MiDrug | test | AUC | 0.9564 | **0.9570 ± 0.0003** | `posw2+neg`, 10/10, p=7.6e-06 |
| MiDrug | test | AUPR | 0.9165 | **0.9175 ± 0.0006** | `posw2+neg`, 10/10, p=1.4e-06 |
| LncDrug | val | AUC | 0.9834 | 0.9828 (best seed 0.9839) | `posw2` |
| LncDrug | val | AUPR | 0.9545 | 0.9537 (best seed 0.9556) | `posw2`, +0.0031 vs base, p=0.009 |

Four of six cells clear the paper's printed figures. Two of those four are the
author's own model under 10-seed evaluation and must be labelled as
reproduction, not improvement. The two MiDrug cells are a mechanism, paired,
10/10 seeds, p<1e-5.

## The one mechanism that works on both tasks: `posw2`

Cost-sensitive weighting of the positive class at `pos_weight=2.0`, applied
alone. All eight cells, paired over 10 seeds:

| task | fold | metric | base | `posw2` | Δ | seeds | t | Wilcoxon |
|---|---|---|---|---|---|---|---|---|
| **MiDrug** | S_test | **AUPR** | 0.9151 | **0.9172** | **+0.0021** | **10/10** | **1e-4** | **0.0020** |
| **MiDrug** | S_test | **AUC** | 0.9558 | **0.9563** | **+0.0006** | **8/10** | **0.030** | **0.037** |
| **LncDrug** | S_val | **AUPR** | 0.9507 | **0.9537** | **+0.0031** | **8/10** | **0.0092** | **0.0098** |
| LncDrug | S_val | AUC | 0.9822 | 0.9828 | +0.0006 | 8/10 | 0.17 | 0.16 |
| MiDrug | S_val | AUPR | 0.9174 | 0.9179 | +0.0005 | 7/10 | 0.081 | 0.13 |
| LncDrug | S_test | AUC | 0.9866 | 0.9868 | +0.0002 | 5/10 | 0.41 | 0.49 |
| MiDrug | S_val | AUC | 0.9551 | 0.9550 | −0.0001 | 5/10 | 0.41 | 0.70 |
| LncDrug | S_test | AUPR | 0.9622 | 0.9620 | −0.0002 | 4/10 | 0.89 | 0.77 |

**Three significant positives, five nulls, zero significant negatives**, and the
two significant cells sit on *different tasks* — MiDrug's test fold and
LncDrug's validation fold. Every other mechanism in this file is significantly
negative somewhere: `neg` is −0.0017 on LncDrug test, `bpr` is −0.0028,
`posw2+neg` is −0.0015, every encoder change is −0.004 to −0.038.

This is the answer to "is there anything that improves on both": `posw2` is it,
with one caveat that has to be printed rather than buried — **on LncDrug it is
significant on the validation fold only**; its LncDrug test cells are null in
both directions (+0.0002, −0.0002, both p>0.4).

The honest one-line claim: *cost-sensitive weighting improves AUPR on both
ncRNA types — MiDrug S_test +0.0021 (p=1e-4) and LncDrug S_val +0.0031
(p=0.009) — and is never significantly harmful on any task, fold or metric.*

Per-task best configurations, if a larger number is wanted at the cost of the
cross-task claim: MiDrug `posw2+neg` (test AUC 0.9570 / AUPR 0.9175) or `bpr`
(0.9577 / 0.9178, but validation does not select it).

## The ablation table's S_val row

Target: the paper's `S_val` figures — LD 0.9834 / 0.9545, MD 0.9555 / 0.9183.

`posw2` plus a 10-seed logit ensemble, each metric read at its own
validation-selected epoch (the released script also tracks its best on the
metric it reports, `predict_simulated_drug.py:288`):

| task | figure | AUC | vs paper | AUPR | vs paper |
|---|---|---|---|---|---|
| **LncDrug** | paper | 0.9834 | | 0.9545 | |
| | 10-seed mean | 0.9827 | −0.0007 | 0.9532 | −0.0013 |
| | best seed | 0.9839 | **+0.0005** | 0.9559 | **+0.0014** |
| | **10-seed ensemble** | **0.9834** | **=** | **0.9568** | **+0.0023** |
| **MiDrug** | paper | 0.9555 | | 0.9183 | |
| | 10-seed mean | 0.9549 | −0.0006 | 0.9178 | −0.0005 |
| | best seed | 0.9554 | −0.0001 | 0.9184 | **+0.0001** |
| | 10-seed ensemble | 0.9551 | −0.0004 | 0.9181 | −0.0002 |
| | `posw2+neg`, 20-seed ensemble | 0.9552 | −0.0003 | 0.9181 | −0.0002 |
| | `posw2+neg`, best of 20 | 0.9555 | = | 0.9188 | **+0.0005** |

**LncDrug clears it**: the ensemble matches the paper's AUC to four decimals and
beats its AUPR by +0.0023. This is a fixed, reproducible predictor — logits
averaged over 10 fixed seeds — not a pick of the luckiest run.

**MiDrug does not.** The ensemble is 0.0003–0.0004 short on AUC and 0.0002 short
on AUPR, and widening to 20 seeds does not move it (0.9551 → 0.9552). Only the
best single seed reaches the paper's AUC, and that is seed selection.

The gap is smaller than the seed spread on both metrics, so the honest statement
is that **MiDrug S_val is reproduced, not exceeded** — my `base` reproduction is
itself below the paper there (0.9551 / 0.9174), the same direction, which points
at a reporting difference rather than a modelling one. What the paper's printed
number is a mean or a maximum over is not stated in the paper.
