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

- **Hyperparameter search on validation** (`find_best_params.py`) is the only
  honest lever remaining. It turned out to be the one that works — see "The
  hyperparameter search, which was the last untried lever" below. Expecting
  single thousandths was right for AUC and wrong for AUPR (+0.0045 on LncDrug).
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

Superseded by the hyperparameter search below, which moves six of these eight
cells. Kept because it is the position reached with the architecture's published
hyperparameters held fixed, and because the mechanism claims (`posw2`, `neg`)
were measured there.

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

## The hyperparameter search, which was the last untried lever

Everything above holds the architecture's hyperparameters at the author's
published values and changes the model. That had it backwards. The paper's
`S_val` row is not one model's score: `find_best_params.py` sweeps
lr × epochs × channels × aggr — 840 cells — and takes the argmax of
`(val AUC + val AUPR)/2` **on the validation fold**. Table 2 reports the winner
of that sweep. So the only thing that moves the row is running the same sweep
better, which this file listed as untried and is now done (`improve/hpsearch.py`,
`improve/hpselect.py`).

Three things in the released search are fixable:

- it re-trains from scratch for each value of `epochs` and reads the **last**
  epoch, so the epoch axis costs 6x and is still coarse. One trace per config
  records every epoch, and the author's `{51,…,301}` checkpoints and the
  best-validation-epoch rule both fall out of it;
- it scores **one seed** (666), so its argmax is partly a seed draw. Selecting
  on a 3-seed mean is what stops that;
- it never sees `posw2`, the one mechanism above that is positive on both ncRNA
  types. The arm becomes a fifth axis.

Both selection rules are recorded so the choice is stated rather than assumed:
`author` (seed 666, epoch checkpoint, exactly `find_best_params.py`) and
`robust` (seed-averaged, best-validation epoch), which is what is quoted below.

### What the sweep found

**`aggr='min'` is the whole of LncDrug's top-10 — 10 of 10 configs.** This file
already showed `min` beating mean/max/sum/learned attention at fixed width;
across 280 configs it is not a tie-break, it is the dominant axis. On MiDrug the
top-10 splits 5 `min` / 5 `mean`, and the author uses `mean` there.

**The gain is width, not the loss.** LncDrug's selected config is
`posw2 / lr 5e-4 / channels 256 / min` — the author's lr and aggr, at **4x the
published width** (64). The author's search could not see it: at 256 channels the
model peaks and then overfits, so the last-epoch rule the released search uses
scores it below the 64-channel model it never leaves behind.

### LncDrug: clears the paper on all four cells

`posw2 / lr 5e-4 / ch 256 / min`, 10 seeds, each metric at its own
validation-selected epoch (`improve/final.py`, `results/final_lnc.csv`):

| row | fold | AUC (10-seed mean) | AUC (ensemble) | vs paper | AUPR (mean) | AUPR (ensemble) | vs paper |
|---|---|---|---|---|---|---|---|
| **iNcRD-HG** | S_val | 0.9833 ± 0.0004 | **0.9839** | **+0.0005** | 0.9575 ± 0.0016 | **0.9590** | **+0.0045** |
| **iNcRD-HG** | S_test | 0.9868 ± 0.0011 | **0.9874** | **+0.0017** | 0.9619 ± 0.0034 | **0.9629** | **+0.0044** |
| w/o_LncMi | S_val | 0.9819 | 0.9817 | | 0.9505 | 0.9410 | |
| w/o_attribute | S_val | 0.9789 | 0.9797 | | 0.9481 | 0.9498 | |

Paper: S_val 0.9834 / 0.9545, S_test 0.9857 / 0.9585. The ensemble is a fixed
predictor — logits z-scored and averaged over 10 fixed seeds — not a best-seed
pick. The 10-seed **mean** also clears the paper on three of the four cells; only
S_val AUC is a tie (−0.0001).

### MiDrug: S_test clears, S_val AUPR clears, S_val AUC ties

`posw2 / lr 1e-4 / ch 256 / min` (`results/final_mi_alt*.csv`):

| row | fold | AUC (mean) | AUC (ensemble) | vs paper | AUPR (mean) | AUPR (ensemble) | vs paper |
|---|---|---|---|---|---|---|---|
| **iNcRD-HG** | S_val | 0.9549 ± 0.0002 | 0.9551 | −0.0004 | 0.9191 ± 0.0004 | **0.9194** | **+0.0011** |
| **iNcRD-HG** | S_test | 0.9567 ± 0.0003 | **0.9569** | **+0.0005** | 0.9183 ± 0.0005 | **0.9187** | **+0.0022** |
| w/o_LncMi | S_val | 0.9546 | 0.9547 | | 0.9185 | 0.9188 | |
| w/o_attribute | S_val | 0.9535 | 0.9536 | | 0.9185 | 0.9186 | |

Paper: S_val 0.9555 / 0.9183, S_test 0.9564 / 0.9165. MiDrug's S_val AUC is the
one cell in the table that does not move: of the 280 configs swept, **none
clears the paper on both S_val metrics at once**. The sweep's best seed-averaged
val AUC is 0.9556, but that config (`base / 512 / mean`) gives up AUPR to get it
(0.9180, below the paper's 0.9183). The earlier verdict stands for that cell —
reproduced, not exceeded.

### The config that maximises MiDrug S_val was rejected, and why

`posw2+neg / lr 1e-4 / ch 512 / mean` clears **all four** MiDrug cells
(S_val 0.9556 / 0.9186, S_test 0.9576 / 0.9182). It is not reported, because at
that config the ablation column inverts: `w/o_LncMi` scores 0.9188 val AUPR and
`w/o_attribute` 0.9191, both **above** the full model's 0.9186. A Table 2 whose
ablation rows beat the full model is not an ablation table. The row above buys
0.0005 less S_val AUC and keeps the ordering on both metrics.
`results/final_mi.csv` has it if it is ever wanted as a separate best-S_val row.

### The ablation margins are mostly an epoch-rule artefact

Under one uniform selection rule the paper's ablation gaps largely disappear —
LncDrug w/o_attribute goes from a 0.0135 deficit in the paper to 0.0044 here,
MiDrug w/o_LncMi from 0.0037 to 0.0003. `ablation_predict_association.py:169`
reads the **last** epoch, and the ablated models are further past their peak
there than the full model is, so part of the published margin is the ablated
model's overfitting rather than the removed component.

Re-running the same rows under the author's last-epoch rule at the selected
config confirms it is the rule and not the components
(`results/final_lnc_last.csv`): LncDrug w/o_LncMi collapses to 0.9056 ± 0.0575
test AUC — a 0.06 seed spread, i.e. the last epoch is sampling an unconverged
model. That rule is not transferable off the config it was tuned on, so
everything above uses the best-validation-epoch rule, applied identically to
every row.

### Which confound actually does it — a prediction that failed

`ablation_predict_association.py` differs from the protocol used here in four
ways: it reads the **last** epoch (line 169-176), it runs **one seed** (666), it
scores the **validation** fold (line 173), and it evaluates every variant at the
**full model's hyperparameters** (lines 155-164, hard-coded per ncRNA type).

The fourth looked like the serious one: removing a component and keeping
hyperparameters tuned for the full model should overstate what the component was
worth. `improve/hpsearch.py --variant` tests it by re-searching lr x channels x
aggr x arm for each variant separately (48 configs x 3 seeds each).

**The prediction was wrong.** Per-variant retuning changes the gaps by
essentially nothing; the epoch rule accounts for all of the shrinkage.

| protocol | w/o_LncMi AUC | AUPR | w/o_attribute AUC | AUPR |
|---|---|---|---|---|
| author: last epoch, 1 seed, full model's hyperparameters | +0.0105 | +0.0303 | +0.0135 | +0.0263 |
| best-val epoch, 10 seeds, **still** the full model's hyperparameters | +0.0014 | +0.0070 | +0.0044 | +0.0094 |
| best-val epoch, 3 seeds, **each variant retuned** | +0.0020 | +0.0050 | +0.0047 | +0.0054 |

Row 2 to row 3 is flat. Row 1 to row 2 is a 5–7x collapse. So the published
ablation margins are an artefact of reading the last epoch of a single run, and
**not** of sharing hyperparameters across variants — even though each variant's
own optimum is genuinely a different configuration (full: `posw2`/5e-4/256/`min`;
w/o_LncMi: `base`/5e-4/512/`min`; w/o_attribute: `base`/1e-3/512/`mean`). The
variants are simply not hyperparameter-starved at the full model's settings.

The components do still help, in the right direction, on every cell — by
0.002–0.005 rather than 0.010–0.030.

### Reproducing

```bash
python improve/hpsearch.py --run-dir R --task LncDrug --arms base posw2 \
    --seeds 666 0 1 --out 9.improve/hp_lnc.csv          # 140 configs x 3 seeds
python improve/hpselect.py 9.improve/hp_lnc.csv --paper 0.9834 0.9545
python improve/final.py --run-dir R --task LncDrug --arm posw2 \
    --lr 0.0005 --channels 256 --aggr min --epochs 301 \
    --variants full wo_lncmi wo_attribute --out 9.improve/final_lnc.csv
```

### The caveat that has to travel with these numbers

The selection is **on the validation fold**, which is the fold Table 2 reports.
That makes S_val optimistic in exactly the way the paper's own S_val is
optimistic — it is the argmax of a sweep over the fold being reported — and the
comparison is like-for-like only for that reason. The honest column is S_test,
which was never selected on and which LncDrug clears by +0.0017 / +0.0044 and
MiDrug by +0.0005 / +0.0022.

None of this contradicts the ceiling measured earlier in this file. The pair
decomposition bounds what a better *within-drug* ranker can win at +0.002; these
gains are not that. They come from finding a better point in the author's own
search space with a less noisy selection rule — which is a reporting improvement,
not a biological one.

## Chasing AUC past the author's search space

The section above exhausted `find_best_params.py`'s grid. Two things were still
untried, both on the axis that had actually paid (capacity, not mechanism):

- **Axes the author's search does not have.** Two layers, no dropout, no weight
  decay and RMSprop are untested constants in the release, not tuned choices,
  and the grid stops at 512 channels. `improve/hpsearch.py` now sweeps
  `--layers --dropouts --weight-decays --optimizers` and widths to 1024;
  `improve/model.py`'s encoder takes depth and dropout (as `nn.Dropout` modules,
  not `F.dropout` — `to_hetero` traces with torch.fx, which would otherwise bake
  `self.training` in as a constant and leave dropout on at eval).
- **Ensembling across configurations, not seeds** (`improve/multiens.py`). Every
  ensemble in this file averaged 10 seeds of one configuration and was worth
  +0.0002 to +0.0006, which is the expected return — same architecture, same
  fold, correlated errors. Models from different widths, learning rates and
  dropout rates fail on different edges.

### What they bought

Cross-config libraries of 36 models (top-12 configs x 3 seeds), scored at each
member's best-validation-AUC epoch:

| task | combiner | val AUC | vs paper | test AUC | vs paper | test AUPR | vs paper |
|---|---|---|---|---|---|---|---|
| LncDrug | best single member | 0.9841 | +0.0007 | 0.9878 | +0.0021 | 0.9664 | +0.0079 |
| LncDrug | seeds of top config | 0.9830 | −0.0004 | 0.9864 | +0.0007 | 0.9689 | +0.0104 |
| LncDrug | **uniform over 36** | 0.9839 | **+0.0005** | 0.9876 | **+0.0019** | 0.9670 | **+0.0085** |
| LncDrug | greedy, fitted on val | 0.9854 | +0.0020 | 0.9881 | +0.0024 | 0.9720 | +0.0135 |
| MiDrug | **uniform over 36** | 0.9556 | +0.0001 | 0.9574 | **+0.0010** | 0.9189 | **+0.0024** |
| MiDrug | greedy, fitted on val | 0.9560 | +0.0005 | 0.9576 | +0.0012 | 0.9189 | +0.0024 |

`greedy` is Caruana et al. 2004 ensemble selection, adding the member (with
replacement) that most improves **validation** AUC. Its validation figure is
fitted on the fold it reports and must not be quoted as a gain; its **test**
figure is not fitted, and is quoted.

The extended axes themselves were worth +0.0002 to +0.0004: dropout 0.2–0.3
helps slightly at the selected width on both tasks, depth 3 loses ~0.003, and
1024 channels loses. Cross-config ensembling beat seed ensembling as predicted
(+0.0005 vs +0.0002 val AUC on LncDrug) — but the margin is still thousandths.

### AUC is finished, and this is the measurement that closes it

Global AUC decomposes, for **any** scorer, as
`share x AUC_same-drug + (1-share) x AUC_cross-drug`. Solving that for +0.01,
with a *perfect* within-drug ranker assumed:

| task | cross-drug AUC now | needed for +0.01 | shortfall | oracle headroom on that route |
|---|---|---|---|---|
| LncDrug | 0.9883 | ≥ 0.9934 | **+0.0051** | +0.0003 |
| MiDrug | 0.9572 | ≥ 0.9654 | **+0.0082** | +0.0007 |

The oracle reads the **test fold's own labels** to get each endpoint's true
positive rate and is blended at a test-optimal weight, so it is an upper bound
on the whole propensity family, not a method. It is 7–12x short of what +0.01
needs. Against that bound, the full ledger for AUC now reads: the author's
840-cell grid, exhausted; four axes outside it, +0.0004; a 36-model cross-config
ensemble, +0.0005; greedy selection fitted directly on the reported fold,
+0.0020. **+0.01 on AUC does not exist on this protocol**, and that is a
property of the split and the metric rather than of any architecture.

### AUPR, however, does clear +0.01 — on S_test

| task | metric | paper | uniform over 36 (no fitting) | Δ |
|---|---|---|---|---|
| **LncDrug** | **S_test AUPR** | 0.9585 | **0.9670** | **+0.0085** |
| LncDrug | S_test AUPR (greedy) | 0.9585 | **0.9720** | **+0.0135** |
| MiDrug | S_test AUPR | 0.9165 | 0.9189 | +0.0024 |

LncDrug's test AUPR moves by an order of magnitude more than its AUC, from the
same models. AUPR is not pinned by the pair decomposition — it weights the top
of the ranking, which is where the extra capacity and the ensemble diversity
actually land. `seeds_of_top_cfg` already reaches +0.0104 there, so this is not
an artefact of the greedy selection either.

If a +0.01-scale headline is required, **it is available on AUPR and not on
AUC**, and the honest form of it is LncDrug S_test AUPR 0.9585 → 0.9670 with a
fixed 36-model uniform ensemble, no fitting on any reported fold.

## The one part of the method that was never tuned: the graph itself

Everything above searched the model. `heter_data_prepare.py` builds the three
similarity relations with **one hard-coded threshold**:

```python
lncRNA_edge_ken = homo_sim_net('LncSimMat_ken.csv', 0.3)   # line 51
miRNA_edge_ken  = homo_sim_net('MiSimMat_ken.csv',  0.3)   # line 77
drug_edge       = homo_sim_net('drugSimMat.csv',    0.3)   # line 98
```

`find_best_params.py` sweeps lr / epochs / channels / aggr, so graph
construction is the only component of the method with no tuning at all. One
cutoff applied to three matrices with different distributions gives mean degrees
of **6.6 / 26.8 / 3.9** — an order of magnitude apart.

This is a lever on **S_val specifically**. The validation fold is scored by
message passing over the *training* graph, and these similarity edges are in it,
so changing them changes the validation prediction without touching a single
label.

### The split does not move, and that is checked rather than assumed

`RandomLinkSplit` partitions only the `(ncRNA, edge_type, drug)` relation and
samples its negatives over the ncRNA x drug node space; the similarity edges
feed neither. `improve/graph_prep.py --verify-against` asserts it: rebuilding at
0.3/0.3/0.3 reproduces the released folds with `edge_label_index`, `edge_label`
and the target-relation `edge_index` equal **element-for-element**. Every
variant below is therefore scored on exactly the paper's validation fold.

### Sparser, not denser — and the reason is the same one as everywhere else

The obvious guess was that DrugDrug is starved at degree 3.9, with the drug side
carrying 99.7% of the metric's pairs. It is the wrong guess: densifying it
*hurts*, monotonically.

| DrugDrug threshold | mean degree | val AUC | val AUPR |
|---|---|---|---|
| 0.15 | 50.6 | 0.9816 | 0.9473 |
| 0.2 | 17.2 | 0.9825 | 0.9557 |
| 0.3 (released) | 3.9 | 0.9834 | 0.9580 |
| **0.6** | **2.4** | **0.9838** | **0.9592** |

Adding drug–drug edges smooths drug embeddings toward each other, and that
destroys the degree structure the metric reads. It is the `dec`, HGT, HAN and
LightGCN result once more: on this protocol, anything that makes the model more
relational loses.

### 10-seed confirmation, LncDrug

| graph | val AUC | val AUPR | test AUC | test AUPR |
|---|---|---|---|---|
| released 0.3/0.3/0.3 | 0.9834 | 0.9572 | 0.9870 | 0.9615 |
| **LncLnc 0.2** (deg 6.6→42) | **0.9841** | 0.9551 | **0.9878** | 0.9602 |
| **DrugDrug 0.6** (deg 3.9→2.4) | 0.9838 | **0.9582** | 0.9873 | 0.9618 |
| kNN k=3 all three | 0.9837 | 0.9568 | 0.9851 | 0.9607 |

`LncLnc 0.2` moves **validation and test by the same amount in the same
direction** (+0.0007 / +0.0008), which is what separates a real gain from
fitting the fold being reported. `kNN k=3` is the counter-example: +0.0003 on
val, −0.0019 on test, and should not be used.

**MiDrug is dead on this axis** — all 20 variants land within 0.0002 of the
released graph (0.9549 → 0.9551 at best, 10 seeds).

### It stacks: the final LncDrug position

Graph `DrugDrug 0.6` x the 16-config diverse library x per-metric epochs x
snapshots x rank-averaging (`improve/multiens.py`, `results/mens_g_drug06.csv`):

| metric | fold | paper | **uniform, no fitting** | Δ | greedy on val | Δ |
|---|---|---|---|---|---|---|
| **AUC** | **S_val** | 0.9834 | **0.9847** | **+0.0013** | 0.9856 | +0.0022 |
| **AUPR** | **S_val** | 0.9545 | **0.9619** | **+0.0074** | 0.9687 | +0.0142 |
| AUC | S_test | 0.9857 | **0.9878** | **+0.0021** | 0.9885 | +0.0028 |
| AUPR | S_test | 0.9585 | **0.9676** | **+0.0091** | 0.9700 | +0.0115 |

The graph change is worth +0.0006 val AUC and +0.0026 val AUPR *on top of* the
ensemble, so it is not redundant with it.

MiDrug's best (`results/mens3_mi.csv`, released graph): val 0.9556 / 0.9198,
test 0.9573 / 0.9189 — +0.0001 / +0.0015 on val, +0.0009 / +0.0024 on test.

### The validation ledger, end to end

| stage | LncDrug val AUC | LncDrug val AUPR |
|---|---|---|
| paper | 0.9834 | 0.9545 |
| author's hyperparameters + `posw2` + 10-seed ensemble | 0.9834 | 0.9568 |
| + hyperparameter search redone (ch 64→256) | 0.9839 | 0.9590 |
| + per-metric epochs, snapshots, rank-averaging | 0.9841 | 0.9593 |
| + **similarity-graph sparsification** | **0.9847** | **0.9619** |
| + greedy selection (fitted on val) | 0.9856 | 0.9687 |

Two caveats travel with the top row. The graph threshold is **selected on
validation**, like every other choice in the paper's own Table 2 — the check
that it is not merely fitting that fold is that S_test moves with it, which it
does (+0.0021 / +0.0091). And `greedy` is fitted on validation outright; its
validation figure is not quotable as a gain, though its test figure is.

### Why the graph pays on LncDrug and not on MiDrug

Twenty-six MiDrug graph variants — both aggregations, both widths, coordinate
moves, combinations, kNN — land within ±0.0004 of the released graph. That is
not "the method fails on miRNA"; it is that miRNA has no room for it, and the
reason is measurable.

| task | training positives | ncRNA nodes | direct drug edges per ncRNA |
|---|---|---|---|
| LncDrug | 1267 | 1322 | **0.96** |
| MiDrug | 5394 | 605 | **8.9** |

Similarity edges are a *substitute* for association evidence. A miRNA already
carries ~9 direct drug edges, so its embedding is fixed by the association graph
and the similarity edges are a perturbation on top; a lncRNA carries ~1, so the
similarity graph is doing real work and its construction matters.

That is a testable claim, and `improve/graph_sweep.py --thin-train` tests it:
keep a fraction of MiDrug's training positives, leave the validation and test
folds untouched, and re-measure the spread across graph variants.

| MiDrug training positives | per miRNA | spread in val AUC across graph variants |
|---|---|---|
| 5394 (released) | 8.9 | **0.0002** |
| 1268 (LncDrug's density) | 2.1 | **0.0017** |

Thinned to LncDrug's density, the axis comes alive — **8x the spread — and it
reproduces LncDrug's ranking exactly**: `LncLnc 0.2` first (0.9368), `DrugDrug
0.6` second (0.9366), the released 0.3/0.3/0.3 last (0.9351). Same two winners,
same order, on the other ncRNA type.

The paper's own ablation table already contains this asymmetry and does not
comment on it: removing the LncMi edges costs LncDrug **0.0105** and MiDrug
**0.0037**, a 2.8x difference on the same intervention. The density ratio is
9.3x. Graph augmentation — the paper's central claim — pays in proportion to how
sparse the association graph it is augmenting is, which is why the method's own
ablation is three times weaker on the ncRNA type with four times the edges.

So the practical rule for this codebase: **tune the graph on LncDrug, not on
MiDrug**, and expect the same to hold for any ncRNA type with dense drug
associations. MiDrug's remaining gains come from the loss and the ensemble
(val 0.9556 / 0.9198, test 0.9573 / 0.9189), not from the graph.

## A model class that is not a GNN, and the largest gain in this file

Every ensemble member above is a variant of the same two-layer GeneralConv
encoder. Widths, dropouts, aggregations and seeds change the weights, not the
inductive bias, so the members fail on the same edges — which is why uniform
averaging over 40 of them buys +0.0005.

`improve/gbdt.py` is a different model class: explicit heterogeneous
link-prediction features fed to a gradient-boosted tree
(`HistGradientBoostingClassifier`; lightgbm is not installed and this is its
equivalent). Nineteen features — endpoint degrees, similarity-weighted
collaborative scores on both sides, 3- and 5-hop bipartite path counts, SVD
latent factors of the association matrix, neighbourhood degree, masked maximum
similarity, and the ncRNA→other-ncRNA→drug bridge count that states the paper's
pan-ncRNA premise as a number.

### The leakage guard is what makes it work at all

A training positive `(r,d)` is an edge of the association matrix its own
features are computed from. Two guards: the similarity matrices carry a zero
diagonal, and training rows get **K-fold feature generation** — for fold k every
feature is computed from the matrix with fold k's positives removed, latent
factors included, which is why the SVD lives inside `featurise`.

The `--leaky` control is worth recording because it does not fail the way one
expects. Without the K-fold guard, validation AUC is **0.4997** — chance, not
0.99. The leak is on the training side only, so the model learns a rule ("this
pair's own edge inflates its path count") that cannot fire on a held-out edge,
and transfer collapses completely. The guard is not hygiene here; it is the
difference between a working model and a useless one.

### The blend beats both components in all eight cells

The GBDT is **worse than the GNN alone in every single cell**, and the blend
still beats the GNN everywhere. Spearman correlation between the two scorers is
0.77–0.85 on LncDrug and 0.93 on MiDrug — that gap is the mechanism, and it
predicts the sizes.

| task | metric | GNN alone | GBDT alone | **blend** |
|---|---|---|---|---|
| LncDrug | val AUC | 0.9846 | 0.9800 | **0.9856** |
| LncDrug | val AUPR | 0.9615 | 0.9510 | **0.9642** |
| MiDrug | val AUC | 0.9556 | 0.9542 | **0.9576** |
| MiDrug | val AUPR | 0.9198 | 0.9150 | **0.9218** |

Final position, GNN ⊕ GBDT-v1 ⊕ GBDT-v2, three weights on a 0.04 grid selected
on validation (`improve/blend.py`, `results/gb*.csv`):

| task | metric | paper | S_val | Δ | S_test | Δ |
|---|---|---|---|---|---|---|
| LncDrug | AUC | 0.9834 | **0.9856** | **+0.0022** | **0.9885** | **+0.0028** |
| LncDrug | AUPR | 0.9545 | **0.9642** | **+0.0097** | **0.9720** | **+0.0135** |
| MiDrug | AUC | 0.9555 | **0.9576** | **+0.0021** | **0.9583** | **+0.0019** |
| MiDrug | AUPR | 0.9183 | **0.9218** | **+0.0035** | **0.9188** | **+0.0023** |

**MiDrug's validation AUC is the cell that matters here.** It had refused to
move for the whole investigation — 0.9548–0.9557 across six encoder families,
nineteen mechanism arms, three hundred configurations, twenty-six graph variants
and every ensemble — and a non-GNN scorer that is itself *worse* than the GNN
moves it by +0.0021, twenty times the best previous result on that cell.

The blend weights are chosen on validation, two of them, which is more fitting
than a single threshold; the check is that S_test moves in the same direction in
all four cells, which it does.

### It is still not +0.01 in four cells

One cell reaches it (LncDrug AUPR, +0.0097 on val and +0.0135 on test) and three
do not. The two AUC cells are bounded, not merely unsolved: global AUC is
`0.0036 x AUC_same + 0.9964 x AUC_cross`, so +0.01 needs cross-drug AUC to rise
by 0.0051 (LncDrug) or 0.0082 (MiDrug), and a leave-one-out oracle that reads
the test fold's own labels and is blended at a test-optimal weight is worth
+0.0003 and +0.0007 on that route. Adding a second model class raised the AUC
cells from +0.0013/+0.0001 to +0.0022/+0.0021 — a real doubling and more, and
still an order of magnitude short of +0.01.

## Widening the pool, and the permutation null that nearly cost the result

`improve/altmodels.py` adds three more scorers on the same 19 features —
`extratrees` (bagged random splits), `logreg` (linear, no interactions) and two
unsupervised heuristics (`labelprop`, `mf`) — because the leverage was
model-class diversity, not more GNNs.

**Two of them beat the GNN outright.**

| scorer | val AUC | val AUPR | test AUC |
|---|---|---|---|
| GNN, 40-member diverse ensemble | 0.9846 | 0.9615 | 0.9876 |
| **`extratrees`** | **0.9877** | **0.9713** | **0.9899** |
| `logreg` (linear!) | 0.9846 | 0.9676 | **0.9911** |
| `gbdt` | 0.9800 | 0.9510 | 0.9881 |
| `labelprop` / `mf` | 0.3371 / 0.5396 | — | — |

### The permutation control, done properly

A single label permutation returned validation AUC 0.6498 rather than 0.5, and
the first reading of that was "leak, retract everything". That reading was
wrong, and the error is worth recording because it is the mirror image of
skipping the control.

The null distribution of validation AUC under permutation is **not** a point
mass at 0.5. With 3801 training rows and a 1263-row validation fold, a
noise-fitted model drifts a long way by chance. Over 20 permutations:

| model | real | null mean | null sd | null max | P(null >= real) |
|---|---|---|---|---|---|
| `extratrees` | 0.9875 | **0.5025** | 0.128 | 0.7314 | **0.00** |
| `gbdt` | 0.9800 | 0.4554 | 0.090 | 0.6416 | **0.00** |
| `logreg` | 0.9846 | 0.5153 | 0.123 | 0.7410 | **0.00** |

The null is centred on chance and the single 0.6498 draw sits 1.1 sd above its
mean. There is no leak. **A permutation control needs a null distribution; one
draw from a null with sd 0.13 carries no information in either direction.**

### The same control on the GNN, and what it exposes instead

Permuting the GNN's training labels leaves validation AUC at **0.9733**, against
0.9836 trained honestly. That is not a defect either, and two controls say why:

| | val AUC |
|---|---|
| untrained, randomly initialised GNN (5 seeds) | 0.4146 ± 0.4126 |
| **scorer reading only the drug's training degree — no learning at all** | **0.9742** |
| GNN trained on permuted labels | 0.9733 |
| GNN trained honestly | 0.9836 |

The permuted GNN lands on the zero-learning degree scorer, to three decimals.
The training positives are simultaneously the supervision targets *and* the
message-passing edges, so the graph carries the drug-popularity ordering whether
or not the labels are readable. **Supervised learning is worth +0.0094 on top of
what the input graph alone gives.**

## What the benchmark actually measures

The obvious objection to a tree beating the GNN is information: the tree reads
the similarity matrices as continuous values, while `heter_data_prepare.py`
hands the GNN a version thresholded at 0.3. `improve/feature_groups.py` nests
the feature set to separate that from model class — `degree` (5 features),
`graph` (+ path counts, SVD factors, the cross-ncRNA bridge; every one a
function of edge sets the GNN also receives), `all` (+ the continuous
similarity-weighted scores).

| task | feature set | n | val AUC | val AUPR | test AUC | test AUPR |
|---|---|---|---|---|---|---|
| LncDrug | **degree only** | 5 | **0.9831** | **0.9545** | 0.9882 | 0.9745 |
| LncDrug | graph (GNN has this information) | 12 | **0.9874** | **0.9702** | 0.9904 | 0.9804 |
| LncDrug | all | 19 | 0.9877 | 0.9710 | 0.9897 | 0.9798 |
| | *paper* | | *0.9834* | *0.9545* | *0.9857* | *0.9585* |
| MiDrug | degree only | 5 | 0.9539 | 0.9139 | 0.9553 | 0.9103 |
| MiDrug | graph | 12 | 0.9547 | 0.9152 | 0.9560 | 0.9104 |
| MiDrug | all | 19 | **0.9566** | **0.9186** | 0.9571 | 0.9136 |
| | *paper* | | *0.9555* | *0.9183* | *0.9564* | *0.9165* |

Two things fall out, and they are not the same thing.

**The information objection fails.** Continuous similarity is worth +0.0003 AUC
on LncDrug (0.9874 → 0.9877). Restricted to information the GNN unambiguously
receives, the tree reaches 0.9874 / 0.9702 against the best GNN configuration in
this whole file at 0.9847 / 0.9619. The advantage is the model class, not the
inputs — the decoder `MLP([z_r || z_d])` cannot represent `(A A^T A)[r,d]` or
`(S_r A)[r,d]` at all. On MiDrug the continuous values are worth more (+0.0019),
so the objection has some force there and should be disclosed.

**On LncDrug the published figures are reproduced by node degree alone.** Five
degree features and a tree give **0.9831 / 0.9545** against a printed
0.9834 / **0.9545** — the AUPR agrees exactly. No heterogeneous graph, no
semantic enhancement, no node attributes, no cross-ncRNA edges.

This does **not** replicate on MiDrug, where degree-only falls 0.0016 / 0.0044
short, and that difference is consistent with the density result above: MiDrug
carries 8.9 training drug edges per miRNA against LncDrug's 0.96, so it has
structure beyond degree that LncDrug does not.

Three independent measurements now say the same thing about LncDrug — the pair
decomposition (99.7% of ranking pairs are cross-drug), the permuted-label and
degree-only scorers (0.9733 / 0.9742), and this row (0.9831 / 0.9545). The
improvements above are real and are reported as measured, but the finding that
matters is that **this protocol mostly measures drug popularity**, and a
+0.004 improvement on it should be read in that light.

## Fusing the two model classes into one architecture — it does not work

If a tree on pair features beats the GNN, the natural move is one model that has
both. `improve/hybrid.py` implements it: the decoder is widened by the same
standardised 19-vector, trained end to end, with the same K-fold guard.

| decoder | val AUC | val AUPR | test AUC | test AUPR |
|---|---|---|---|---|
| `none` — `MLP([z_r \|\| z_d])`, the control | 0.9833 | 0.9577 | 0.9872 | 0.9630 |
| `concat` — `MLP([z_r \|\| z_d \|\| phi])` | 0.9828 | 0.9569 | 0.9874 | 0.9627 |
| `additive` — `MLP([z_r \|\| z_d]) + g(phi)` | 0.9832 | 0.9577 | **0.9881** | **0.9665** |

5 seeds, seed sd 0.0006. **On validation the three are indistinguishable**, and
the score-level blend of the same components reaches 0.9880 / 0.9727 — 0.005 and
0.015 above the best of them.

The features are not the problem; `extratrees` reaches 0.9877 on the same phi.
Joint training is: the gradient through 2x256 embedding dimensions dominates a
19-dimensional side path, `concat` is actively worse than the control because
those dimensions only dilute the first Linear, and a small MLP in 300 full-batch
epochs does not learn the thresholded, piecewise structure a tree gets from a
few hundred splits. The two components also want different regularisation, and
end-to-end forces one learning rate and one stopping point on both.

So **two-stage with score-level fusion is the correct architecture here, not a
shortcut** — which is the same conclusion BUDDY (ICLR 2023) argues for keeping
structural features on a pathway that does not compete with the GNN trunk. These
three rows are the ablation that makes that claim rather than assuming it.

## Final position, with confidence intervals

One predictor per task: the four scorers rank-normalised and averaged, weights
chosen on validation by the author's own `(AUC + AUPR)/2` criterion, so a single
model reports both metrics. 2000-resample bootstrap over fold edges.

| task | fold | metric | paper | ours | Δ | 95% CI | paper outside CI |
|---|---|---|---|---|---|---|---|
| LncDrug | val | AUC | 0.9834 | 0.9880 | +0.0046 | [0.9831, 0.9919] | no |
| **LncDrug** | **val** | **AUPR** | 0.9545 | **0.9727** | **+0.0182** | [0.9603, 0.9834] | **yes** |
| LncDrug | test | AUC | 0.9857 | 0.9908 | +0.0051 | [0.9856, 0.9951] | no |
| **LncDrug** | **test** | **AUPR** | 0.9585 | **0.9816** | **+0.0231** | [0.9718, 0.9898] | **yes** |
| MiDrug | val | AUC | 0.9555 | 0.9582 | +0.0027 | [0.9520, 0.9642] | no |
| MiDrug | val | AUPR | 0.9183 | 0.9227 | +0.0044 | [0.9104, 0.9348] | no |
| MiDrug | test | AUC | 0.9564 | 0.9600 | +0.0036 | [0.9540, 0.9662] | no |
| MiDrug | test | AUPR | 0.9165 | 0.9217 | +0.0052 | [0.9083, 0.9344] | no |

Weights: LncDrug `extratrees` 0.50 / `logreg` 0.25 / `gnn` 0.25; MiDrug `gnn`
0.40 / `logreg` 0.30 / `gbdt` 0.20 / `extratrees` 0.10.

**Eight cells improve and two of them survive a bootstrap.** The folds are
small — 1263 validation rows, 421 of them positive — so the bootstrap sd of AUC
and AUPR is 0.002–0.006 here, an order of magnitude above the MiDrug deltas.
That is a resolution limit of the benchmark, not of the method, and it applies
equally to the paper's own printed figures.

The defensible claim is therefore:

> On the lncRNA–drug task AUPR improves on both folds (+0.018 validation,
> +0.023 test; the published figures fall outside the 95% bootstrap interval).
> AUC on that task, and all four miRNA–drug cells, are level or slightly better,
> with differences smaller than within-fold bootstrap noise.

Anything stronger than that does not survive the first bootstrap a reviewer
runs.

## Fitting the fusion on the training fold instead of the reported one

`poolplus.py` and `feature_extra.py` closed the mechanism search with a result
that points somewhere else. `mlp` is the highest single-member **test** AUC on
both tasks (LncDrug 0.9917, MiDrug 0.9593), above every GNN configuration and
every tree in the pool, and `xt35` improves LncDrug on all four cells — and the
author's `(AUC + AUPR)/2` validation rule assigns both of them weight **0.00**.
So the binding constraint was no longer "is there a better member", it was that
the selector is a 1263-row fold (LncDrug; 3234 on MiDrug) whose bootstrap sd is
0.002–0.006, an order of magnitude above the differences it is ranking.

`improve/stack.py` tests three changes against that, all of them symmetric
across the two ncRNA types by construction. **One of the three works, and it
works on one task.**

| # | change | LncDrug | MiDrug |
|---|---|---|---|
| 1 | OOF stacking — meta-learner fitted on the training fold's out-of-fold member scores | −0.0005 test AUC vs the val-fitted combiner, but nothing fitted on val except one scalar | −0.0006 test AUC, same |
| 2 | degree-gated weights | **−0.0014 … −0.0060** test AUPR; `--gate both` collapses to val AUC 0.65 | **−0.015** test AUPR |
| 3 | extra sampled negatives | **+0.0025 test AUPR, CI [+0.0005, +0.0050]** | +0.0005, CI [−0.0016, +0.0026] |

### Change 1: it buys honesty, not accuracy

Every member already gets K-fold-guarded features (`gbdt.py`: for fold k,
features come from the association matrix with fold k's positives deleted).
Nesting one more level — fit the member on folds ≠ k, predict fold k — turns
the training fold into honest out-of-fold member scores, so the combiner can be
fitted on 3801 rows (LncDrug) / 9714 (MiDrug) instead of 1263 / 3234.

`val_grid` in the table below is the combiner being replaced, run on
**identical member scores**: non-negative weights on a simplex grid, argmax of
the author's own `(AUC + AUPR)/2`, fitted on validation.

| task | combiner | val AUC / AUPR | test AUC / AUPR | fitted on val |
|---|---|---|---|---|
| LncDrug | `val_grid` | 0.9880 / 0.9759 | **0.9921 / 0.9856** | 6 weights |
| LncDrug | `nnls` OOF ⊕ GNN | 0.9878 / 0.9747 | 0.9916 / 0.9846 | 1 scalar |
| LncDrug | `uniform` | 0.9871 / 0.9744 | 0.9913 / 0.9853 | **nothing** |
| MiDrug | `val_grid` | 0.9585 / 0.9244 | **0.9604 / 0.9228** | 6 weights |
| MiDrug | `nnls` OOF ⊕ GNN | 0.9582 / 0.9225 | 0.9598 / 0.9212 | 1 scalar |
| MiDrug | `uniform` | 0.9578 / 0.9192 | 0.9590 / 0.9178 | **nothing** |

The prediction was that the val-fitted weights were fitting noise and would lose
on test. **They do not** — `val_grid` is the best test column on both tasks. Its
advantage is small (0.0005–0.0010 AUC) and it costs six weights fitted on the
reported fold, so the trade is real and should be stated rather than hidden: the
OOF stack gives up ~0.0006 test AUC to make S_val a held-out number. On a
2-member pool the OOF stack *did* win on test (0.9895 vs 0.9884), which is why
this looked promising; with the full 5-member pool it does not, and the
2-member result was the artefact.

The GNN enters as one val-selected scalar because it has no training-fold
scores — producing them means retraining it K times, which is the open item on
this route.

### Change 2: degree-gated weights are negative on both tasks

The pair oracle in `altmodels.py` says "pick whichever member ranks each pair
correctly" is worth 0.9904 against a 0.9846 blend, and a constant weight vector
cannot recover any of that. Making the weights a function of the drug's
training degree (3 bins, interaction terms) is the cheapest way to try:

| task | arm | test AUC | test AUPR |
|---|---|---|---|
| LncDrug | `nnls`, no gate | 0.9915 | 0.9856 |
| LncDrug | `nnls`, gate on drug degree | 0.9899 | 0.9817 |
| LncDrug | `nnls`, gate on both degrees (9 bins) | **0.6432** | 0.5913 |
| MiDrug | `nnls`, no gate | 0.9592 | 0.9189 |
| MiDrug | `nnls`, gate on drug degree | 0.9454 | 0.9126 |

Nine bins x 5 members is 54 parameters and it degenerates completely. Three bins
is 18 and it loses by 0.0016–0.014. The 83% of the pair oracle that a flat
weight cannot reach is **not** reachable by conditioning on degree either — the
members do not have complementary degree ranges, they have complementary
individual edges, and nothing at endpoint granularity separates those.

### Change 3: more negatives, and it is real on LncDrug

Every tree in this repo is fitted on the frozen 2:1 labelled set — 3801 rows out
of 1322 x 216 = 285,552 possible pairs. The GNN arm `neg` (resample negatives
each epoch) is *not* this: it reshuffles a fixed budget rather than enlarging it,
and it was never tried on the tree members at all. Extra negatives are sampled
from pairs absent from the **training** association matrix (strict: nothing
about the val/test edge sets is read), featurised through the same per-fold
guard, and used to fit the base members only — the combiner still sees the
protocol's 2:1 balance and the evaluation negatives are untouched.

At 5x the positive count it lifts the members, not just the fusion — LncDrug
`mlp` val AUPR 0.9648 → 0.9736, `gbdt` 0.9525 → 0.9610 — and every combiner
moves with them. Paired bootstrap over test edges, 2000 resamples
(`improve/boot_pair.py`):

| task | combiner | metric | 0 → 5x | delta | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|---|
| **LncDrug** | `val_grid` | test AUPR | 0.9831 → 0.9856 | **+0.0025** | **[+0.0005, +0.0050]** | **0.997** |
| **LncDrug** | `val_grid` | test AUC | 0.9913 → 0.9921 | **+0.0008** | **[+0.0001, +0.0018]** | **0.982** |
| **LncDrug** | `uniform` | test AUPR | 0.9819 → 0.9853 | **+0.0034** | **[+0.0003, +0.0076]** | **0.988** |
| LncDrug | `uniform` | test AUC | 0.9903 → 0.9913 | +0.0010 | [−0.0000, +0.0024] | 0.967 |
| MiDrug | `val_grid` | test AUPR | 0.9223 → 0.9228 | +0.0005 | [−0.0016, +0.0026] | 0.698 |
| MiDrug | `val_grid` | test AUC | 0.9602 → 0.9604 | +0.0002 | [−0.0007, +0.0010] | 0.651 |
| MiDrug | `uniform` | test AUPR | 0.9186 → 0.9178 | **−0.0008** | [−0.0037, +0.0020] | 0.303 |

Three LncDrug cells have a 95% interval that excludes zero, including the
`uniform` combiner which fits nothing on any fold. **All eight MiDrug cells span
zero**, the best reaches P(Δ>0) = 0.90 on validation only, and the unfitted
combiner is negative. Sampling 10x instead of 5x saturates (LncDrug `val_grid`
test 0.9917/0.9848; MiDrug 0.9607/0.9232), so the mechanism is not
budget-starved at 5x.

**It also breaks the linear member on MiDrug.** `logreg` val AUPR falls
0.9163 → 0.8722 at 5x. The obvious diagnosis is the shifted class prior, and
`--balance-extra` tests it by downweighting the negatives so their total mass is
unchanged: it recovers only to 0.8926. So it is not a prior shift — a single
linear boundary genuinely cannot cover the wider negative region, and that drags
`uniform` and `nnls` down with it. The balance correction is kept because it
helps LncDrug's fitted combiners (+0.0004 test AUC), not because it fixed
MiDrug.

### Why the asymmetry, again

This is the density result for the fourth time. MiDrug carries 8.9 training drug
edges per miRNA against LncDrug's 0.96, so its collaborative features are
already estimated from enough evidence that widening the negative pool adds
little; LncDrug's are estimated from ~1 edge per node, so the extra negatives
sharpen a boundary that was genuinely underdetermined. Graph tuning
(0.0017 vs 0.0002 spread), the `w/o_LncMi` ablation (0.0105 vs 0.0037),
`feature_extra`'s +6 columns (LncDrug all four cells, MiDrug flat-negative), and
now extra negatives all split the same way and in the same ratio.

### Position after this section

| task | fold | metric | paper | previous best in this file | now |
|---|---|---|---|---|---|
| LncDrug | test | AUC | 0.9857 | 0.9908 | **0.9921** |
| LncDrug | test | AUPR | 0.9585 | 0.9816 | **0.9856** |
| LncDrug | val | AUC | 0.9834 | 0.9880 | 0.9880 |
| LncDrug | val | AUPR | 0.9545 | 0.9727 | **0.9759** |
| MiDrug | test | AUC | 0.9564 | 0.9600 | 0.9604 |
| MiDrug | test | AUPR | 0.9165 | 0.9217 | 0.9228 |
| MiDrug | val | AUC | 0.9555 | 0.9582 | 0.9585 |
| MiDrug | val | AUPR | 0.9183 | 0.9227 | 0.9244 |

Part of the LncDrug movement is the pool expansion `poolplus`/`feature_extra`
already measured (0.9908 → 0.9913 test AUC, 0.9816 → 0.9831 AUPR); the rest is
change 3. **MiDrug's four cells move by 0.0003–0.0017 and not one of them
survives a bootstrap** — unchanged from every other section in this file.

The question this section was opened to answer was "is there a mechanism that
improves both ncRNA types". The answer it produces is no: of three changes
designed to be type-symmetric, two are negative on both and one is significant
on lncRNA only. `posw2` remains the only mechanism in this file that is
positive on both, and it is +0.0021 / +0.0031.

### Reproducing

```bash
python improve/stack.py --run-dir $R --net-dir $R/2.construct_network \
    --task LncDrug --gnn $R/9.improve/gnn_lnc.npz --seeds 666 0 1 --folds 5 \
    --gate none --extra-neg 5 --balance-extra --metas nnls \
    --dump-scores $R/9.improve/stk_LncDrug_n5.npz --out /tmp/stk.csv

python improve/boot_pair.py --a $R/9.improve/stk_LncDrug_n0.npz \
    --b $R/9.improve/stk_LncDrug_n5.npz --key val_grid --boot 2000
```

`--gate {drug,both}` reproduces the negative arms; `--extra-neg 0` is the
control every delta above is measured against, on the same member scores.

## What this file is actually for

The +0.004-to-+0.023 improvements are reported because they were measured, but
they are not the result. Three findings do not depend on the fourth decimal and
get stronger, not weaker, when someone reproduces them:

1. **The protocol mostly measures drug popularity, on LncDrug.** Three
   independent measurements agree: 99.7% of ranking pairs are cross-drug; a
   zero-learning drug-degree scorer reaches 0.9742 and a permuted-label GNN
   0.9733 against an honestly trained 0.9836; and five degree features with a
   tree reproduce the published 0.9834 / 0.9545 as 0.9831 / **0.9545**.
   Supervised learning is worth **+0.0094** over the input graph alone.
2. **The published ablation margins are an epoch-rule artefact.** Under one
   uniform best-validation-epoch rule they fall 5–7x (w/o_attribute 0.0135 →
   0.0044), and per-variant hyperparameter retuning changes nothing — so it is
   the last-epoch read, not the shared hyperparameters.
3. **A tree on explicit pair features beats the GNN on information the GNN also
   has** (0.9874 / 0.9702 against the best GNN configuration here at
   0.9847 / 0.9619), because `MLP([z_r || z_d])` cannot represent
   `(A A^T A)[r,d]`. Fusing the two into one end-to-end architecture does not
   work; score-level fusion does.

## Widening the pool further: five more model classes, one real find

`improve/poolplus.py` added five members to the fusion pool on the same 19
K-fold-guarded features and the same GNN scores, chosen to occupy the functional
middle the pool lacks — `rf` (bagged best-cut splits, vs extratrees' random
cuts), `balrf` (class-weighted forest), `mlp` (a shallow nonlinear member
between the single linear logreg and the tree families), `subspace` (random
feature subspaces) and `posfocus` (extratrees trained on all positives + the
hardest negatives only, targeting the ranking head that AUPR measures).
5 seeds, each metric averaged over seeds, per `altmodels.py`.

Single-member validation AUC (val / test AUC):

| member | LncDrug val | LncDrug test | MiDrug val | MiDrug test |
|---|---|---|---|---|
| **`mlp`** | 0.9853 | **0.9910** | **0.9571** | **0.9593** |
| `subspace` | 0.9860 | 0.9881 | 0.9567 | 0.9565 |
| `posfocus` | 0.9865 | 0.9888 | 0.9568 | 0.9568 |
| `rf` | 0.9828 | 0.9868 | 0.9558 | 0.9558 |
| `balrf` | 0.9821 | 0.9858 | 0.9562 | 0.9562 |
| *gnn (diverse ens)* | *0.9846* | *0.9876* | *0.9556* | *0.9573* |
| *extratrees* | *0.9877* | *0.9899* | *0.9547* | *0.9560* |
| *logreg* | *0.9846* | *0.9911* | *0.9562* | *0.9571* |

**`mlp` is the best single-member test-AUC scorer on both tasks** (0.9910
LncDrug, 0.9593 MiDrug) — above every GNN configuration, gbdt, extratrees and
logreg. It is the functional middle class the pool was missing: it represents
feature interactions without thresholds, which neither logreg (no interactions)
nor a tree (thresholds) does.

### Why it does not move the headline

Re-running the author's `(AUC + AUPR)/2` validation-selected fusion with `mlp`
in the pool:

- **LncDrug: validation assigns `mlp` weight 0.00.** Its val AUPR (0.9606) sits
  below extratrees (0.9713), and the joint grid keeps the existing
  extratrees .50 / logreg .25 / gnn .25. Forcing a test-side reweight
  (extratrees .35 / logreg .40 / gnn .10 / mlp .15) raises test AUC +0.0006 and
  test AUPR +0.0023 but costs val criterion — a trade the author's rule does not
  take, and the same pattern as every test-side gain in this file.
- **MiDrug: validation assigns `mlp` ~0.27–0.40**, shifting weight off gnn.
  The gain is real but small: val AUPR 0.9225 → 0.9229 (+0.0004), test flat
  (0.9596 / 0.9208 in both configurations).

The four other members are dead on arrival for the fusion: `rf`/`balrf` are
redundant with extratrees (same family, worse tuning), and `subspace`/`posfocus`
receive zero weight in every fusion — the pool's existing diversity already
covers what they add.

### Verdict

The pool-expansion hypothesis is confirmed but bounded: a genuinely new model
class (MLP) exists and is the best single-member test-AUC scorer on both tasks,
but on LncDrug the validation criterion selects against it and on MiDrug it
buys +0.0004 val AUPR / flat test. That is consistent with the ceiling measured
throughout this file — the validation fold has already been squeezed, so a
member that is test-good but val-weak does not change a validation-selected
report. The one-line claim from this branch: *an MLP on the 19 pair features is
the highest single-member test AUC of any scorer in this repo, but validation
selection does not put it in the fusion.*

## Continuous-feature enrichment: real on the single member, absent in the fusion

`improve/feature_extra.py` adds six continuous columns to the 29 base features —
`katz` (geometric-decay path sum), `deg_gap` / `deg_ratio` (degree difference and
ratio between the endpoints), `wpath3` (similarity-weighted 3-path `S_r A A^T
S_d`), and `adamic_adar_r/d` (degree-refined endpoint propensities, i.e. a
sharper version of the exact cross-drug signal the metric reads).  All six are
functions of the same (A, S_r, S_d) as the existing set, so the K-fold guard
covers them unchanged.  `improve/poolplus.py` now exposes this as the `xt35`
pool member.

Single-member extratrees, 5 seeds, current-code baseline:

| task | set | val AUC | val AUPR | test AUC | test AUPR |
|---|---|---|---|---|---|
| LncDrug | base29 | 0.9864 | 0.9651 | 0.9888 | 0.9778 |
| LncDrug | **+6** | **0.9870** | **0.9674** | **0.9894** | **0.9782** |
| MiDrug | base29 | 0.9568 | 0.9188 | 0.9568 | 0.9117 |
| MiDrug | +6 | 0.9565 | 0.9183 | 0.9561 | 0.9118 |

**LncDrug improves on all four cells** (val AUPR +0.0023, test AUC +0.0006),
stable across 5 seeds and reproduced in a per-feature leave-one-out ablation:
dropping any of the six costs 0.0008–0.0016 val AUPR, so the gain is not one
feature but the block.  **MiDrug is flat-to-negative on every cell**, the same
density asymmetry measured throughout this file (LncDrug 0.96 direct drug edges
per ncRNA vs MiDrug 8.9 — the sparse side has room for similarity structure, the
dense side does not).

### The fusion does not reward it, and the reason is the same as the MLP

Re-running the author's `(AUC+AUPR)/2` validation-selected fusion with `xt35` in
the pool: validation assigns it **weight 0.00**, on both the historical and the
current-code baselines, because `xt35` and `extratrees` are the same model class
(Spearman 0.95+ on both folds) and the +6 information is fully absorbed by the
existing member's weight.  Swapping `xt35` for `extratrees` at fixed weights
does not help either (test AUC 0.9906 vs 0.9908), because the single-member gain
is inside the model's ranking head, not in new pair-level wins the blend can
collect.

### Verdict

The feature-enrichment hypothesis is confirmed at the model level and denied at
the fusion level.  Six new continuous features improve the strongest tree member
on LncDrug by a stable +0.0023 val AUPR / +0.0006 test AUC — a real, reproducible
model-level result that matches the file's density finding exactly.  But a
validation-selected score-level fusion of members that are 0.95-correlated with
what is already in the pool does not pay for that gain; it is the MLP result
again, in the same-model-class version.  The one-line claim from this branch:
*continuous feature enrichment is a genuine single-member improvement on the
task with headroom (LncDrug) and a loss on the saturated one (MiDrug), and it is
invisible to the fusion because it is redundant with the member it would
replace.*

## AUPR-targeted fusion: the criterion is not the lever, the normalisation is

AUPR is the metric with the real residual (LncDrug test +0.023, ~3x AUC's), and
its residual sits at the top of the ranking.  The published fusion selects its
weights on validation by the author's own `(AUC+AUPR)/2`, so a single weight
vector reports both metrics.  `improve/aupr_fusion.py` tests the two untried
levers on that rule — select the weight for **AUPR alone**, and fuse on **raw
probabilities instead of ranks** (AUPR is scale-sensitive at the head; AUC is
not).

Members gnn / extratrees / logreg / gbdt from the historical `alt_*.npz` (the
pool the published fusion is built on), grid-searched on validation:

| task | normalisation | rule | S_val AUC / AUPR | S_test AUC / AUPR |
|---|---|---|---|---|
| LncDrug | rank | joint | 0.9879 / 0.9729 | 0.9909 / 0.9819 |
| LncDrug | rank | **AUPR-only** | 0.9879 / 0.9729 | 0.9908 / 0.9823 |
| LncDrug | **raw** | joint | **0.9882 / 0.9738** | **0.9912 / 0.9814** |
| LncDrug | raw | AUPR-only | 0.9881 / 0.9737 | 0.9912 / 0.9818 |
| MiDrug | rank | joint | **0.9580 / 0.9225** | 0.9594 / 0.9203 |
| MiDrug | rank | AUPR-only | 0.9578 / 0.9225 | 0.9593 / 0.9204 |
| MiDrug | raw | joint | 0.9573 / 0.9216 | 0.9590 / 0.9208 |

Two findings, one dead and one live.

**Selecting the fusion weight for AUPR alone buys nothing.** On both tasks the
AUPR-only rule lands on the same val AUPR as the joint rule (0.9729, 0.9225) —
the joint criterion's argmax already sits at the AUPR optimum, so the extra
degree of freedom is unused.  This closes the last plausible "point the
validation rule at AUPR" move: the AUPR residual is not reachable by re-pointing
the weight selection, any more than by the focal/cost-sensitive losses tried
earlier.

**Fusing on raw probabilities instead of ranks is task-dependent and, where it
helps, helps validation.**  LncDrug's raw-probability fusion beats the published
rank-normalised one on val AUPR (+0.0011, 0.9738 vs 0.9727) and val AUC
(+0.0002), with test AUC also up (+0.0004) — but test AUPR is flat-to-negative
(0.9814 vs 0.9816 at the joint weights, +0.0007 at the AUPR-only weights).
MiDrug is the opposite: rank fusion wins on val AUPR (0.9225 vs 0.9216) and the
raw fusion only edges test AUPR (+0.0005) at a cost to both validation metrics.
There is no normalisation that wins on both folds or both metrics, and the
validation-selected choice differs by task — the same "the validation fold has
been squeezed" pattern, now on the input transform instead of the model class.

**Verdict.**  The AUPR-targeted fusion direction is closed.  Re-pointing the
weight-selection criterion at AUPR is exactly neutral on both tasks, and the
raw-vs-rank normalisation is a task-specific choice that trades val AUPR against
test AUPR in both directions — not a lever on the +0.023 LncDrug AUPR residual.
Consistent with everything above: the residual is real but the validation fold
cannot see how to collect it.

## GNN embeddings as tree features: the reverse hybrid is also negative

`improve/hybrid.py` fed the 19 pair features INTO the GNN decoder and failed —
the embedding dimensions swamp the side path.  The untried reverse is to feed
the GNN's own learned embeddings INTO the tree member (`improve/gnn_emb_feat.py`),
so the tree sees the latent structure the GNN recovered as explicit numbers and
can learn interactions between the two channels that score-level fusion (linear
in ranks) cannot.  Features: `emb_dot`, `emb_norm`, `emb_cos`, `emb_dot_a`
(z·z, |z_r−z_d|, cosine, standardised dot) plus 12 leading principal components
of the concatenated pair.

The leak guard is harder than in gbdt.py: a node embedding is a function of the
whole message-passing graph, so a training positive leaks into both endpoint
embeddings.  The guard is per-fold GNN training — for fold k, a GNN is trained
on the graph with fold k's positives removed and that fold's rows get its
embeddings; val/test use a full-training-graph GNN.  (The `--leaky` control
confirms it matters: full-graph embeddings on training rows collapse the tree
to val ~0.93, the same "leak is training-side, transfer collapses" failure as
gbdt.py.)

extratrees, 3 seeds, on the base 29 + 6 continuous features with and without the
16 embedding columns:

| task | set | val AUC | val AUPR | test AUC | test AUPR |
|---|---|---|---|---|---|
| LncDrug | base+extra (35) | 0.9869 | 0.9678 | 0.9892 | 0.9782 |
| LncDrug | **+emb (51)** | 0.9864 | **0.9606** | 0.9882 | 0.9766 |
| MiDrug | base+extra (35) | 0.9562 | 0.9179 | 0.9558 | 0.9113 |
| MiDrug | +emb (51) | 0.9556 | 0.9176 | 0.9569 | 0.9140 |

**Negative on LncDrug (val AUPR −0.0072, test AUPR −0.0016) and noise on
MiDrug** (val −0.0003 AUPR, test +0.0027 AUPR).  The diagnostic explains why:
the GNN's `emb_dot` alone scores only 0.761 AUC on LncDrug, an order of
magnitude below the explicit `log_deg` features — the latent structure the GNN
recovered on this protocol *is* drug popularity, which the explicit degree
features already state more cleanly, so 16 correlated embedding dimensions only
dilute the tree's splits.

This is the same coin as `hybrid.py`: feeding GNN structure into the feature
path is redundant with what the explicit features already carry, exactly as
feeding explicit features into the GNN decoder was redundant with what the
embeddings carry.  Both directions of the two-channel idea fail on this
protocol because the information is saturated on both sides.  The one-line
claim from this branch: *GNN embeddings add no information to the tree members
that the explicit pair features do not already state, and on the task with
headroom (LncDrug) they actively hurt.*

## Finer fusion weights: the step-0.1 grid was already at the optimum

The headline `val_grid` numbers (LncDrug test 0.9921 / 0.9856, MiDrug 0.9604 /
0.9228) were found by `improve/stack.py` on a simplex grid with `--weight-step
0.1`, i.e. weights quantized to multiples of 1/10.  The last plausible lever on
the AUC/AUPR headline was that this quantization leaves weight on the table.
`improve/fine_search.py` re-searches the SAME six member scores (extratrees,
xt35, mlp, gbdt, logreg, gnn) with a multi-resolution scheme — a coarse simplex
pass then a local refinement at 0.001-0.002 around the argmax — under both the
author's `(AUC+AUPR)/2` criterion and an AUPR-only criterion.

Both tasks, both objectives: **the fine search lands on the same weights and
the same numbers**.  LncDrug coarse-15 gives val 0.9881/0.9759, test
0.9920/0.9853 against the grid's 0.9921/0.9856; MiDrug is identical to four
decimals (0.9604 / 0.9228).  The criterion surface is flat around the optimum —
tiny weight perturbations cost or gain nothing, and the joint and AUPR-only
objectives agree on the argmax.

**Verdict.**  The fusion-weight search is closed.  The step-0.1 grid did not
quantize away anything recoverable: the pool's ceiling under validation
selection is LncDrug test AUC 0.9921 / AUPR 0.9856 and MiDrug 0.9604 / 0.9228,
and no amount of weight resolution moves it.  This was the last untried degree
of freedom on the AUC/AUPR headline that did not refit a member or change the
selection fold.

