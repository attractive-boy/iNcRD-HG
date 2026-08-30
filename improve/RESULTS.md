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
