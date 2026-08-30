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

## Why the protocol has no headroom

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
The protocol is saturated. Anything that makes the model more relational moves
away from the metric; anything that makes it better at reading degree is
redundant, because it already does.

## What is left if the number has to go up

- **Hyperparameter search on validation** (`find_best_params.py`) is untried
  here and is the only honest lever remaining. Expect single thousandths.
- **Reporting the max over seeds** gives LncDrug 0.9873 / AUPR 0.9669, above the
  published 0.9857 / 0.9585. This is seed selection, not an improvement, and the
  spread it is drawn from is 0.0005 wide. It should not be reported as a gain.

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
