# Differentiable optimal-transport (Sinkhorn) head: results

Date 2026-09-18. `algo_ot_v1.py` / `cells_ot` (bank-only), `algo_ot_v2.py` / `cells_ot2`
(bank + provenance). Same 7-split gate and controls as the rest.

## The architecture

Each entity is a set of study tokens (same construction as the set-Transformer scheme:
`MiniLM(text_ctx 384) ++ [year, throughput, effect_resistant, effect_sensitive]`).
The pair score is produced by solving an **entropic optimal-transport problem between
the two study distributions inside the network**:

```
xr, xd = normalize(proj(tokens of r)), normalize(proj(tokens of d))
C_ij   = 1 - cos(xr_i, xd_j)
P      = argmin_{P in U(a,b)} <P, C> - eps H(P)      (log-domain Sinkhorn, 20 iters)
z      = [ <P,C>, max P, mass, eps, <P,C>/mass, h_r, h_d, bank ]
score  = MLP(z)
```

`eps` is learned; the Sinkhorn iterations are differentiable; trained end-to-end with
BCE, gradient clipping 5.0. Cross-fitting and the permuted-record control follow the
other schemes.

## Result 1 — novel architecture on the accepted bank (`ot`)

| surface | MiDrug ΔAUC/ΔAP | LncDrug ΔAUC/ΔAP |
|---|---:|---:|
| inner | +0.00032 / +0.00072 | +0.00002 / +0.00003 |
| resplits | +0.00052 / +0.00112 | +0.00026 / +0.00075 |
| extra | +0.00044 / +0.00078 | +0.00018 / −0.00024 |

**Tolerance pass at w=0.05** (worst LncDrug AP cell −0.00078); control `ot_perm` fails
(MiDrug inner +0.00014/+0.00047 and MiDrug min dAP −0.00013). The OT member is the
strongest *single* novel member found (standalone MiDrug inner AUC/AP 0.9583/0.9218,
above the classical HGB/ET/MLP/LR pool), yet its fused gain stays small because the
reference is already at the ceiling.

## Result 2 — it too collapses onto the counts (`ot2`)

With the provenance columns added:

| recipe | MiDrug inner ΔAUC/ΔAP |
|---|---:|
| `ot` (real records) | +0.00136 / +0.00369 |
| `ot_perm` (records permuted) | **+0.00120 / +0.00400** |

Again the permuted-record control is equal (AP slightly better) to the real one. The
transport plan carries nothing beyond the provenance count columns.

## Combined reading of schemes S2 and S3

Both genuinely novel architectures (study-set cross-attention; differentiable OT)
behave identically: they pass the tolerance gate on the accepted bank with a failing
record-permutation control, but once the provenance count block is present their
contribution vanishes (control ≥ real). The study hypergraph's usable signal is exactly
the co-membership that scalar counts already capture; there is no residual higher-order
structure for attention or transport to model on this metric. This extends the
87-mechanism saturation from graph operators to neural architectures.

The `ot` head is still a valid, novel, verified technique for the bank-only setting
(+0.0003…+0.0008 MiDrug with LncDrug preserved), but it is not an improvement over the
provenance-information configuration.

## Artifacts

`algo_ot_v1.py`, `algorithmic_next_20260917/cells_ot/`; `algo_ot_v2.py`,
`cells_ot2/`; gate table `/tmp/opencode/eval_ot_gates.csv`.
