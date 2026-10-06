# Three algorithm-level schemes that pass the verification gate

Date 2026-09-17. Harness: frozen TabICLv2 `tiered_0p5000` reference, accepted
shared-anchor fusion `sigmoid((1-w)·logit(ref) + w·comp)`, seven splits
(inner ×3 seeds, resplits 42/2026/31415, extra 7/77/777). Every scheme has a
label-preserving control that must fail. No default model replaced; no held-out test
source opened. Files under `improve/research/algorithmic_next_20260917/`.

All three schemes take the same accepted input bank — `context76` plus the provenance
block established in the previous round — and change the **algorithm**, not the input:

| # | scheme | level | comp | w | MiDrug inner ΔAUC/ΔAP | Mi resplit | Mi extra | LncDrug inner ΔAUC/ΔAP | Lnc min (AUC/AP) | gate | control |
|---|---|---|---|---|---:|---:|---:|---:|---|:--:|
| 1 | **Neural member replaces the classical pool** (`pv_bce`) | model class | 2-layer MLP on `[bank\|prov]`, BCE | 0.10 | +0.00180 / +0.00561 | +0.00220 / +0.00594 | +0.00250 / +0.00559 | +0.00056 / +0.00145 | +0.00023 / +0.00049 | **strict pass** | `pv_bce_perm` negative |
| 2 | **Cross-drug pairwise ranking objective** (`pv_rank`) | objective | same net, `softplus(-(s_pos−s_neg))` over cross-drug pairs | 0.10 | +0.00189 / +0.00595 | +0.00231 / +0.00617 | +0.00255 / +0.00555 | +0.00037 / +0.00101 | +0.00031 / +0.00003 | **strict pass** | `pv_rank_perm` negative |
| 3 | **Two-part popularity-offset structure** (`offset2`) | model structure | `score = β·(log1p deg_r+log1p deg_d) + MLP([bank\|prov])`, β learned end-to-end | 0.05 | +0.00125 / +0.00410 | +0.00140 / +0.00383 | +0.00143 / +0.00313 | +0.00043 / +0.00115 | +0.00023 / +0.00052 | **strict pass** | `offset2_perm` negative |

"strict pass" = MiDrug inner mean AUC/AP up with ≥2/3 seeds positive, MiDrug AUC/AP
positive **in every** resplit and extra split, and LncDrug AUC/AP **≥ reference in every**
seed/split. All three meet it.

Controls (permuted labels / permuted offset) are negative everywhere: `pv_bce_perm` gives
MiDrug inner −0.00081/−0.00175, `pv_rank_perm` −0.00096/−0.00161, `offset2_perm`
−0.00012/−0.00020.

## Extra: a pure-algorithm scheme with no new columns

`aggrank` (Borda rank-average of the four accepted members, `algo_aggregate_v1.py`,
`cells_algoD`) changes only the fusion operator on the plain `context76` bank:
tolerance pass at every w (MiDrug inner +0.00003…+0.00010, LncDrug never below −0.00027),
but the MiDrug gain is ~1e-4, so it is only a robustness/floor result.

## What does not pass (recorded)

`algo_pairwise_v1.py` (pairwise on context76), `algo_offset_v1.py` (offset on context76),
`algo_stack_v1.py` (stacked ranking meta-learner), `algo_gate_v1.py` (label-free adaptive
shrinkage), `algo_aggregate_v1.py` median/trimmed/prob, and `algo_prov_v2.py` listwise
InfoNCE (diverges) all fail. Consistent with the 87-mechanism saturation: the algorithmic
lever alone does not move the metric; it pays off only on the provenance information.

## Reproduction

```bash
# scheme 1+2 (pv_bce / pv_rank); control added in place
python improve/research/algorithmic_next_20260917/algo_pairwise_prov_v1.py <TASK> <SURFACE> cells_algoE
python improve/research/algorithmic_next_20260917/algo_pv_bce_perm_v1.py <TASK> <SURFACE> cells_algoE
# scheme 3 (offset2)
python improve/research/algorithmic_next_20260917/algo_prov_v2.py <TASK> <SURFACE> cells_algoF
# gates
python improve/research/provenance_next_20260917/evaluate_generic_v1.py \
  improve/research/algorithmic_next_20260917/cells_algoE '{task}_{surface}.npz' \
  member pv_bce,pv_bce_perm,pv_rank,pv_rank_perm 0.1  /tmp/evalE
python improve/research/provenance_next_20260917/evaluate_generic_v1.py \
  improve/research/algorithmic_next_20260917/cells_algoF '{task}_{surface}.npz' \
  member offset2,offset2_perm 0.05,0.1,0.2 /tmp/evalF
```
Run with `OMP_NUM_THREADS=1` (torch/sklearn oversubscribe otherwise).

**Note on the evaluator:** cached `logit_*` arrays are already logits; the gate computes
`sigmoid((1-w)·logit(ref) + w·comp)` and must not apply `logit()` to `comp` again.
