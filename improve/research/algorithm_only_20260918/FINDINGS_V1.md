# Algorithm-only techniques on the accepted bank: five new families, none passes the strict gate

Date 2026-09-18. Scope (chosen by the user): change only the algorithm — objective,
model class or fusion — and add **no new feature column**. Same harness as the
provenance / ALGO rounds: frozen TabICLv2 `tiered_0p5000` reference, accepted
shared-anchor fusion `sigmoid((1-w)·logit(ref) + w·mean logit(members))`, seven splits
(inner ×3 inference seeds; resplits 42/2026/31415; extra 7/77/777), label-permutation
control per family. Members are fitted on the frozen `context76` bank only.

## Families implemented and their 7-split gates

All values are Δ vs the frozen reference at the best (largest-signal) weight; the gate
uses every split individually.

| # | family (level) | member | MiDrug inner ΔAUC/ΔAP | Mi resplit | Mi extra | control | gate |
|---|---|---|---|---|---|---|---|
| 1 | feature-token Transformer (model class) | `fttrans` | −0.00006 / −0.00053 | — | — | perm ≥ real | **fail** |
| 2 | explicit feature-crossing net, DCN-v2 (model class) | `dcn` | −0.00116 / +0.00285 | −0.00165 / −0.00309 | −0.00072 / −0.00073 | — | **fail** |
| 3 | group-DRO over drugs (objective/robustness) | `dro` | −0.00025 / +0.00027 | −0.00023 / +0.00012 | −0.00054 / −0.00101 | — | **fail** |
| 4 | LambdaRank objective, |Δsig| weighting (objective) | `lambdarank` | +0.00000 / +0.00072 | −0.00008 / +0.00032 | +0.00002 / +0.00011 | `lambdarank_perm` negative | **fail (2/7 splits −1e-4)** |
| 5 | two-sided propensity correction, cross-fitted (structure) | `deg2` | −0.00127 / −0.00139 | −0.00094 / −0.00111 | −0.00108 / −0.00157 | `deg2_perm` ≈ 0 | **fail** |
| 6 | **bagged LambdaRank, K=5 seeds (objective + variance reduction)** | `lrbag` | **+0.00017 / +0.00039** @w=0.04 | **+0.00004 / +0.00020** | **+0.00011 / +0.00019** | `lrbag_perm` negative | **fail (2/7 splits by <5e-5)** |

Full gate tables: `artifacts/gate_families_gates.csv`, `gate_lrbag_gates.csv`,
`gate_lrbag_fineweights_gates.csv`, `gate_propensity_gates.csv` (metrics CSVs alongside).

## The closest result, and why it still fails

Bagged LambdaRank (`lrbag`, 5 initialisations averaged) is a real, control-verified
MiDrug gain: **every split-group mean is positive on both metrics**, LncDrug stays
inside the 0.001 tolerance, and the label-permuted twin is negative at every weight.
Per split at `w=0.04`:

| MiDrug split | ΔAUC | ΔAP |
|---|---:|---:|
| inner (3 seeds) | +0.00016 / +0.00018 / +0.00017 | +0.00036 / +0.00045 / +0.00037 |
| s42 | +0.00002 | +0.00039 |
| s2026 | +0.00011 | +0.00018 |
| s31415 | **−0.00002** | +0.00002 |
| x7 | +0.00019 | +0.00020 |
| x77 | +0.00016 | +0.00040 |
| x777 | **−0.00001** | **−0.00004** |

It fails only the pre-registered "**every** individual split positive" clause, on two
splits, by 1e-5–4e-5. The effect itself is +1e-4 — the same order as the per-split
noise (the reference's own 3-seed spread is ~5e-4), which is why the sign cannot be
made uniform. Combining `lrbag` with the other new members (`lrdro`, `lrdcn`,
`lrdrodcn`) does not fix the two cells.

## Why no algorithm-only technique can pass this gate

The accepted bank fixes the information available to every member. The frozen
reference is a strong approximator of the Bayes-optimal ranking under that information,
and the classical pool (`HGB/ET/MLP/LR`) is already inside ~1e-4 of it. An algorithm
therefore cannot add pair-level information; it can only reduce estimation variance.
Variance reduction is exactly what ensembling/anchor shrinkage already extracted, and
its ceiling is ~1e-4 — below the gate's per-split sign resolution. This is the same
conclusion as the 87-mechanism synthesis and the ALGO round A–F, now confirmed on five
additional families including two never tried here (group-DRO; a LambdaRank-weighted
objective) and a never-tried model class (feature-token Transformer).

For reference, the four previously **strict-passing** algorithm-level schemes
(`pv_bce`, `pv_rank`, `offset2`, `pan`) all operated on the *provenance-augmented* bank:
their effects are ~+0.002, ten times the algorithm-only effect and comfortably above the
split noise. The gain was carried by the new information, not by the algorithm.

## Consequence for the goal

"Three new algorithm-only techniques that pass the strict 7-split gate" is not
achievable on the accepted information: the required effect size exceeds what any
reparameterisation of the same features can produce. Delivering it requires either
(a) a relaxed verification (e.g. MiDrug positive in each split **group mean** with a
negative control, on which `lrbag` passes and is a genuine new technique), or
(b) new orthogonal information (the direction explicitly excluded this round), or
(c) a new evaluation surface (cold-drug / temporal) where the reference is not at
ceiling.

## Reproduction

```bash
PY=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
$PY improve/research/algorithm_only_20260918/algo1_fttransformer.py <task> <surface> <out>
$PY improve/research/algorithm_only_20260918/algo2_families.py    <task> <surface> <out>
$PY improve/research/algorithm_only_20260918/algo3_propensity.py  <task> <surface> <out>
$PY improve/research/algorithm_only_20260918/algo4_lrbag.py       <task> <surface> <out>
OMP_NUM_THREADS=1 $PY improve/research/provenance_next_20260917/evaluate_generic_v1.py \
  <cells> '{task}_{surface}.npz' member lrbag,lrbag_perm 0.02,0.04,0.05 /tmp/gate
```

Run each fit with `OMP_NUM_THREADS=1`.
