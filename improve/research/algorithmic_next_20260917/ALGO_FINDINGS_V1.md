# Algorithm-level optimisation schemes for iNcRD-HG: results

Date 2026-09-17. Scope: changes to the objective, the model structure or the fusion
rule — no new feature columns as the *primary* change, no hyper-parameter tuning, no
default model replacement. Same harness as the provenance round: frozen TabICLv2
`tiered_0p5000` reference, accepted shared-anchor recipe, seven splits (inner x3 seeds,
resplits 42/2026/31415, extra 7/77/777), label-preserving permutation control per scheme.
Files under `improve/research/algorithmic_next_20260917/`.

## The three schemes and what verification shows

| # | scheme | level | control | MiDrug inner ΔAUC/ΔAP | gate | verdict |
|---|---|---|---|---|---:|---|
| A | cross-drug pairwise ranking loss `softplus(-(s_pos - s_neg))`, positives vs cross-drug negatives | objective | BCE on the identical net; permuted-label net | +0.00034 / +0.00172 | fail (Mi resplit −0.0002) | objective alone does **not** help; BCE net is equal (+0.00018/+0.00181) |
| B | two-part model `score = beta*log1p(deg_r)+beta*log1p(deg_d) + MLP(x)`, cross-fitted offset | structure | permuted offset; no offset | +0.00014 / +0.00149 | fail (Mi resplit/extra negative) | not distinguishable from its control |
| C | cross-fitted stacking: 4 accepted members → 5-fold OOF → pairwise-ranking meta-net | ensemble | uniform mean of the same OOF members; permuted-label meta | +0.00012 / +0.00020 | fail (LncDrug extra AP −0.0048) | meta-learner ≈ uniform mean on MiDrug, worse on LncDrug |
| D | rank/logit aggregation operators (median, trimmed, prob-mean, Borda rank-mean) | fusion | arithmetic-mean logits (accepted) | +0.00003…+0.00010 | tolerance only | rank-mean is robust but the gain is ~1e-4, negligible |
| F | label-free pair-adaptive shrinkage `w(pair) = w0·(1−sigmoid((\|lr−lm\|−d0)/T))` | fusion | fixed w | ≤ +0.0002 | fail every (w0,d0,T) | disagreement gate loses the LncDrug gain |

**Conclusion of the three schemes: none robustly improves global MiDrug AUC/AP while
preserving LncDrug when it acts on the existing information alone.** The 87-mechanism
saturation result (mechanism margin mean ≈ 0) extends from graph operators to the
objective and the model structure. The metric is sampling-propensity dominated and the
reference is at its ceiling for this feature set, so a pure algorithm change can only
re-shuffle signal already present.

## The one configuration that does pass — and it is algorithm + information

The cross-drug pairwise objective becomes an improvement when it is applied to the
**provenance-augmented bank** `[context76 | 14 provenance columns]`:

| recipe | objective | MiDrug inner | Mi resplit | Mi extra | LncDrug inner | gate |
|---|---|---:|---:|---:|---:|---|
| `pv_rank` | cross-drug pairwise | +0.00189 / +0.00595 | +0.00231 / +0.00617 | +0.00255 / +0.00555 | +0.00037 / +0.00101 | **strict pass at w=0.1** |
| `pv_bce` | BCE, same net | +0.00180 / +0.00561 | +0.00220 / +0.00594 | +0.00250 / +0.00559 | +0.00056 / +0.00145 | strict pass at w=0.1 |
| `pv_rank_perm` | pairwise, permuted labels | −0.00096 / −0.00161 | −0.00056 / −0.00131 | −0.00068 / −0.00118 | −0.00216 / −0.01431 | fail |

Two readings:
1. The provenance information is what makes the configuration pass; the pairwise
   objective adds a small MiDrug margin over BCE (+0.0001 AUC / +0.0003 AP inner) and
   costs a little LncDrug.
2. A single neural MLP on `[bank | provenance]` is a **stronger member** than the
   accepted classical pool (which gave +0.00088/+0.00237 at w=0.1): the model class,
   not the loss, is where the extra margin comes from.

## Recommendation

* Use `pv_bce`/`pv_rank` as the operational configuration (same frozen reference, same
  no-test-source rule): +0.0018 to +0.0019 AUC / +0.0056 to +0.0060 AP on MiDrug inner,
  LncDrug positive on every split, permutation control negative.
* Do not spend further effort on objective/structure/fusion changes over the existing
  features alone: schemes A, B, C, D, F all sit inside the split noise.

## Artifacts

* `algo_pairwise_v1.py`, `cells_algoA` — scheme A (+BCE and permuted controls)
* `algo_offset_v1.py`, `cells_algoB` — scheme B
* `algo_stack_v1.py`, `cells_algoC` — scheme C
* `algo_aggregate_v1.py`, `cells_algoD` — scheme D
* `algo_pairwise_prov_v1.py`, `cells_algoE` — the passing configuration
* `algo_gate_v1.py` — scheme F
* `algo_common.py` — shared loader/saver; gate tables via `evaluate_generic_v1.py`
  (`/tmp/opencode/eval_algo{A,B,C,D,E}_gates.csv`)
