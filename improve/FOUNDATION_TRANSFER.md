# Frozen tabular-foundation-model transfer (2026-09-14)

This document describes a new, sealed experiment that adds a **second tabular
foundation model, TabDPT 1.3 (Layer6; checkpoint released September 8, 2026)**, to the accepted iNcRD-HG
predictor. It transfers the already accepted evidence/context feature banks onto
TabDPT and combines the result with the frozen TabICLv2 reference.

No pretrained or trained neural parameters are modified. The frozen TabICLv2
reference `0.5 * graph_context + 0.5 * tiered_evidence`
(`results/tabicl_pathway_context_20260913`) is left untouched.

## Motivation

Every accepted iNcRD-HG improvement so far uses the **same** TabICLv2 backbone
and only changes the feature/context adapter. Prior attempts to add a different
learned encoder (TabM, HGT, GNN, M2UMol, RNA-FM) either failed the joint gate or
were negative. A TabDPT 1.3 in-context classifier is a genuinely different
inductive bias:

- retrieval-normalised feature processing and a context-reduction step,
- a different pretraining prior-data distribution,
- a different member/permutation ensembling scheme.

Used zero-shot (`fit` on the support rows, `predict_proba` on the query rows) it
needs no gradient updates, exactly like TabICLv2 in this pipeline.

## Data and feature banks

The inputs are reconstructed from the accepted pathway-context encoders without
refitting anything (`foundation_transfer_features.py`):

| Bank | Columns | Contents |
|---|---:|---|
| `base36` | 36 | frozen graph pair features (first 36 columns of any bank) |
| `tiered62` | 62 | 36 graph columns + 26 tiered evidence columns (`tiered` variant) |
| `context76` | 76 | 36 graph columns + 26 tiered + 14 DepMap expression/context columns (`context` variant) |

`MiDrug` uses the pre-registered `raw` input view and `LncDrug` the pre-registered
`quantile` view, matching the accepted reference. The frozen inputs are hash-sealed
in `results/foundation_transfer_20260914/inputs/INPUTS.json`.

## Runner

`foundation_transfer_runner.py` verifies the TabDPT checkpoint
`tabdpt1_3.safetensors` by SHA-256
(`97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb`) and writes
one score file per (bank, task, inference seed 666/0/1), using
`context_reduction="subsample"` and `n_ensembles=8`. It runs in the dedicated
frontier environment; it does not import the `improve` runtime.

## Score recipes (pre-registered)

All recipes are fixed equal-logit combinations. No coefficient is searched after
metrics are seen.

| Recipe | Formula |
|---|---|
| `reference` | accepted `0.5*graph + 0.5*tiered` |
| `tdp_base` | TabDPT on `base36` |
| `tdp_tiered62` | TabDPT on `tiered62` |
| `tdp_context76` | TabDPT on `context76` |
| `ensemble_ref_base36` | `sigmoid(0.5*logit(ref) + 0.5*logit(tdp_base))` |
| `ensemble_ref_tiered62` | `sigmoid(0.5*logit(ref) + 0.5*logit(tdp_tiered62))` |
| `ensemble_ref_context76` | `sigmoid(0.5*logit(ref) + 0.5*logit(tdp_context76))` |
| `ensemble_ref_tiered62_context76` | equal-logit mean of `ref`, `tdp_tiered62`, `tdp_context76` |

## Acceptance gate

Identical to the project's common rule:

- **LncDrug**: AUC and AP must be `>=` the accepted reference in **every**
  inference seed, zero negative tolerance.
- **MiDrug**: mean AUC and AP must both increase and each must be positive in at
  least 2/3 seeds.
- Macro drug AP is reported separately and is not a hard gate.

## Reproduce

```bash
cd /data1/licheng/incrd-hg/iNcRD-HG
TABICL=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
FRONTIER=/data1/licheng/incrd-hg/.venv-frontier/bin/python

$TABICL -m improve.foundation_transfer_features
$FRONTIER improve/foundation_transfer_runner.py --task MiDrug  --device cuda:0
$FRONTIER improve/foundation_transfer_runner.py --task LncDrug --device cuda:1
$TABICL -m improve.foundation_transfer_eval prepare
$TABICL -m improve.foundation_transfer_eval run
$TABICL -m unittest improve.test_foundation_transfer -v
```

Sealed outputs: `results/foundation_transfer_20260914/{inputs,tabdpt,eval}`.
The evaluation protocol hash is recorded in `eval/protocol.json`; the source
snapshot is in `eval/source_snapshot/`.

## Boundaries

- The historical inner partition has been repeatedly used in development.
- The three inference seeds are randomness, not independent data splits.
- No historical outer, original test or source-holdout array is evaluated.
- TabDPT/DEPMap-derived columns are model/annotation priors, not measured
  ncRNA regulatory function. `0` labels remain sampled-unobserved pairs.
