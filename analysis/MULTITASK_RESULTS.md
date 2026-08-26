# Multi-task iNcRD-HG — Results Summary

This documents the **multi-task extension** (`multitask_train.py`) that
addresses improvement direction #2 from the paper review: *"fine-grained
multi-task learning to avoid task interference"*.

## What changed vs. the released code

| Aspect | Released `predict_association.py` | `multitask_train.py` |
|---|---|---|
| Tasks | Two **separate** models in a for-loop | One **shared** model, both tasks jointly |
| Encoder | `to_hetero(GeneralConv)` per task | Shared `to_hetero(GeneralConv)`, one forward pass |
| Decoders | Per-task, independent | Per-task, independent (task-specific heads) |
| Cross-task bridge | None (tasks don't share params) | LncMi-edge consistency regularisation |
| Contrastive | None | Task-aware InfoNCE within ncRNA type |
| Loss weighting | Manual hyperparams | Uncertainty-weighted (Kendall et al., 2018) |
| Model selection | Test set every epoch (§2.3 bias) | Validation per-drug AUC early stopping |
| Cross-task leakage | Full other-task edges in graph | Training edges only from each task |

## Five design choices and their grounding

1. **Shared encoder + task-specific decoders** — the released code trains two
   completely independent models. The multi-task design shares representation
   learning while keeping task-specific scoring.
2. **Cross-task consistency via LncMi edges** — the graph already has
   lncRNA–miRNA interaction edges. Projected embeddings of interacting pairs
   are pushed into a shared drug-association subspace, bridging tasks through
   known biology rather than blind parameter sharing.
3. **Task-aware contrastive (InfoNCE)** — pulls positive ncRNA–drug pairs
   together and pushes random negative drugs apart, but only *within the same
   ncRNA type* so task-specific signals are preserved.
4. **Uncertainty-weighted loss** — each task has a learnable `log σ²`; the
   aggregate loss is `Σ [exp(−sᵢ)·Lᵢ + sᵢ]`, automatically down-weighting the
   noisier task.
5. **Validation-based model selection** — the released loop scores the test
   set every epoch and the headline matches the best epoch seen (§2.3 test-set
   selection bias). The multi-task script selects on validation per-drug AUC.

## Results (50 epochs, CPU, seed 666)

### Original splits (`3.heter_data`)

| Task | AUC | AUPR | perDrugAUC |
|---|---|---|---|
| LncDrug | 0.9835 | 0.9585 | 0.7763 |
| MiDrug  | 0.9430 | 0.9024 | 0.7584 |

Paper reported LncDrug AUC 0.9857 / AUPR 0.9585; MiDrug AUC 0.9564 / AUPR 0.9297.
The multi-task model matches LncDrug AUPR exactly and approaches the AUC with a
shared encoder and fewer epochs.

### Popularity-debiased benchmark (per-drug AUC — the honest metric)

| Split | Task | Single-task (FINDINGS.md) | Multi-task v2 | Δ |
|---|---|---|---|---|
| matched | LncDrug | 0.648 | **0.752** | +0.104 |
| matched | MiDrug  | 0.797 | 0.732 | −0.065 |
| cold_drug | LncDrug | 0.627 | **0.814** | +0.187 |
| cold_drug | MiDrug  | 0.820 | 0.779 | −0.041 |

**Key signal:** the multi-task design consistently improves LncDrug perDrugAUC
on the debiased splits (especially +0.187 on cold_drug), at a small cost to
MiDrug. This is biologically expected: lncRNA (fewer training edges: 2109 vs
5394) benefits more from the shared representation and cross-task drug context
provided by MiDrug edges, while miRNA loses some task-specific capacity.

The uncertainty parameters converge to `s_lnc > s_mi`, confirming that the
adaptive reweighting correctly identifies LncDrug as the noisier task.

> **Caveat:** single-task numbers are 8-seed averages from FINDINGS.md;
> multi-task numbers are single-seed. A rigorous comparison needs multi-seed
> runs on GPU. The trend (LncDrug gains, MiDrug trades off, mean improves on
> cold_drug) is consistent across the splits tested.

## Running

```bash
# Original splits
python multitask_train.py --data-dir ./3.heter_data --epochs 200 --device cuda

# Popularity-controlled benchmark
python multitask_train.py --data-dir ./9.bench/matched --epochs 200 --device cuda
python multitask_train.py --data-dir ./9.bench/cold_drug --epochs 200 --device cuda

# Tunable regularisation weights
python multitask_train.py --data-dir ./3.heter_data \
    --w-contrast 0.1 --w-consist 0.1 --hidden 128 --aggr mean
```
