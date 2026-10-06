#!/bin/bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$(dirname "$0")"
run() { echo "### $*" ; $PY cold_probe.py "$@" 2>&1 | grep -E "^##|Error|Traceback" ; }
for ds in MDR MDS; do
  run --dataset $ds --protocol plmf --families ctx --fit lodo --tag ctx_lodo
  run --dataset $ds --protocol plmf --families ctx,diff --fit lodo --tag ctxdiff_lodo
  run --dataset $ds --protocol plmf --families ctx,knn --fit lodo --tag ctxknn_lodo
  run --dataset $ds --protocol plmf --families ctx,diff,knn --fit lodo --tag ctxdiffknn_lodo
  run --dataset $ds --protocol plmf --families ctx,knn --fit warm --tag ctxknn_warm
  run --dataset $ds --protocol plmf --families ctx,diff --scorer transfer --tag transfer_ctxdiff
  run --dataset $ds --protocol plmf --families ctx,diff --scorer transfer:10 --tag transfer10_ctxdiff
  run --dataset $ds --protocol plmf --families ctx --scorer transfer --tag transfer_ctx
  run --dataset $ds --protocol plmf --families ctx,knn --scorer rankavg:degree_r+knn_tan0_mean --tag rankavg_degr_knn
  run --dataset $ds --protocol plmf --families ctx,knn --scorer rankavg:degree_r+knn_tan0_mean+drug_attribute_support_sum --tag rankavg_degr_knn_attr
  run --dataset $ds --protocol plmf --families ctx,knn --scorer rankavg:degree_r+knn_tan0_mean+knn_cb0_mean --tag rankavg_degr_tan_cb
done
echo ALLDONE2
