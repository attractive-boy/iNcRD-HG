#!/bin/bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$(dirname "$0")"
run() { echo "### $*" ; $PY cold_probe.py "$@" 2>&1 | grep -E "^##|Error|Traceback" ; }
for ds in MDR MDS; do
  # PLMF Table-2 protocol, 5 seeds
  run --dataset $ds --protocol plmf --families ctx --fit warm   --tag ctx_warm
  run --dataset $ds --protocol plmf --families ctx --fit drugcv --tag ctx_drugcv
  run --dataset $ds --protocol plmf --families ctx,diff --fit drugcv --tag ctxdiff_drugcv
  run --dataset $ds --protocol plmf --families ctx,diff --fit warm --tag ctxdiff_warm
  run --dataset $ds --protocol plmf --families knn --fit drugcv --tag knn_drugcv
  run --dataset $ds --protocol plmf --families mech --fit drugcv --tag mech_drugcv
  run --dataset $ds --protocol plmf --families ctx,diff,knn --fit drugcv --tag ctxdiffknn_drugcv
  run --dataset $ds --protocol plmf --families ctx,diff,knn,mech --fit drugcv --tag ctxdiffknnmech_drugcv
  run --dataset $ds --protocol plmf --families ctx,diff,cdiff,align,knn --fit drugcv --tag ctxdiffcdiffalignknn_drugcv
  run --dataset $ds --protocol plmf --families ctx,diff,knn,mech --fit drugcv --model hgb --tag ctxdiffknnmech_drugcv_hgb
  # single-column scorers (no training)
  run --dataset $ds --protocol plmf --families ctx --score_col degree_r --tag score_degree_r
  run --dataset $ds --protocol plmf --families knn --score_col knn_tan0_mean --tag score_knn_tan0_mean
  run --dataset $ds --protocol plmf --families knn --score_col knn_tan5_mean --tag score_knn_tan5_mean
  run --dataset $ds --protocol plmf --families knn --score_col knn_cb0_mean --tag score_knn_cb0_mean
  run --dataset $ds --protocol plmf --families knn --score_col knn_tan0_rpc --tag score_knn_tan0_rpc
  run --dataset $ds --protocol plmf --families mech --score_col mech_b0_norm --tag score_mech_b0_norm
  run --dataset $ds --protocol plmf --families ctx --score_col collab_d --tag score_collab_d
  run --dataset $ds --protocol plmf --families ctx --score_col drug_attribute_support_sum --tag score_drug_attr_support
done
for ds in MDR MDS; do
  # leave-drugs-out over all drugs, 3 seeds
  run --dataset $ds --protocol ldo --seeds 0,1,2 --families ctx --fit warm   --tag ctx_warm
  run --dataset $ds --protocol ldo --seeds 0,1,2 --families ctx --fit drugcv --tag ctx_drugcv
  run --dataset $ds --protocol ldo --seeds 0,1,2 --families ctx,diff --fit drugcv --tag ctxdiff_drugcv
  run --dataset $ds --protocol ldo --seeds 0,1,2 --families ctx,diff,knn --fit drugcv --tag ctxdiffknn_drugcv
  run --dataset $ds --protocol ldo --seeds 0,1,2 --families ctx,diff,knn,mech --fit drugcv --tag ctxdiffknnmech_drugcv
  run --dataset $ds --protocol ldo --seeds 0,1,2 --families ctx --score_col degree_r --tag score_degree_r
  run --dataset $ds --protocol ldo --seeds 0,1,2 --families knn --score_col knn_tan0_mean --tag score_knn_tan0_mean
done
echo ALLDONE
