#!/bin/bash
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd "$(dirname "$0")"
while ! grep -q ALLDONE2 logs/cold_batch2.log; do sleep 20; done
run() { echo "### $*" ; $PY cold_probe.py "$@" 2>&1 | grep -E "^##|Error|Traceback" ; }
for ds in MDR MDS; do
  run --dataset $ds --protocol plmf --families ctx --scorer kron:1.0 --tag kron1
  run --dataset $ds --protocol plmf --families ctx --scorer kron:0.3 --tag kron03
  run --dataset $ds --protocol plmf --families ctx --scorer kron:3.0 --tag kron3
  # miRNA-cold (PLMF Table 2: miRNAs with 5-6 drugs)
  run --dataset $ds --protocol plmf_mirna --families ctx --fit warm --tag mcold_ctx_warm
  run --dataset $ds --protocol plmf_mirna --families ctx,diff --fit warm --tag mcold_ctxdiff_warm
  run --dataset $ds --protocol plmf_mirna --families ctx --score_col degree_d --tag mcold_score_degree_d
  # k-shot curve for cold drugs (ctx+diff warm fit)
  for k in 1 3 5 10; do
    run --dataset $ds --protocol plmf --families ctx,diff --fit warm --kshot $k --tag ctxdiff_warm_k$k
    run --dataset $ds --protocol plmf --families ctx,diff,knn --fit warm --kshot $k --tag ctxdiffknn_warm_k$k
  done
done
echo ALLDONE3
