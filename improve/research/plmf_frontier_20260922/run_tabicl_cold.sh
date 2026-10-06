#!/bin/bash
V=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
export OMP_NUM_THREADS=8
cd "$(dirname "$0")"
run() { echo "### $*"; $V tabicl_cold.py "$@" 2>&1 | grep -E "^##|Error|Traceback|seed" ; }
for ds in MDR MDS; do
  run --dataset $ds --protocol plmf --families ctx --fit warm --device cuda:0 --tag tab_ctx_warm
  run --dataset $ds --protocol plmf --families ctx,knn --fit warm --device cuda:0 --tag tab_ctxknn_warm
  run --dataset $ds --protocol plmf --families ctx,diff --fit warm --device cuda:0 --tag tab_ctxdiff_warm
  for k in 1 3 5; do run --dataset $ds --protocol plmf --families ctx,diff --fit warm --kshot $k --device cuda:0 --tag tab_ctxdiff_warm_k$k; done
done
echo TABDONE
