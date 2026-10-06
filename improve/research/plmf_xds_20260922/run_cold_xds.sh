#!/bin/bash
cd "$(dirname "$0")"
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
export OMP_NUM_THREADS=8
for ds in MDR MDS; do
  $PY cold_xds.py --dataset $ds --score_col x_pair --seeds 0
  $PY cold_xds.py --dataset $ds --score_col x_ppr --seeds 0
  $PY cold_xds.py --dataset $ds --xds none
  $PY cold_xds.py --dataset $ds --xds full
  $PY cold_xds.py --dataset $ds --xds full --shuffle_other
  $PY cold_xds.py --dataset $ds --xds pair
  $PY cold_xds.py --dataset $ds --xds deg
done
