#!/bin/bash
cd "$(dirname "$0")"
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
export OMP_NUM_THREADS=8
for ds in MDR MDS; do
  $PY xds_probe.py --dataset $ds --families ctx,diff,xds
  $PY xds_probe.py --dataset $ds --families ctx,diff,xds --shuffle_other
  $PY xds_probe.py --dataset $ds --families ctx,diff,xdsp
  $PY xds_probe.py --dataset $ds --families ctx,diff,xdsd
  $PY xds_probe.py --dataset $ds --families ctx,diff,xdsx
  $PY xds_probe.py --dataset $ds --families xds --model lr:0.3
done
