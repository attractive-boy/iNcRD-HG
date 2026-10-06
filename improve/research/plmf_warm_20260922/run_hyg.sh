#!/bin/bash
export OMP_NUM_THREADS=6; PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
for ds in MDR MDS; do
  $PY hygiene.py --dataset $ds --K 10
  $PY hygiene.py --dataset $ds --K 20
  $PY hygiene.py --dataset $ds --K 5 --R 3
done
