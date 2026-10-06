#!/bin/bash
export OMP_NUM_THREADS=6; PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
for s in 0 1 2 3 4; do for ds in MDR MDS; do for v in base final final_shuf prov mprior; do
  $PY final.py --dataset $ds --split single --seed $s --inner 10 --variant $v
done; done; done
