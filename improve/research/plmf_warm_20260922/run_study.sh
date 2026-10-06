#!/bin/bash
export OMP_NUM_THREADS=6; PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
for s in 42 0; do for ds in MDR MDS; do for v in base prov mprior final final_shuf; do
  $PY study_split.py --dataset $ds --seed $s --variant $v
done; done; done
