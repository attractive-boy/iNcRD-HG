#!/bin/bash
export OMP_NUM_THREADS=6; PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
for s in 42 0 1 2; do for ds in MDR MDS; do for v in base prov mprior final final_shuf pmid_only; do
  $PY final.py --dataset $ds --split 5cv --seed $s --inner 10 --variant $v
done; done; done
