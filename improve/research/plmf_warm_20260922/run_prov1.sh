#!/bin/bash
export OMP_NUM_THREADS=6; PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
for ds in MDR MDS; do
  $PY prov.py --dataset $ds
  $PY prov.py --dataset $ds --shuffle
  $PY prov.py --dataset $ds --only
  $PY prov.py --dataset $ds --mprior fam,seed,hmdd,xdeg
done
