#!/bin/bash
export OMP_NUM_THREADS=8; PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
for ds in MDR MDS; do
  $PY mprior.py --dataset $ds --fams ''                       --tag base
  $PY mprior.py --dataset $ds --fams fam
  $PY mprior.py --dataset $ds --fams seed
  $PY mprior.py --dataset $ds --fams hmdd
  $PY mprior.py --dataset $ds --fams xdeg
  $PY mprior.py --dataset $ds --fams fam,seed,hmdd,xdeg
  $PY mprior.py --dataset $ds --fams fam,seed,hmdd
  $PY mprior.py --dataset $ds --fams fam,seed,hmdd,xdeg --shuffle fam,seed,hmdd,xdeg
  $PY mprior.py --dataset $ds --fams hmdd --shuffle hmdd
  $PY mprior.py --dataset $ds --fams fam --shuffle fam
  $PY mprior.py --dataset $ds --fams fam,seed,hmdd,xdeg --only
done
