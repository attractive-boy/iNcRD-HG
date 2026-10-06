#!/bin/bash
export OMP_NUM_THREADS=6; PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
for s in 1 2; do for ds in MDR MDS; do
  $PY mprior.py --dataset $ds --seed $s --fams '' --tag base
  $PY mprior.py --dataset $ds --seed $s --fams fam,seed,hmdd,xdeg
  $PY mprior.py --dataset $ds --seed $s --fams fam,seed,hmdd,xdeg --shuffle fam,seed,hmdd,xdeg
  $PY prov.py --dataset $ds --seed $s --groups pmid,costudy,ctx,rec --mprior fam,seed,hmdd,xdeg
done; done
