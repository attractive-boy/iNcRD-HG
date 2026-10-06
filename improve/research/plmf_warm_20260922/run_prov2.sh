#!/bin/bash
export OMP_NUM_THREADS=6; PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
for ds in MDR MDS; do
  $PY prov.py --dataset $ds --groups pmid,costudy,ctx,rec
  $PY prov.py --dataset $ds --groups pmid,costudy,ctx,rec --shuffle
  $PY prov.py --dataset $ds --groups pmid,costudy
  $PY prov.py --dataset $ds --groups pmid,ctx
  $PY prov.py --dataset $ds --groups pmid,rec
  $PY prov.py --dataset $ds --groups pmid,costudy,ctx,rec --mprior fam,seed,hmdd,xdeg
done
