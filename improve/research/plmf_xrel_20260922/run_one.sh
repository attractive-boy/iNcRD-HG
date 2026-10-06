#!/bin/bash
# usage: run_one.sh <dataset> <split> <seed> <variant>
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_xrel_20260922
$PY final2.py --dataset $1 --split $2 --seed $3 --variant $4 >> logs/grid_$1_$2_s$3.log 2>&1
