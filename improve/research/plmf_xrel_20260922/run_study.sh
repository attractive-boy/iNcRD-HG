#!/bin/bash
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_xrel_20260922
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python study_split2.py --dataset $1 --seed $2 --variant $3 >> logs/study_$1_s$2.log 2>&1
