#!/bin/bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_xrel_20260922
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python study_split2.py --dataset $1 --seed $2 --variant $3 --records full >> logs/studyF_$1_s$2.log 2>&1
