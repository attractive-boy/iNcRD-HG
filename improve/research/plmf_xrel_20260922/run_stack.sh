#!/bin/bash
# usage: run_stack.sh <dataset> <split> <seed> <mult> <xrel-groups or ->
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_xrel_20260922
X=""; if [ "$5" != "-" ]; then X="--xrel $5"; fi
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python probe_extraneg.py --dataset $1 --split $2 --seed $3 --mult $4 --prov full $X > logs/stack_$1_$2_s$3_x$4_$5.log 2>&1
