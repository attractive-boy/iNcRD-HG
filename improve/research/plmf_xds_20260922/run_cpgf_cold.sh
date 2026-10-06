#!/bin/bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
P=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
$P polyfilt.py --dataset MDR --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --stack ctx,knn
$P polyfilt.py --dataset MDS --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --stack ctx,knn
$P polyfilt.py --dataset MDR --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --no_content
$P polyfilt.py --dataset MDR --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --shuffle_content
$P polyfilt.py --dataset MDR --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --fix_p 0.5
$P polyfilt.py --dataset MDS --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --no_content
$P polyfilt.py --dataset MDR --protocol cold_mirna --K 6 --epochs 600 --seeds 0,1,2
