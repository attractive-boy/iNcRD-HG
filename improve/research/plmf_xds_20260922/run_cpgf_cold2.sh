#!/bin/bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
P=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
$P polyfilt.py --dataset MDR --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --fit drugcv --stack ctx,knn
$P polyfilt.py --dataset MDR --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --fit both --stack ctx,knn
$P polyfilt.py --dataset MDS --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --fit drugcv --stack ctx,knn
$P polyfilt.py --dataset MDS --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --fit both --stack ctx,knn
$P polyfilt.py --dataset MDR --protocol cold --K 6 --epochs 600 --seeds 0,1,2,3,4 --fit drugcv --shuffle_content
$P polyfilt.py --dataset MDR --protocol warm --K 6 --epochs 600 --fit both
