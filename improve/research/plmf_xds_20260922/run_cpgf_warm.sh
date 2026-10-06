#!/bin/bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
P=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
$P polyfilt.py --dataset MDR --protocol warm --K 8 --epochs 800 --stack ctx,diff
$P polyfilt.py --dataset MDS --protocol warm --K 8 --epochs 800 --stack ctx,diff
$P polyfilt.py --dataset MDR --protocol warm --K 8 --epochs 800 --no_content
$P polyfilt.py --dataset MDR --protocol warm --K 8 --epochs 800 --fix_p 0.5
