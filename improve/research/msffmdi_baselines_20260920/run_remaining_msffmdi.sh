#!/bin/bash
# MSFFMDI baselines: remaining methods, balanced across 2 GPUs.
#   GPU0: dlstmda (both datasets)
#   GPU1: gcnnma  (both datasets)
set -u
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd "$(dirname "$0")"
mkdir -p logs results

(
  export CUDA_VISIBLE_DEVICES=0 MSFFM_DEV=cuda:0 PYTHONUNBUFFERED=1
  for ds in ncDR RNAInter; do
    echo "### dlstmda $ds start $(date)"
    $PY run_msffmdi.py --model dlstmda --dataset $ds --seed 42
    echo "### dlstmda $ds done $(date)"
  done
) > logs/workerA2.stdout 2>&1 &
PID_A=$!

(
  export CUDA_VISIBLE_DEVICES=1 MSFFM_DEV=cuda:0 PYTHONUNBUFFERED=1
  for ds in ncDR RNAInter; do
    echo "### gcnnma $ds start $(date)"
    $PY run_msffmdi.py --model gcnnma --dataset $ds --seed 42
    echo "### gcnnma $ds done $(date)"
  done
) > logs/workerB2.stdout 2>&1 &
PID_B=$!

echo "A=$PID_A B=$PID_B"
wait $PID_A $PID_B
echo "ALL DONE $(date)"
