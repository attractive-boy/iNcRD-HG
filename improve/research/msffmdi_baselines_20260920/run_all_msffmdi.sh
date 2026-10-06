#!/bin/bash
# MSFFMDI baselines: 5 methods x {ncDR, RNAInter} x 5 folds, seed 42. Two GPU workers.
set -u
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd "$(dirname "$0")"
mkdir -p logs results

# worker A (GPU0): lagcn, mfidma, ldaman
(
  export CUDA_VISIBLE_DEVICES=0 MSFFM_DEV=cuda:0 PYTHONUNBUFFERED=1
  for ds in ncDR RNAInter; do
    for m in lagcn mfidma ldaman; do
      echo "### $m $ds start $(date)"
      $PY run_msffmdi.py --model $m --dataset $ds --seed 42
      echo "### $m $ds done $(date)"
    done
  done
) > logs/workerA.stdout 2>&1 &
PID_A=$!

# worker B (GPU1): gcnnma, dlstmda
(
  export CUDA_VISIBLE_DEVICES=1 MSFFM_DEV=cuda:0 PYTHONUNBUFFERED=1
  for ds in ncDR RNAInter; do
    for m in gcnnma dlstmda; do
      echo "### $m $ds start $(date)"
      $PY run_msffmdi.py --model $m --dataset $ds --seed 42
      echo "### $m $ds done $(date)"
    done
  done
) > logs/workerB.stdout 2>&1 &
PID_B=$!

echo "workerA pid=$PID_A workerB pid=$PID_B"
wait $PID_A $PID_B
echo "ALL DONE $(date)"
