#!/bin/bash
# PLMF-MDA reference on the baseline splits (fast cached-encoder runner).
# GPU0 -> MDR, GPU1 -> MDS; each does 5cv seed42 then single seeds 0..4.
set -u
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd "$(dirname "$0")"
mkdir -p logs results

run_ds() {
  local gpu=$1 ds=$2
  export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1
  echo "### $ds 5cv start $(date)"
  $PY plmf_ref_5cv.py --dataset $ds --split 5cv --seed 42 --device cuda:0
  echo "### $ds 5cv done $(date)"
  for s in 0 1 2 3 4; do
    echo "### $ds single seed=$s start $(date)"
    $PY plmf_ref_5cv.py --dataset $ds --split single --seed $s --folds 0 --device cuda:0
    echo "### $ds single seed=$s done $(date)"
  done
}

( run_ds 0 MDR ) > logs/plmf_ref_MDR.stdout 2>&1 &
( run_ds 1 MDS ) > logs/plmf_ref_MDS.stdout 2>&1 &
wait
echo "PLMF REF ALL DONE $(date)"
