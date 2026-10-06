#!/bin/bash
# Run all 5 DLST-MDA baseline reproductions (positive:negative 1:1, author 5 folds).
# Two GPU workers to use both idle cards; logs land in logs/*.stdout.
set -u
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd "$(dirname "$0")"

# worker A -> cuda:0 : graphdta, submdta, mldti
(
  export DLST_DEV=cuda:0 PYTHONUNBUFFERED=1
  for m in graphdta submdta mldti; do
    echo "### $m start $(date)"
    $PY run_bench.py --model $m --seed 42 --epochs 150
    echo "### $m done $(date)"
  done
) > logs/workerA.stdout 2>&1 &
PID_A=$!

# worker B -> cuda:1 : gcnnma, daestb
(
  export DLST_DEV=cuda:1 PYTHONUNBUFFERED=1
  echo "### gcnnma start $(date)"
  $PY gcnnma.py --seed 42 --epochs 100
  echo "### gcnnma done $(date)"
  echo "### daestb start $(date)"
  $PY daestb.py --seed 42 --sim gip
  echo "### daestb done $(date)"
) > logs/workerB.stdout 2>&1 &
PID_B=$!

echo "workerA pid=$PID_A  workerB pid=$PID_B"
wait $PID_A $PID_B
echo "ALL DONE $(date)"
