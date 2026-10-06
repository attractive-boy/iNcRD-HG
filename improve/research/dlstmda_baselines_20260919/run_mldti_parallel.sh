#!/bin/bash
set -u
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd "$(dirname "$0")"
for f in 0 1 2 3 4; do
  dev=$(( f % 2 ))
  DLST_DEV=cuda:$dev PYTHONUNBUFFERED=1 $PY run_bench.py --model mldti --seed 42 \
      --epochs 150 --folds $f --out results/mldti_fold$f.csv > logs/mldti_f$f.stdout 2>&1 &
done
wait
echo "MLDTI PARALLEL DONE"
