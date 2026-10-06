#!/bin/bash
# Graph-model ratio runs (GraphDTA, SubMDTA) for 1:5 and 1:10, on cuda:1.
set -u
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd "$(dirname "$0")"
export DLST_DEV=cuda:1 PYTHONUNBUFFERED=1
for r in 5 10; do
  export DLST_PROC="ratios/1to$r" DLST_TAG="_1to$r"
  echo "### graphdta 1to$r start $(date)"
  $PY run_bench.py --model graphdta --epochs 40 --seed 42 --out results/ratio_1to${r}_graphdta.csv
  echo "### submdta 1to$r start $(date)"
  $PY run_bench.py --model submdta --epochs 40 --seed 42 --out results/ratio_1to${r}_submdta.csv
done
echo "### graph ratios done $(date)"
