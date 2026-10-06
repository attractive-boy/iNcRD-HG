#!/bin/bash
# Balanced-setting reproduction (equal negatives sampled to match positives).
# usage: bash run_balanced.sh <gpu> <dataset>
set -u
PY=/data1/licheng/incrd-hg/.venv-dgl/bin/python
cd "$(dirname "$0")"
GPU=${1:-0}
DS=${2:-1}
for v in deeplda gcn gat; do
  echo "===== balanced dataset $DS variant $v ====="
  CUDA_VISIBLE_DEVICES=$GPU $PY repro_deeplda_official.py \
    --dataset "$DS" --variant "$v" --seeds 72 666 0 1 2 --epochs 1050 --balanced \
    --out "results_bal_${v}_ds${DS}.csv"
done
echo "DONE balanced ds$DS"
