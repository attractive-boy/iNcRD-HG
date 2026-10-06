#!/bin/bash
# Reproduce official DeepLDA + GCN-only + GAT-only on Dataset1/2.
# usage: bash run_all_official.sh <gpu> <dataset>
set -u
PY=/data1/licheng/incrd-hg/.venv-dgl/bin/python
cd "$(dirname "$0")"
GPU=${1:-0}
DS=${2:-1}
for v in deeplda gcn gat; do
  echo "===== dataset $DS variant $v ====="
  CUDA_VISIBLE_DEVICES=$GPU $PY repro_deeplda_official.py \
    --dataset "$DS" --variant "$v" --seeds 72 666 0 1 2 --epochs 1050 \
    --out "results_${v}_ds${DS}.csv"
done
echo "DONE ds$DS"
