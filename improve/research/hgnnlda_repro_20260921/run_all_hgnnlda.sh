#!/usr/bin/env bash
# Reproduce HGNNLDA (TCBB 2023) on RNAactDrug: 5-fold CV + independent set.
set -e
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DEV="${DEV:-cuda:0}"
for S in 1 2 3; do
  $PY run_hgnnlda.py --seed "$S" --device "$DEV"
done
$PY run_hgnnlda.py --seed 1 --device "$DEV" --propagate iter --out results/hgnnlda_iter_seed1.csv
$PY run_hgnnlda_independent.py --seed 1 --device "$DEV"
$PY summarize_hgnnlda.py
