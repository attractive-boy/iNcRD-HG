#!/bin/bash
# Additional strong-miRNA baselines on PLMF-MDA (MDR/MDS), paper 5-fold, seed 42.
#   DLST-MDA (JCIM 2025)  official GCNNetmuti
#   DLMVF    (JCIM 2026)  official model
#   SubKNet  (JBHI 2024)  official subgraph + RW graph-kernel
# Usage: ./run_all.sh 5cv
set -u
B=/data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_strong_baselines_20260922
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
export PYTHONUNBUFFERED=1
SPLIT="${1:-5cv}"
cd "$B"

# GPU0
(
  $PY run_dlstmda_plmf.py --dataset MDR --split $SPLIT --seed 42 --epochs 100 --device cuda:0
  $PY run_dlmvf_plmf.py   --dataset MDR --split $SPLIT --seed 42 --epochs 60  --device cuda:0
  $PY run_subknet_plmf.py --dataset MDR --split $SPLIT --seed 42 --epochs 30  --device cuda:0
) > logs/worker_gpu0.log 2>&1 &
P0=$!

# GPU1
(
  $PY run_dlstmda_plmf.py --dataset MDS --split $SPLIT --seed 42 --epochs 100 --device cuda:1
  $PY run_dlmvf_plmf.py   --dataset MDS --split $SPLIT --seed 42 --epochs 60  --device cuda:1
  $PY run_subknet_plmf.py --dataset MDS --split $SPLIT --seed 42 --epochs 30  --device cuda:1
) > logs/worker_gpu1.log 2>&1 &
P1=$!

echo "gpu0 pid=$P0 gpu1 pid=$P1"
wait $P0 $P1
echo "ALL DONE $(date)"
