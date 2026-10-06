#!/bin/bash
# Leakage-free model selection for the main variant: wait for the grid, then run kmer/real with a 10% validation hold-out.
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921
while pgrep -f "run_all.sh" > /dev/null; do sleep 60; done
PY=/data1/licheng/incrd-hg/.venv-dgl/bin/python
{ for ds in MDR MDS; do echo "$ds 5cv 42"; for s in 0 1 2 3 4; do echo "$ds single $s"; done; done; } | nl -v0 | while read i ds split seed; do
  echo "$PY run_mphgnn_plmf_val.py --dataset $ds --split $split --seed $seed --feat kmer --gene real --val_frac 0.1 --epochs 50 --device cuda:$((i % 2)) > logs/mphgnn_kmer_real_val_${ds}_${split}_seed${seed}.log 2>&1"
done > logs/jobs_val.txt
xargs -P 8 -I{} bash -c "{}" < logs/jobs_val.txt
echo VAL DONE
