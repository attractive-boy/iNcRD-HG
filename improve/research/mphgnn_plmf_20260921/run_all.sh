#!/bin/bash
# MPHGNN on PLMF-MDA: full grid.  Jobs are independent; 8 run concurrently across the two A40s.
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921
PY=/data1/licheng/incrd-hg/.venv-dgl/bin/python
EP=${EP:-50}
gen() {  # priority order: main -> gene ablations -> feature variants
  for cfg in "kmer real" "kmer none" "kmer shuffled" "expr real" "kmer_expr real"; do
    for ds in MDR MDS; do
      echo "$ds 5cv 42 $cfg"
      for s in 0 1 2 3 4; do echo "$ds single $s $cfg"; done
    done
  done
}
gen | nl -v0 | while read i ds split seed feat gene; do
  echo "$PY run_mphgnn_plmf.py --dataset $ds --split $split --seed $seed --feat $feat --gene $gene --epochs $EP --device cuda:$((i % 2)) > logs/mphgnn_${feat}_${gene}_${ds}_${split}_seed${seed}.log 2>&1"
done > logs/jobs.txt
echo "$(wc -l < logs/jobs.txt) jobs"
xargs -P 8 -I{} bash -c "{}" < logs/jobs.txt
echo ALL DONE
