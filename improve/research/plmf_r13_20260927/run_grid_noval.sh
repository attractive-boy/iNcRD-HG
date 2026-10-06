#!/bin/bash
# official PLMF-MDA protocol (100% training pairs, no hold-out).  usage: ./run_grid_noval.sh grid.txt [parallel] DS:split:seed ...
GRID=$1; P=${2:-6}; shift 2; SPECS="$@"
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python; export OMP_NUM_THREADS=4
cd "$(dirname "$0")"
source cfg.sh
i=0
grep -v '^#' "$GRID" | grep -v '^\s*$' | while IFS='|' read -r tag args; do
  args=$(eval echo "$args")
  for spec in $SPECS; do
    IFS=':' read -r ds sp sd <<< "$spec"
    gpu=$((i % 2)); i=$((i+1))
    out=results/${tag}_${ds}_${sp}_seed${sd}.csv
    [ -f "$out" ] && continue
    echo "$PY run_r13.py --dataset $ds --split $sp --seed $sd --tag $tag $args --device cuda:$gpu > logs/${tag}_${ds}_${sp}_s${sd}.log 2>&1"
  done
done | xargs -P "$P" -I{} bash -c "{}"
