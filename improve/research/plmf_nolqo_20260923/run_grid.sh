#!/bin/bash
# usage: ./run_grid.sh grid.txt [parallel] DS:split:seed ...   ; grid line: "tag|extra args"
GRID=$1; P=${2:-6}; shift 2; SPECS="$@"
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python; export OMP_NUM_THREADS=4
cd "$(dirname "$0")"
i=0
grep -v '^#' "$GRID" | grep -v '^\s*$' | while IFS='|' read -r tag args; do
  for spec in $SPECS; do
    IFS=':' read -r ds sp sd <<< "$spec"
    gpu=$((i % 2)); i=$((i+1))
    out=results/${tag}_${ds}_${sp}_seed${sd}.csv
    [ -f "$out" ] && continue
    echo "$PY run_nl.py --dataset $ds --split $sp --seed $sd --val --tag $tag $args --device cuda:$gpu > logs/${tag}_${ds}_${sp}_s${sd}.log 2>&1"
  done
done | xargs -P "$P" -I{} bash -c "{}"
