#!/bin/bash
# usage: ./run_grid.sh grid.txt [jobs_per_gpu]
#   grid line "tag|args"; runs every tag on MDR/MDS x (5cv seed 42/43/44 + official 80/20 seed 0-4),
#   jobs alternate between cuda:0 and cuda:1 (total parallel = 2 x jobs_per_gpu); finished results are skipped,
#   so the script can be re-run after an interruption.
GRID=${1:-grid.txt}; J=${2:-3}
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python; export OMP_NUM_THREADS=4
cd "$(dirname "$0")"; mkdir -p results logs
SPECS="MDR:5cv:42 MDR:5cv:43 MDR:5cv:44 MDS:5cv:42 MDS:5cv:43 MDS:5cv:44"
for ds in MDR MDS; do for s in 0 1 2 3 4; do SPECS="$SPECS $ds:single:$s"; done; done
i=0
grep -v '^#' "$GRID" | grep -v '^\s*$' | while IFS='|' read -r tag args; do
  for spec in $SPECS; do
    IFS=':' read -r ds sp sd <<< "$spec"
    gpu=$((i % 2)); i=$((i+1))
    [ -f "results/${tag}_${ds}_${sp}_seed${sd}.csv" ] && continue
    echo "$PY run_le.py --dataset $ds --split $sp --seed $sd --val --tag $tag $args --device cuda:$gpu > logs/${tag}_${ds}_${sp}_s${sd}.log 2>&1 || echo FAILED ${tag}_${ds}_${sp}_s${sd}"
  done
done | xargs -P $((2 * J)) -I{} bash -c "{}"
echo "done: $(ls results/*_seed*.csv 2>/dev/null | grep -vc '_scores.csv') result files"
