#!/usr/bin/env bash
# Pilot grid: plain vs M1 vs M2M3 (strict, no M1) vs full (M1+M2+M3) on the iNcRD-HG own MiDrug task.
# official = released seed-666 split; rs = re-split 60/20/20 with seeds 0-4. 300 epochs, valsel epoch selection.
set -u
cd "$(dirname "$0")"
PY=${PY:-/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python}
export OMP_NUM_THREADS=4
mkdir -p logs results
GRID=${1:-grid_final.txt}
echo "grid: $GRID"
while IFS='|' read -r tag split seed mseed flags; do
  [ -z "$tag" ] && continue
  case "$tag" in \#*) continue;; esac
  out="results/${tag}_${split}_s${seed}_m${mseed}.csv"
  if [ -f "$out" ]; then echo "skip $out"; continue; fi
  echo "== $tag $split s$seed m$mseed $flags"
  $PY run_own.py --tag "$tag" --split "$split" --seed "$seed" --model_seed "$mseed" --epochs 300 $flags \
      > "logs/${tag}_${split}_s${seed}_m${mseed}.log" 2>&1 || echo "FAILED $tag $split s$seed m$mseed"
done < "$GRID"
echo DONE
