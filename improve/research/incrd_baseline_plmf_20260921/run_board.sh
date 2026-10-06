#!/bin/bash
# usage: ./run_board.sh <tag> [extra args...]   -> all 4 boards (MDR/MDS x 5cv seed42 + single seeds 0-4)
# MDR on cuda:0, MDS on cuda:1 in parallel.
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
RUNNER=${RUNNER:-run_incrd_plmf.py}
cd "$(dirname "$0")"
TAG=$1; shift
for ds in MDR MDS; do
  ( dev=cuda:0; [ $ds = MDS ] && dev=cuda:1
    $PY $RUNNER --dataset $ds --split 5cv --seed 42 --tag $TAG --device $dev "$@"
    for s in 0 1 2 3 4; do $PY $RUNNER --dataset $ds --split single --seed $s --tag $TAG --device $dev "$@"; done
  ) > logs/${TAG}_${ds}.log 2>&1 &
done
wait
grep -h "^##" logs/${TAG}_MDR.log logs/${TAG}_MDS.log
