#!/bin/bash
cd "$(dirname "$0")"
export RUNNER=run_v3.py OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# wait for sweep3 to finish
while pgrep -f sweep3.sh > /dev/null; do sleep 30; done
run() { tag=$1; shift; echo "### $tag $*"; ./run_board.sh $tag "$@"; }
run ctx33_late --ctx33 --ctx_mode late
run ctx33_late_val --ctx33 --ctx_mode late --val
run ctx33_late_shuf --ctx33 --ctx_mode late --ctx_shuffle
run ctx33_neg1 --ctx33 --neg_mult 1
run ctx33_late_neg1 --ctx33 --ctx_mode late --neg_mult 1
run ctx33_late_morgan --ctx33 --ctx_mode late --drugfeat morgan
run ctx33_late_neg1_val --ctx33 --ctx_mode late --neg_mult 1 --val
