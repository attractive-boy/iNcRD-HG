#!/bin/bash
cd "$(dirname "$0")"
export RUNNER=run_v3.py OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while pgrep -f sweep5.sh > /dev/null; do sleep 30; done
run() { tag=$1; shift; echo "### $tag $*"; ./run_board.sh $tag "$@"; }
run ctx33_knn20_e500_refit --ctx33 --knn 20 --epochs 500 --val --refit
run ctx33_knn20_neg1_e500_refit --ctx33 --knn 20 --neg_mult 1 --epochs 500 --val --refit
run plain_e500_refit --epochs 500 --val --refit
