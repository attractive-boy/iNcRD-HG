#!/bin/bash
cd "$(dirname "$0")"
export RUNNER=run_v3.py OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
while pgrep -f sweep4.sh > /dev/null; do sleep 30; done
run() { tag=$1; shift; echo "### $tag $*"; ./run_board.sh $tag "$@"; }
run plain_refit --val --refit
run ctx33_knn20_refit --ctx33 --knn 20 --val --refit
run ctx33_knn20_neg1_refit --ctx33 --knn 20 --neg_mult 1 --val --refit
run ctx33_late_knn20_refit --ctx33 --ctx_mode late --knn 20 --val --refit
run ctx33_knn20_morgan_refit --ctx33 --knn 20 --drugfeat morgan --val --refit
run ctx33_knn20_shuf_refit --ctx33 --knn 20 --ctx_shuffle --val --refit
