#!/bin/bash
cd "$(dirname "$0")"
export RUNNER=run_v4.py OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
run() { tag=$1; shift; echo "### $tag $*"; ./run_board.sh $tag "$@"; }
run ctx33_knn20_bpr_refit --ctx33 --knn 20 --loss bpr --val --refit
run ctx33_knn20_bcebpr_refit --ctx33 --knn 20 --loss bce_bpr --val --refit
run ctx33_knn20_selaupr_refit --ctx33 --knn 20 --val --sel aupr --refit
run plain_bcebpr_refit --loss bce_bpr --val --refit
