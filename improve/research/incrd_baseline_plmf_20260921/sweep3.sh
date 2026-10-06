#!/bin/bash
cd "$(dirname "$0")"
export RUNNER=run_v3.py
run() { tag=$1; shift; echo "### $tag $*"; ./run_board.sh $tag "$@"; }
run ctx33 --ctx33
run ctx33_shuf --ctx33 --ctx_shuffle
run ctx33_val --ctx33 --val
run ctx33_morgan --ctx33 --drugfeat morgan
run ctx33_knn20 --ctx33 --knn 20
run ctx33_ens5_val --ctx33 --ens 5 --val --snap 100
run ctx33_lrdecay_val --ctx33 --lr_decay 0.5 --val
run ctx33_skip --ctx33 --skip
