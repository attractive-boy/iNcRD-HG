#!/bin/bash
cd "$(dirname "$0")"
export RUNNER=run_v2_tmp.py
run() { tag=$1; shift; echo "### $tag $*"; ./run_board.sh $tag "$@"; }
run ens5 --ens 5
run ctx --ctx
run ctx_val --ctx --val
run ens5_val --ens 5 --val --snap 100
run ctx_ens5_val --ctx --ens 5 --val --snap 100
run lrdecay --lr_decay 0.5
run ctx_neg1 --ctx --neg_mult 1
run ctx_skip --ctx --skip
