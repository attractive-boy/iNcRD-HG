#!/bin/bash
cd "$(dirname "$0")"
run() { tag=$1; shift; echo "### $tag $*"; ./run_board.sh $tag "$@"; }
run plain_val --val
run morgan --drugfeat morgan
run kmer_expr --feat kmer_expr
run onehot --feat onehot --drugfeat onehot
run knn0 --knn 0
run knn20 --knn 20
run skip --skip
run drop3 --dropout 0.3
run wd1e-4 --wd 1e-4
run adam1e-3 --opt adam --lr 1e-3
run layers3 --layers 3
run neg1 --neg_mult 1
run neg3 --neg_mult 3
run focal --loss focal
run hid128 --hidden 128
run aggr_sum --aggr sum
