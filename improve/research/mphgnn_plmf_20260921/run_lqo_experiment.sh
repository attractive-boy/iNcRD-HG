#!/bin/bash
# LQO on MPHGNN: baseline (no LQO) vs leave-query-out, 5cv seed42, no-leakage protocol
# (val_frac 0.1 + global_graph train).  4 jobs over the two A40s.
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921
PY=/data1/licheng/incrd-hg/.venv-dgl/bin/python
COMMON="--split 5cv --seed 42 --val_frac 0.1 --global_graph train"

$PY run_mphgnn_plmf_val.py --dataset MDR $COMMON --device cuda:0 > logs/lqo_base_MDR.log 2>&1 &
$PY run_mphgnn_plmf_lqo.py --dataset MDS $COMMON --lqo 5 --device cuda:0 > logs/lqo5_MDS.log 2>&1 &
$PY run_mphgnn_plmf_val.py --dataset MDS $COMMON --device cuda:1 > logs/lqo_base_MDS.log 2>&1 &
$PY run_mphgnn_plmf_lqo.py --dataset MDR $COMMON --lqo 5 --device cuda:1 > logs/lqo5_MDR.log 2>&1 &
wait
echo "ALL DONE"
