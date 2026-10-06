#!/bin/bash
# Cold-start (unseen miRNA / unseen drug) reproduction for the 5 baselines.
set -u
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd "$(dirname "$0")"

# ---- GPU0: GraphDTA + SubMDTA ----
(
  export DLST_DEV=cuda:0 PYTHONUNBUFFERED=1
  for s in mirna drug; do
    export DLST_PROC="coldstart/$s" DLST_TAG="_cold_$s"
    echo "### graphdta cold_$s start $(date)"
    $PY run_bench.py --model graphdta --epochs 40 --seed 42 --out results/coldstart_${s}_graphdta.csv
    echo "### submdta cold_$s start $(date)"
    $PY run_bench.py --model submdta --epochs 40 --seed 42 --out results/coldstart_${s}_submdta.csv
  done
  echo "### cold GPU0 done $(date)"
) > logs/cold_worker0.stdout 2>&1 &

# ---- GPU1: GCNNMMA + DAESTB ----
(
  export DLST_DEV=cuda:1 PYTHONUNBUFFERED=1
  for s in mirna drug; do
    export DLST_PROC="coldstart/$s" DLST_TAG="_cold_$s"
    echo "### gcnnma cold_$s start $(date)"
    $PY gcnnma.py --epochs 60 --seed 42 --out results/coldstart_${s}_gcnnma.csv
    echo "### daestb cold_$s start $(date)"
    $PY daestb_foldwise.py --seed 42 --out results/coldstart_${s}_daestb.csv
  done
  echo "### cold GPU1 done $(date)"
) > logs/cold_worker1.stdout 2>&1 &

# ---- ML-DTI fold-parallel across both GPUs ----
(
  export PYTHONUNBUFFERED=1
  for s in mirna drug; do
    export DLST_PROC="coldstart/$s" DLST_TAG="_cold_$s"
    echo "### mldti cold_$s start $(date)"
    for f in 0 1 2 3 4; do
      DLST_DEV=cuda:$(( f % 2 )) $PY run_bench.py --model mldti --epochs 40 --seed 42 \
          --folds $f --out results/coldstart_${s}_mldti_fold$f.csv > logs/coldstart_${s}_mldti_f$f.stdout 2>&1 &
    done
    wait
    $PY - <<PYEOF
import csv
s="$s"
fields=['dataset','method','fold','seed','auc','aupr','acc','precision','recall','specificity','f1','mcc','best_epoch','last_auc','last_aupr']
rows=[]
for f in range(5):
    rows += list(csv.DictReader(open(f'results/coldstart_{s}_mldti_fold{f}.csv')))
with open(f'results/coldstart_{s}_mldti.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)
print('merged mldti cold_%s'%s)
PYEOF
  done
  echo "### cold MLDTI done $(date)"
) > logs/cold_worker2.stdout 2>&1 &

wait
echo "COLDSTART ALL DONE $(date)"
