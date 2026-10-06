#!/bin/bash
# Extension: reproduce the paper's Supp. Table 1 for positive:negative = 1:5 and 1:10.
# Positives identical to the author 5-fold; negatives sampled (see make_ratio_splits.py).
set -u
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd "$(dirname "$0")"

# ---- worker A (cuda:0): graph models ----
(
  export DLST_DEV=cuda:0 PYTHONUNBUFFERED=1
  for r in 5 10; do
    export DLST_PROC="ratios/1to$r" DLST_TAG="_1to$r"
    echo "### graphdta 1to$r start $(date)"
    $PY run_bench.py --model graphdta --epochs 40 --seed 42 --out results/ratio_1to${r}_graphdta.csv
    echo "### submdta 1to$r start $(date)"
    $PY run_bench.py --model submdta --epochs 40 --seed 42 --out results/ratio_1to${r}_submdta.csv
  done
  echo "### workerA done $(date)"
) > logs/ratio_workerA.stdout 2>&1 &
A=$!

# ---- worker B (cuda:1): gcnnma + daestb ----
(
  export DLST_DEV=cuda:1 PYTHONUNBUFFERED=1
  for r in 5 10; do
    export DLST_PROC="ratios/1to$r" DLST_TAG="_1to$r"
    echo "### gcnnma 1to$r start $(date)"
    $PY gcnnma.py --epochs 60 --seed 42 --out results/ratio_1to${r}_gcnnma.csv
    echo "### daestb 1to$r start $(date)"
    $PY daestb_foldwise.py --seed 42 --out results/ratio_1to${r}_daestb.csv
  done
  echo "### workerB done $(date)"
) > logs/ratio_workerB.stdout 2>&1 &
B=$!

# ---- worker C (both GPUs): ML-DTI fold-parallel ----
(
  export PYTHONUNBUFFERED=1
  for r in 5 10; do
    export DLST_PROC="ratios/1to$r" DLST_TAG="_1to$r"
    echo "### mldti 1to$r start $(date)"
    for f in 0 1 2 3 4; do
      DLST_DEV=cuda:$(( f % 2 )) $PY run_bench.py --model mldti --epochs 40 --seed 42 \
          --folds $f --out results/ratio_1to${r}_mldti_fold$f.csv > logs/ratio_1to${r}_mldti_f$f.stdout 2>&1 &
    done
    wait
    $PY - <<PYEOF
import csv
r="$r"
fields=['dataset','method','fold','seed','auc','aupr','acc','precision','recall','specificity','f1','mcc','best_epoch','last_auc','last_aupr']
rows=[]
for f in range(5):
    rows += list(csv.DictReader(open(f'results/ratio_1to{r}_mldti_fold{f}.csv')))
with open(f'results/ratio_1to{r}_mldti.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)
print('merged mldti 1to%s'%r)
PYEOF
  done
  echo "### workerC done $(date)"
) > logs/ratio_workerC.stdout 2>&1 &
C=$!

echo "ratio workers A=$A B=$B C=$C"
wait $A $B $C
echo "RATIOS ALL DONE $(date)"
