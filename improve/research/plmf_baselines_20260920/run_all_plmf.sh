#!/bin/bash
# Reproduce PLMF-MDA Fig 3 baselines on MDR/MDS.
#   protocol "5cv"    : paper's stated 5-fold CV (seed 42)
#   protocol "single" : released-code 80/20 split, seeds 0..4
# Two GPU workers.
set -u
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
cd "$(dirname "$0")"
mkdir -p logs results

PROTO=${1:-5cv}
case "$PROTO" in
  5cv)    SEEDS="42"; FOLDS="0,1,2,3,4" ;;
  single) SEEDS="0 1 2 3 4"; FOLDS="0" ;;
  *) echo "usage: $0 [5cv|single]"; exit 1 ;;
esac

# ---- worker A (GPU0): GraphDTA + SubMDTA ----
(
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
  for ds in MDR MDS; do
    for s in $SEEDS; do
      for m in graphdta submdta; do
        echo "### $m $ds $PROTO seed=$s start $(date)"
        $PY run_bench_plmf.py --model "$m" --dataset "$ds" --split "$PROTO" \
            --seed "$s" --folds "$FOLDS" --epochs 150
        echo "### $m $ds $PROTO seed=$s done $(date)"
      done
    done
  done
) > logs/plmfA_${PROTO}.stdout 2>&1 &
PID_A=$!

# ---- worker B (GPU1): ML-DTI + GCNNMMA ----
(
  export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
  for ds in MDR MDS; do
    for s in $SEEDS; do
      echo "### mldti $ds $PROTO seed=$s start $(date)"
      $PY run_bench_plmf.py --model mldti --dataset "$ds" --split "$PROTO" \
          --seed "$s" --folds "$FOLDS" --epochs 150
      echo "### gcnnma $ds $PROTO seed=$s start $(date)"
      $PY gcnnma_plmf.py --dataset "$ds" --split "$PROTO" --seed "$s" \
          --folds "$FOLDS" --epochs 100
      echo "### mldti+gcnnma $ds $PROTO seed=$s done $(date)"
    done
  done
) > logs/plmfB_${PROTO}.stdout 2>&1 &
PID_B=$!

echo "workerA pid=$PID_A  workerB pid=$PID_B  proto=$PROTO"
wait $PID_A $PID_B
echo "ALL DONE $PROTO $(date)"
