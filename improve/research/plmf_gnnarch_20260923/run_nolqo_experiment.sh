#!/bin/bash
# iNcRD-HG GNN modules WITHOUT LQO: does RT-NCN / prov / side help on their own?
# Same recipe as run_gnn.py --lqo 5 runs, just no --lqo (K=1, message graph = all training positives).
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_gnnarch_20260923
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
run() { local tag="$1"; local flags="$2"; local dev="$3"; local ds="$4"
  $PY run_gnn.py --dataset $ds --split 5cv --seed 42 --val --device $dev --tag ${tag} $flags \
    > logs/nolqo_${tag}_${ds}.log 2>&1
}
i=0
for ds in MDR MDS; do
  dev=$([ $((i % 2)) -eq 0 ] && echo cuda:0 || echo cuda:1); i=$((i+1))
  run cnt_nolqo        "--cnt"                 $dev $ds &
  dev=$([ $((i % 2)) -eq 0 ] && echo cuda:0 || echo cuda:1); i=$((i+1))
  run ncn_nolqo        "--ncn"                 $dev $ds &
  dev=$([ $((i % 2)) -eq 0 ] && echo cuda:0 || echo cuda:1); i=$((i+1))
  run pair_nolqo       "--cnt --ncn"           $dev $ds &
  dev=$([ $((i % 2)) -eq 0 ] && echo cuda:0 || echo cuda:1); i=$((i+1))
  run pair_prov_nolqo  "--cnt --ncn --prov"    $dev $ds &
  dev=$([ $((i % 2)) -eq 0 ] && echo cuda:0 || echo cuda:1); i=$((i+1))
  run full_nolqo       "--cnt --ncn --prov --side" $dev $ds &
done
wait
echo "ALL DONE"
