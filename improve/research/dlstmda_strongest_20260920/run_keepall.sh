#!/bin/bash
# DLMVF 论文口径: 作者原始 5 折, 测试行全部打分(不做实体级剔除), 4 个推理种子
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/dlstmda_strongest_20260920
V=/data1/licheng/incrd-hg/.venv-tabicl/bin/python
export PYTHONUNBUFFERED=1
for s in 20260914 666 0 1; do
  echo "### seed $s start $(date)"
  $V run_tabicl.py --device cuda:1 --n-estimators 32 --seed $s --keep-all --out results/strongest_keepall_seed$s.csv
  echo "### seed $s done $(date)"
done
echo "KEEPALL DONE $(date)"
