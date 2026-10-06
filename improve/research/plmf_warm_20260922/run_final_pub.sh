#!/bin/bash
# wait for the PubMed miRNA fetch to cover >=95% of names, then run the final_pub variant
export OMP_NUM_THREADS=6; PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
while true; do
  n=$($PY -c "import pandas as pd; p=pd.read_csv('data/pubmed_mirna_plmf.csv', on_bad_lines='skip'); print(p['name'].nunique())")
  if [ "$n" -ge 1560 ]; then break; fi
  sleep 120
done
echo "coverage reached: $n names"
for s in 42 0 1 2; do for ds in MDR MDS; do $PY final.py --dataset $ds --split 5cv --seed $s --inner 10 --variant final_pub; done; done
for s in 0 1 2 3 4; do for ds in MDR MDS; do $PY final.py --dataset $ds --split single --seed $s --inner 10 --variant final_pub; done; done
$PY summarize_final.py > results/final/SUMMARY_TABLES.txt 2>&1
echo ALLDONE
