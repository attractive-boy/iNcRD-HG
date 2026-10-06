#!/bin/bash
BASE=/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919
REPO=$BASE/repos/PLMF-MDA-main
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python
PP=$REPO/code:$BASE/venvs/plmf_site
cd $REPO || exit 1

for s in 1 2 3 4; do
  echo "===== seed $s start $(date) ====="
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PP $PY -u code/train.py \
    --pos_sample data/MDR/pos_MDA_R.edgelist --neg_sample data/MDR/neg_MDA_R.edgelist \
    --mirna_file data/MDR/miRNA_ID_R.xlsx --drug_file data/MDR/drug_ID_R.xlsx \
    --seed $s > $BASE/logs/plmf_mdr_seed$s.log 2>&1 &
  p0=$!
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PP $PY -u code/train.py \
    --pos_sample data/MDS/pos_MDA_S.edgelist --neg_sample data/MDS/neg_MDA_S.edgelist \
    --mirna_file data/MDS/miRNA_ID_S.xlsx --drug_file data/MDS/drug_ID_S.xlsx \
    --seed $s > $BASE/logs/plmf_mds_seed$s.log 2>&1 &
  p1=$!
  wait $p0 $p1
  echo "MDR seed$s: $(grep 'Final Test Result' $BASE/logs/plmf_mdr_seed$s.log)"
  echo "MDS seed$s: $(grep 'Final Test Result' $BASE/logs/plmf_mds_seed$s.log)"
done
echo "ALL DONE $(date)"
