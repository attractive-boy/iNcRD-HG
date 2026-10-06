#!/bin/bash
# Reproduce this round (plmf_xrel_20260922). PY = ncRNAdrug conda env; features cached from plmf_warm_20260922/cache.
PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python; export OMP_NUM_THREADS=4
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_xrel_20260922
# 0. external resources (already fetched): data_external/ncrnadrug_20260922/{DR_Curated,DR_GEO,DT_GEO,DT_Curated,...}.txt + gse_pmid.json,
#    data_external/mirbase_20260922/hsa.gff3, data/drug_synonyms.json (PubChem), GDSC2 re-download
$PY fetch_drug_synonyms.py; $PY fetch_drug_synonyms2.py            # PubChem synonyms for CTD/LINCS/GDSC name matching
$PY build_full_records.py; $PY build_full_records.py --genes        # provenance completion: pair -> PMIDs/context from full ncRNADrug
for ds in MDR MDS; do $PY xrel.py --dataset $ds --groups dz,gene,cerna,cluster,pgx,geo,geoall,dt,dtall; done   # external-relation caches
# 1. ablations on the paper 5-fold protocol (seeds 42,0,1,2) and the official 80/20 split (seeds 0-4)
for s in 42 0 1 2; do for ds in MDR MDS; do for v in base prov mpr phl provf provfs phlf phlfs phlf_cluster phlf_geo phlf_geo_shuf phlf_x phlf_x_shuf; do
  $PY final2.py --dataset $ds --split 5cv --seed $s --variant $v; done; done; done
for s in 0 1 2 3 4; do for ds in MDR MDS; do for v in base phl phlf phlf_cluster; do $PY final2.py --dataset $ds --split single --seed $s --variant $v; done; done; done
# 2. unlabeled-pool training (+x10 negatives) and the final stack
for s in 42 0 1 2; do for ds in MDR MDS; do $PY probe_extraneg.py --dataset $ds --seed $s --mult 10 --prov full; $PY probe_extraneg.py --dataset $ds --seed $s --mult 10 --prov full --xrel cluster,geo; done; done
for s in 0 1 2 3 4; do for ds in MDR MDS; do $PY probe_extraneg.py --dataset $ds --split single --seed $s --mult 10 --prov full --xrel cluster,geo; done; done
# 3. study-grouped splits (primary-study groups from the full records) and diagnostics
for s in 42 0 1; do for ds in MDR MDS; do for v in base phl provf phlf phlfs phlf_geo; do $PY study_split2.py --dataset $ds --seed $s --variant $v --records full --group primary; done; done; done
$PY summarize.py base; $PY summarize.py phlf; $PY summarize_study.py phlf; $PY gain_analysis.py base phl provf phlf phlf_extraneg10 phlf_cluster-geo_extraneg10
$PY error_after_phl.py; $PY probe_univariate.py MDR; $PY probe_raredrug.py MDR 10; $PY probe_gate.py --dataset MDR --diag
