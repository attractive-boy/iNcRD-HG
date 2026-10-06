#!/bin/bash
# Wait (by PID only, no pgrep patterns) for the 20 isolated-eval jobs, then rebuild summary/Excel/report.
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921
PIDS="3234297 3234298 3234301 3234303 3234305 3234307 3234309 3234311 3234312 3234313 3255418 3255420 3255422 3255424 3255426 3255428 3255430 3255432 3255433 3255434"
for p in $PIDS; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
V=/data1/licheng/incrd-hg/.venv-dgl/bin/python
{
  echo "all iso jobs finished at $(date)"
  echo "iso csv count: $(ls results/mphgnn_kmer_val_iso_*.csv 2>/dev/null | grep -v curves | wc -l) (expect 20)"
  $V summarize.py && $V build_excel.py && $V write_report.py && echo FINALIZE_OK
} > logs/finalize.log 2>&1
