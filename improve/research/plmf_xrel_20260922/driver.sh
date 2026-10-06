#!/bin/bash
cd /data1/licheng/incrd-hg/iNcRD-HG/improve/research/plmf_xrel_20260922
sed -i 's/OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4/OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3/' run_one.sh
cat grid_seeds.txt | xargs -P 8 -L 1 ./run_one.sh
cat grid_study1.txt | xargs -P 4 -L 1 ./run_study.sh
cat grid_study2.txt | xargs -P 8 -L 1 ./run_study.sh
echo DRIVER_DONE
