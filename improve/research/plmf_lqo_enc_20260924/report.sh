#!/bin/bash
# paired valsel tables (each rung vs the previous one; Δ, fold wins, seed wins)
cd "$(dirname "$0")"; PY=/usr/bin/python3
for sp in 5cv single; do
  echo "===== $sp : E1–E4 ladder on the LQO full model"; $PY summarize.py --split $sp --ladder full,full_E1,full_E12,full_E123,full_E1234
  echo "===== $sp : E1–E4 on lower LQO rungs";          $PY summarize.py --split $sp --ref lqo  --tags lqo,lqo_E1234
  $PY summarize.py --split $sp --ref pair --tags pair,pair_E1234
  [ -n "$(ls results | grep full_E1234_no)" ] && { echo "===== $sp : removal"; $PY summarize.py --split $sp --ref full_E1234 --tags full_E1234,full_E1234_noCKA,full_E1234_noNBFd,full_E1234_noNBFm,full_E1234_noJK; }
done
